"""Session registry — persistent tracking for all agent sessions.

Every session (manager or agent) is tracked here. Sessions persist
across restarts via state.json files in per-session directories.
Each session wraps a brain-agnostic session adapter, built through
``Brain.make_session()`` (#485) — a Claude SDK client, a per-turn ``codex
exec`` subprocess, or the stub, depending on the configured brain. Nothing
in this module is Claude-specific.

All session state lives under a selected Bobi Agent runtime root:
``<BOBI_HOME>/agents/<name>/run/state/sessions/``. Every process binds
that root explicitly — the manager through ``bobi agent <name> start``,
children via the ``root`` their spawner passes in the args blob. Nothing
resolves runtime identity from cwd.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, asdict, replace
from pathlib import Path
from typing import Any

from bobi import paths
from bobi.fsutil import atomic_write_json

log = logging.getLogger(__name__)

# Terminal-status vocabulary (MDS-65 D1). A run ends in exactly one of these.
# ``done`` is retained as a backward-compatible alias for reading old records —
# new code writes the honest vocabulary so a failed/crashed run is never
# recorded as a success.
TERMINAL_COMPLETED = "completed"
TERMINAL_FAILED = "failed"
TERMINAL_CRASHED = "crashed"
# Statuses that mean "still running" — everything else is terminal/inactive.
ACTIVE_STATUSES = ("starting", "running", "idle")
# Honest-failure terminal statuses (used by the reconciler / delivery routing).
FAILED_STATUSES = (TERMINAL_FAILED, TERMINAL_CRASHED)
# Honest terminal vocabulary (``done`` kept as a legacy read alias).
TERMINAL_STATUSES = (TERMINAL_COMPLETED, TERMINAL_FAILED, TERMINAL_CRASHED, "done")
# What a run that vanished without reporting a terminal status is recorded as.
# Shared so the two places that close one - a list read's reap here, and launch
# admission's close in bobi/reconcile.py - cannot drift apart.
DIED_WITHOUT_TERMINAL = "agent process died without reporting a terminal status"
# A session in any of these has torn down its inbox/subscription — publishing to
# it would succeed but no one would consume it (inbox.py guard).
DEAD_STATUSES = (
    "stopped", "error", "cancelled", "done",
    TERMINAL_COMPLETED, TERMINAL_FAILED, TERMINAL_CRASHED,
)


def compute_manifest_hash(project_path: Path | None = None) -> str:
    """Compute a stable hash of the install-manifest.json file list.

    Returns the hex digest of the sorted file-hash entries, or "" if
    no manifest exists.  Used to detect when the installed agent image
    has changed so sessions can be rotated.
    """
    import hashlib

    manifest = paths.install_manifest_path(project_path)
    try:
        data = json.loads(manifest.read_text())
        files = data.get("files", {})
    except (json.JSONDecodeError, OSError):
        return ""
    if not files:
        return ""
    content = "".join(f"{k}:{v}" for k, v in sorted(files.items()))
    return hashlib.sha256(content.encode()).hexdigest()


def check_image_rotation(session_name: str, project_path: Path) -> bool:
    """Clear a session if the installed image has changed since it was stamped.

    Returns True if the session was rotated.  Safe no-ops: no manifest,
    no prior session, no stored hash (first run after upgrade).
    """
    current_hash = compute_manifest_hash(project_path)
    if not current_hash:
        return False
    saved_id = load_session_id(session_name, root=project_path)
    if not saved_id:
        return False
    registry = SessionRegistry(project_path)
    entry = registry.get(session_name)
    if not entry or not entry.image_hash or entry.image_hash == current_hash:
        return False
    log.info("Installed image changed — rotating session for %s", session_name)
    save_session_id(session_name, "", root=project_path)
    return True


def set_project_root(path: Path) -> None:
    """Bind the installation root (delegates to bobi.paths)."""
    paths.bind_root(path)


def get_project_root() -> Path | None:
    """The bound installation root, or None (delegates to bobi.paths)."""
    return paths.bound_root()


def pid_alive(pid: int) -> bool:
    """Whether a pid refers to a live process (signal-0 probe)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_int_file(path: Path) -> int:
    """Read a single-integer state file — a pid, a bound port — as an int.

    Returns 0 when the file is missing, unreadable, or does not hold an
    integer, so a caller can treat "no usable value" as one case.
    """
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return 0


def read_pid(pid_path: Path) -> int:
    """Read a pid file, returning 0 when missing or malformed."""
    return read_int_file(pid_path)


def _pid_file_alive(pid_path: Path) -> bool:
    """Check whether a manager.pid file points to a running process."""
    return pid_alive(read_pid(pid_path))


@contextmanager
def _state_file_lock(path: Path) -> Iterator[None]:
    """Serialize whole-document state changes across processes.

    The lock is held only while the document it guards exists. Every writer
    here re-checks *path* under the lock and no-ops when the entry has already
    vanished, so on that path the lock file is the only thing the call would
    leave behind - inside a session directory that is being removed. A file
    created after ``shutil.rmtree`` has scandir'd the directory makes the
    final ``os.rmdir`` raise ``ENOTEMPTY``, so unlink it while still holding
    it: nothing can be mid-write on a document that is already gone.
    """
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "a+") as lock_file:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if not path.exists():
                lock_path.unlink(missing_ok=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def find_runtime_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* to find the nearest live runtime root.

    This is a liveness DETECTOR (used by the nested-start guard), not
    config resolution — config resolution is paths.resolve_root(). Returns
    the directory containing the live manager pid, or None if no ancestor
    has one. The search starts at *start* (defaulting to the bound root)
    and stops at the filesystem root.
    """
    current = (start or paths.bound_root())
    if current is None:
        return None
    current = current.resolve()
    while True:
        if _pid_file_alive(paths.manager_pid_path(current)):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _sessions_dir(root: Path | None = None) -> Path:
    """Sessions directory under *root*, else the bound installation root.

    In-team processes (manager at start, children via the spawn args'
    `root`) bind their root once and pass None; multi-team hosts like the
    webapp pass the target team's root explicitly per call.
    """
    return paths.sessions_dir(root)


def session_handoff_path(name: str, step: str, *,
                         root: Path | None = None) -> Path:
    """A session's handoff file for one workflow step."""
    return _sessions_dir(root) / name / f"handoff-{step}.yaml"


def session_log_path(name: str, *, root: Path | None = None) -> Path:
    """A session's activity log, parent dir created."""
    p = _sessions_dir(root) / name / "log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _has_tokens(entry: Any) -> bool:
    """Whether a ``model_usage`` entry carries recorded token volume."""
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("input_tokens") or entry.get("output_tokens"))


def _same_model(a: str, b: str) -> bool:
    """Whether two model ids name the same physical model.

    The live cost writer keys ``model_usage`` by the SDK's own dict key, which
    can carry a context-variant marker (``claude-opus-4-8[1m]``) or a dated
    snapshot suffix (``claude-haiku-4-5-20251001``) that a recovered
    transcript's bare ``message.model`` does not. Matching on the base id keeps
    one model on one row. The prefix arm only accepts a match on a ``-``
    boundary, so a genuinely different model is never collapsed in.
    """
    a, b = a.split("[", 1)[0], b.split("[", 1)[0]
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if not short or not long.startswith(short + "-"):
        return False
    # Only a YYYYMMDD snapshot suffix marks the same model. Any digits at all
    # is too loose: it would read claude-sonnet-4-5 as a snapshot of
    # claude-sonnet-4 and merge two different models onto one row.
    suffix = long[len(short) + 1:]
    return len(suffix) == 8 and suffix.isdigit()


def _token_total(usage: dict) -> int:
    """Total token volume recorded across a ``model_usage`` map."""
    return sum(
        _tok(e.get("input_tokens")) + _tok(e.get("output_tokens"))
        for e in usage.values() if isinstance(e, dict)
    )


def _tok(value: Any) -> int:
    """A token count usable in arithmetic (bool is an int subclass)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _matching_usage_key(usage: dict, provider: str, model: str) -> str:
    """The ``model_usage`` key recovered usage for *model* belongs under.

    Reuses an existing key for the same physical model rather than adding a
    second one: splitting a model across two keys would double its rows in the
    cost fold forever, and :func:`_has_tokens` would then report the session
    populated, making the split unrepairable by a re-run.
    """
    key = f"{provider}:{model}" if provider else model
    if key in usage:
        return key
    for existing in usage:
        known = existing.split(":", 1)[1] if ":" in existing else existing
        if _same_model(known, model):
            return existing
    return key


@dataclass
class SessionEntry:
    name: str
    session_id: str = ""
    role: str = ""
    run_key: str = ""
    title: str = ""
    phase: str = ""
    project: str = ""
    cwd: str = ""
    status: str = "starting"
    pid: int = 0
    image_hash: str = ""
    model: str = ""
    provider: str = ""
    total_cost_usd: float = 0.0
    # model_usage: "provider:model" -> {cost_usd, input_tokens, output_tokens,
    # cached_input_tokens}. The presence of the cached_input_tokens key marks
    # an entry whose cached tokens are split out per #760 and licenses
    # fold-time dollar estimation (bobi.costs.rollup_costs).
    #
    # Two writers, with different rules about that key:
    #   record_cost      - the live path. Never adds the key to an entry that
    #                      lacks it: one post-upgrade turn on a straddling
    #                      session would otherwise mark the whole merged entry
    #                      priceable and bill its inflated legacy input.
    #   backfill_usage   - the #935 repair path. Writes the key, because it
    #                      replaces zeroed counters wholesale with a genuine
    #                      split recovered from the transcript, and only ever
    #                      touches entries carrying no token volume at all.
    model_usage: dict = field(default_factory=dict)
    rotation_count: int = 0
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    requested_by: dict = field(default_factory=dict)
    # Honest terminal status + reconciler backstop (MDS-65).
    # error: terminal failure message; "" on success/while running.
    error: str = ""
    # terminal_at: when a terminal status was durably written (0.0 = not terminal).
    terminal_at: float = 0.0
    # emit_confirmed: whether the terminal lifecycle bus POST is known to have
    # landed. The reconciler re-emits terminal-but-unconfirmed runs.
    emit_confirmed: bool = False
    # timeout: the run's declared/effective timeout in seconds, persisted at
    # register so the dead-man reconciler knows each run's deadline.
    timeout: int = 0
    # reconciled_at: set when the reconciler closed a stranded run (idempotency).
    reconciled_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEntry":
        """Build from a persisted dict, ignoring fields no longer in the schema.

        State files written by older code may carry retired keys (e.g.
        ``inbox_port`` before the inbox HTTP transport was removed). Filtering
        unknown keys keeps stale sessions readable across upgrades instead of
        raising and silently dropping them.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class SessionRegistry:
    """Directory-per-session registry.

    Each session gets a directory at <run>/state/sessions/<name>/
    containing state.json, handoff-<step>.yaml files, and log.jsonl.
    Active sessions have a live pid; completed ones remain for history.

    *root* selects the runtime root; None means the process's bound root
    (in-team processes). Multi-team hosts construct one registry per target
    root instead of mutating process state.
    """

    def __init__(self, root: Path | None = None):
        self._root = root
        _sessions_dir(root)

    def session_dir(self, name: str) -> Path:
        return _sessions_dir(self._root) / name

    def _state_path(self, name: str) -> Path:
        return _sessions_dir(self._root) / name / "state.json"

    @staticmethod
    def _write_state(path: Path, data: dict) -> None:
        """Publish one complete state document for concurrent readers."""
        atomic_write_json(path, data)

    def register(self, entry: SessionEntry) -> None:
        d = self.session_dir(entry.name)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "state.json"
        with _state_file_lock(path):
            self._write_state(path, asdict(entry))

    def update(self, name: str, **kwargs) -> None:
        path = self._state_path(name)
        if not path.exists():
            return
        with _state_file_lock(path):
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, TypeError):
                return
            # Normalize through the dataclass so fields added in a later schema
            # version (e.g. emit_confirmed/terminal_at — MDS-65) are present and can
            # be written. Without this, update() only touches keys already in the
            # on-disk JSON, so an entry written by older code would silently DROP a
            # set to a new field — and the reconciler, unable to persist
            # emit_confirmed=True, would re-emit that completion on every wake.
            data = asdict(SessionEntry.from_dict(data))
            for k, v in kwargs.items():
                if k in data:
                    data[k] = v
            data["last_activity"] = time.time()
            self._write_state(path, data)

    def record_cost(self, name: str, cost_usd: float,
                    model: str = "", provider: str = "",
                    input_tokens: int = 0, output_tokens: int = 0,
                    cached_input_tokens: int = 0) -> None:
        """Accumulate cost and model_usage on a session entry.

        ``cached_input_tokens`` is the cached SUBSET of ``input_tokens``, kept
        split because cache reads bill at a per-model discount. The key is
        written on every FRESH entry (even 0), which marks it as carrying the
        split: the fold-time dollar estimator (#760) refuses to price entries
        without it, since legacy recordings folded cached tokens into
        ``input_tokens`` at full weight. An existing entry that lacks the key
        never gains it - one post-upgrade turn on a straddling session would
        otherwise mark the whole merged entry priceable and bill its inflated
        legacy input at the full rate.
        """
        path = self._state_path(name)
        if not path.exists():
            return
        with _state_file_lock(path):
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, TypeError):
                return
            data["total_cost_usd"] = data.get("total_cost_usd", 0.0) + cost_usd
            if model:
                usage = data.get("model_usage", {})
                key = f"{provider}:{model}" if provider else model
                if key:
                    entry = usage.get(key)
                    if entry is None:
                        entry = {"cost_usd": 0.0, "input_tokens": 0,
                                 "output_tokens": 0, "cached_input_tokens": 0}
                    entry["cost_usd"] = entry.get("cost_usd", 0.0) + cost_usd
                    entry["input_tokens"] = entry.get("input_tokens", 0) + input_tokens
                    entry["output_tokens"] = entry.get("output_tokens", 0) + output_tokens
                    if "cached_input_tokens" in entry:
                        entry["cached_input_tokens"] += cached_input_tokens
                    usage[key] = entry
                    data["model_usage"] = usage
            if model and not data.get("model"):
                data["model"] = model
            if provider and not data.get("provider"):
                data["provider"] = provider
            data["last_activity"] = time.time()
            self._write_state(path, data)

    def backfill_usage(self, name: str, usage_by_model: dict[str, dict],
                       *, provider: str = "anthropic",
                       write: bool = True) -> str:
        """Fill ZEROED or missing token telemetry from recovered usage (#935).

        A repair, not a recording: it never adds dollars and never bumps
        ``last_activity``.

        A recorded fact outranks a reconstructed one: any session already
        carrying token volume is left exactly as it is, which is also what
        makes re-running a no-op. That rule has a real cost, so it is reported
        rather than hidden. The recording fix and this repair ship together, so
        a long-lived agent can book one post-fix turn before an operator runs
        the backfill; its pre-fix history is then visible in the transcript but
        NOT recoverable here, and comes back as ``"partially_recorded"`` so the
        operator can see what was left behind instead of reading a clean
        ``already_populated``. Completing those entries would mean overwriting
        recorded numbers, which this repair deliberately never does.

        ``usage_by_model`` maps a bare model id to
        ``{input_tokens, output_tokens, cached_input_tokens}``. The merge runs
        under the shared state lock so the decision and the write cannot
        straddle a concurrent ``record_cost``. ``write=False`` reaches the same
        verdict against the same locked state and returns without writing, so a
        dry run reports what the real run would do rather than a second
        opinion about it.

        Returns ``"repaired"``, ``"already_populated"``, or ``"missing"``.
        """
        path = self._state_path(name)
        if not path.exists():
            return "missing"
        if True:
            if not path.exists():
                return "missing"
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, TypeError):
                return "missing"
            usage = data.get("model_usage") or {}
            recorded = _token_total(usage)
            if recorded:
                recovered = sum(t.get("input_tokens", 0)
                                + t.get("output_tokens", 0)
                                for t in usage_by_model.values())
                return ("partially_recorded" if recovered > recorded
                        else "already_populated")
            if not write:
                return "repaired"
            for model, tokens in usage_by_model.items():
                key = _matching_usage_key(usage, provider, model)
                entry = usage.get(key) or {"cost_usd": 0.0}
                entry.setdefault("cost_usd", 0.0)
                entry["input_tokens"] = tokens.get("input_tokens", 0)
                entry["output_tokens"] = tokens.get("output_tokens", 0)
                # Written even when zero: the key marks a post-#760 entry whose
                # cached tokens are split out, which is what licenses the fold
                # to price it (bobi.costs.rollup_costs).
                entry["cached_input_tokens"] = tokens.get(
                    "cached_input_tokens", 0)
                usage[key] = entry
            data["model_usage"] = usage
            if provider and not data.get("provider"):
                data["provider"] = provider
            # Only an unambiguous single-model session earns a `model` label.
            if len(usage_by_model) == 1 and not data.get("model"):
                data["model"] = next(iter(usage_by_model))
            self._write_state(path, data)
            return "repaired"

    def mark_done(self, name: str) -> None:
        self.update(name, status="done", pid=0)

    def mark_terminal(self, name: str, status: str, *, error: str = "",
                      session_id: str | None = None, phase: str | None = None,
                      emit_confirmed: bool = False,
                      reconciled: bool = False) -> None:
        """Durably record an honest terminal status (MDS-65 RC#2/#3).

        Writes ``status`` (one of completed/failed/crashed) plus a monotonic
        ``terminal_at`` and clears the pid — synchronously to local disk, before
        and independent of any best-effort bus POST, so a swallowed lifecycle
        emit never loses the outcome. The reconciler reads ``state.json`` as the
        source of truth and re-emits when ``emit_confirmed`` is still False.
        """
        updates: dict = {"status": status, "pid": 0, "terminal_at": time.time()}
        if error:
            updates["error"] = error
        if session_id is not None:
            updates["session_id"] = session_id
        if phase is not None:
            updates["phase"] = phase
        if emit_confirmed:
            updates["emit_confirmed"] = True
        if reconciled:
            updates["reconciled_at"] = time.time()
        self.update(name, **updates)

    def get(self, name: str) -> SessionEntry | None:
        path = self._state_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return SessionEntry.from_dict(data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _reap_if_dead(self, entry: SessionEntry) -> SessionEntry:
        """An active status with a dead pid is a crash, not a clean finish.

        Recording it `crashed` (not the old `done`) is the core honest-status
        fix (MDS-65 RC#2/#3): "crashes recorded as done" is exactly the
        gtm-team bug. The reconciler then re-emits an honest
        agent/session.failed for it (emit_confirmed is False). Runs on every
        list read so history views never show a dead session as running.
        """
        if entry.status not in ACTIVE_STATUSES:
            return entry
        if not entry.pid or pid_alive(entry.pid):
            return entry
        msg = DIED_WITHOUT_TERMINAL
        self.mark_terminal(entry.name, TERMINAL_CRASHED, error=msg)
        # Return the crashed view synthesized in memory, NOT a re-read of
        # state.json: the marking can race a concurrent cleanup or rewrite
        # (mark_terminal no-ops on a missing/torn file), and a re-read could
        # then hand back the stale active entry - the one thing a reap must
        # never return.
        return replace(entry, status=TERMINAL_CRASHED, pid=0,
                       error=entry.error or msg, terminal_at=time.time())

    def list_active(self) -> list[SessionEntry]:
        return [e for e in self.list_all(reap_dead=True)
                if e.status in ACTIVE_STATUSES]

    def list_all(self, *, reap_dead: bool = False) -> list[SessionEntry]:
        """Every session still on disk, history included.

        ``reap_dead`` runs the same dead-pid crash marking ``list_active``
        performs, so a history render (#733 session log) never shows a dead
        session as running. The default stays raw on purpose: the reconciler
        sweeps ``list_all()`` and owns crash-closing there (``reconciled_at``
        + the bus emit), including the exclusion window it grants a
        restarting manager - reaping unconditionally would steal that branch.
        """
        result = []
        sd = _sessions_dir(self._root)
        for d in sd.iterdir():
            if not d.is_dir():
                continue
            state = d / "state.json"
            if not state.exists():
                continue
            try:
                entry = SessionEntry.from_dict(json.loads(state.read_text()))
            except (json.JSONDecodeError, TypeError):
                continue
            result.append(self._reap_if_dead(entry) if reap_dead else entry)
        return result

    def get_by_role(self, role: str) -> list[SessionEntry]:
        return [e for e in self.list_all() if e.role == role]

    def handoff_path(self, name: str, step: str) -> Path:
        return session_handoff_path(name, step, root=self._root)


_registry: SessionRegistry | None = None


def get_registry() -> SessionRegistry:
    global _registry
    if _registry is None:
        _registry = SessionRegistry()
    return _registry


def save_session_id(name: str, session_id: str, model: str | None = None,
                    *, root: Path | None = None) -> None:
    """Persist a session's resume id, plus the model and brain it runs under.

    ``model=None`` leaves any recorded model untouched (for callers that do
    not know it); clearing the id (``session_id=""``) clears the model and
    brain records too, so a fresh session never inherits a stale note. The
    brain record (#642) is the session's provenance: a resume token is only
    meaningful to the brain CONFIGURATION that minted it, endpoint included -
    ``session_brain_label`` keeps gateway sessions on the historical alias
    labels so pre-#789 records still match and a native<->gateway endpoint
    switch reads as a mismatch. ``root=None`` means the process's bound root;
    multi-team hosts pass the target root explicitly.
    """
    from bobi.brain import session_brain_label

    sd = _sessions_dir(root)
    (sd / f"{name}.id").write_text(session_id)
    model_path = sd / f"{name}.model"
    brain_path = sd / f"{name}.brain"
    if not session_id:
        model_path.unlink(missing_ok=True)
        brain_path.unlink(missing_ok=True)
    else:
        if model is not None:
            model_path.write_text(model)
        brain_path.write_text(session_brain_label())
    registry = SessionRegistry(root)
    registry.update(name, session_id=session_id)


def load_session_id(name: str, *, root: Path | None = None) -> str:
    path = _sessions_dir(root) / f"{name}.id"
    if path.exists():
        return path.read_text().strip()
    return ""


def load_session_brain(name: str, *, root: Path | None = None) -> str:
    """The brain kind recorded for *name*'s session, or "" if none is recorded.

    Written alongside the ``.id`` by :func:`save_session_id` (``<name>.brain``);
    the transcript reader needs it to pick the right on-disk format (a Codex
    rollout vs a Claude JSONL transcript)."""
    path = _sessions_dir(root) / f"{name}.brain"
    if path.exists():
        return path.read_text().strip()
    return ""


def load_resumable_session_id(name: str, model: str) -> str:
    """The saved session id, if the active brain can continue it under *model*.

    Same-model sessions always resume. A session recorded under a different
    model continues natively only when the brain supports cross-model resume
    (#642); otherwise it starts fresh - the same rule the workflow
    orchestrator applies. An empty recorded model means "ran under the
    provider default" and still counts as a model; only sessions with no
    record at all (saved before #617) resume unconditionally, and they get a
    model record on their next save.
    """
    from bobi.brain import continuation_token, get_brain, session_brain_label

    saved_id = load_session_id(name)
    if not saved_id:
        return ""
    brain = get_brain()
    active_label = session_brain_label()
    recorded_brain = load_session_brain(name)
    if recorded_brain and recorded_brain != active_label:
        # A resume token is only meaningful to the brain configuration that
        # minted it - engine AND endpoint (see session_brain_label).
        log.info(
            "Session %s was recorded under brain %r but %r is active; "
            "starting fresh.", name, recorded_brain, active_label,
        )
        return ""
    model_path = _sessions_dir() / f"{name}.model"
    if not model_path.exists():
        return saved_id
    recorded = model_path.read_text().strip()
    if not recorded_brain and recorded != (model or ""):
        # No brain provenance (record predates #642) plus a model change:
        # the old guard would have started fresh here, and we cannot prove
        # the token belongs to the active brain. Stay conservative.
        log.info(
            "Session %s has no recorded brain and model changed (%r -> %r); "
            "starting fresh.", name, recorded or "<default>",
            model or "<default>",
        )
        return ""
    token = continuation_token(
        brain, session_id=saved_id,
        from_model=recorded, to_model=model or "",
    )
    if not token:
        log.info(
            "Session %s was recorded under model %r but %r is now resolved; "
            "starting fresh.", name, recorded or "<default>",
            model or "<default>",
        )
    elif recorded != (model or ""):
        log.info(
            "Session %s continues natively from model %r to %r.",
            name, recorded or "<default>", model or "<default>",
        )
    return token


def log_activity(event: str, data: dict | None = None, session: str = "") -> None:
    entry = {"event": event, "ts": time.time()}
    if data:
        entry.update(data)
    log_path = session_log_path(session)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
