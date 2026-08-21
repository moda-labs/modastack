"""The ``TeamRuntime`` seam between webapp handlers and team operations.

#690 stage 1: the HTTP handlers in ``server.py`` speak only to a
``TeamRuntime``; the local app binds ``LocalRuntime`` at build time. A hosted
deployment binds a different implementation of the same interface - one
webapp codebase, two deployments, each with a fixed binding. There is no
mode-switching logic.

``LocalRuntime`` operates on this machine's ``$BOBI_HOME/agents/`` tree via
explicit-root service calls. It never binds the process to a runtime root, so
one process serves any number of teams concurrently.

Methods return plain machine-readable data (dicts/lists ready for JSON) and
raise the typed errors below; mapping to HTTP status codes stays in the
handlers.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from bobi import paths
from bobi.chat_history import read_chat, read_transcript_messages

DEFAULT_CHAT_TIMEOUT = 300


# --- Errors -----------------------------------------------------------------

class TeamRuntimeError(Exception):
    """Base for runtime-surface errors the handlers translate to HTTP."""


class UnknownTeam(TeamRuntimeError):
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"unknown agent '{name}'")


class UnknownRun(TeamRuntimeError):
    """No run with that id under this team - a 404, not a lifecycle failure.

    Separate from ``UnknownTeam`` because the team resolved fine; it is the
    run inside it that is gone, which is the ordinary case once a record ages
    past the retention cap while a browser still holds its row.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"unknown run '{run_id}'")


class TeamAlreadyRunning(TeamRuntimeError):
    def __init__(self, pid: int) -> None:
        self.pid = pid
        super().__init__(f"already running (pid {pid})")


class TeamPreflightFailed(TeamRuntimeError):
    def __init__(self, report: str) -> None:
        self.report = report
        super().__init__("preflight failed")


class TeamDidNotStop(TeamRuntimeError):
    def __init__(self, pid: int) -> None:
        self.pid = pid
        super().__init__("manager did not stop")


class TeamLifecycleError(TeamRuntimeError):
    """Any other lifecycle failure; the message is user-facing."""


# --- Interface ----------------------------------------------------------------

class TeamRuntime(ABC):
    """Everything the webapp needs from a fleet of teams.

    The stage-1 surface from #690: dashboard snapshot, per-team status,
    lifecycle, subagent roster, messages/transcripts, and submit-then-poll
    chat. Chat is deliberately two calls (submit returns a message id, the
    outcome lands on the job) so no request is held open for a minutes-long
    agent reply regardless of what transport an implementation uses.

    Widening this ABC: both implementers now live in-tree - ``LocalRuntime``
    below and ``EventBusRuntime`` in ``bobi/webapp/event_bus.py`` - so a new
    ``@abstractmethod`` is caught by this repo's own CI, in the same commit
    that adds it. Add the method and both implementations together; there is
    no longer a sequencing rule to follow, because there is no longer an
    implementer this repo cannot see.

    What that rule protected against, and why it is gone: a private consumer
    carried its own COPY of ``EventBusRuntime`` and tracked this repo's
    ``dev`` channel, so an abstract method here broke it on merge - Python
    rejects instantiating a subclass that has not implemented one. That copy
    is being retired in favour of the in-tree class. Until it is, a consumer
    pinning a RELEASED ``bobi`` is unaffected (it moves when it chooses to);
    one tracking ``dev`` sees the break immediately, which is the accepted,
    announced cost of this change rather than a surprise.

    Keep new methods read-only-safe unless they are deliberate operator
    writes, and document the wire shape in the docstring, as below - both
    runtimes must emit it identically, it is rendered once.
    """

    @abstractmethod
    def dashboard(self) -> dict:
        """Every team slot this runtime can see."""

    @abstractmethod
    def team_status(self, name: str) -> dict:
        """One team's card (installed/running/pid/description)."""

    @abstractmethod
    def start_team(self, name: str) -> dict:
        """Start the team's manager; returns ``{"ok": True, "pid": int}``."""

    @abstractmethod
    def stop_team(self, name: str) -> dict:
        """Stop the team's manager; returns the stop outcome fields."""

    @abstractmethod
    def restart_team(self, name: str) -> dict:
        """Stop then start; returns ``{"ok": True, "pid": int}``."""

    @abstractmethod
    def subagents(self, name: str) -> list[dict]:
        """The team's session roster, manager first."""

    @abstractmethod
    def messages(self, name: str, session: str) -> list[dict]:
        """A session's transcript messages (chat-log fallback)."""

    @abstractmethod
    def transcript(self, name: str, session: str) -> dict:
        """A session's transcript as a DEBUGGING view: timestamped lines,
        tool calls included.

        Shape::

            {"session": str,
             "entries": [{"kind",     # message | tool | tool_result
                          "role",     # user | agent
                          "text", "at",        # ISO 8601, "" if unrecorded
                          "tool",              # "" unless a tool line
                          "truncated",         # a clipped tool result
                          "is_error"}, ...],
             "usage": {"started_at", "ended_at", "tokens", "cost_usd",
                       "status"}}

        Distinct from ``messages`` on purpose, and ``messages`` is unchanged:
        that is the CHAT view (two roles, prose only), which is what a
        conversation panel wants. Debugging a run wants when each turn
        happened and what the agent DID between saying things. Both read the
        same transcript; they differ in what they discard.

        ``at`` is empty where the on-disk format records no per-entry
        timestamp (Codex rollouts), rather than synthesized. ``usage`` feeds
        the slab header.
        """

    @abstractmethod
    def run_details(self, name: str, run_id: str) -> dict:
        """What to show for a run that has no transcript to open.

        Shape (see ``bobi.webapp.details.build_details``)::

            {"kind": "monitor",
             "run": {...},          # the run record, whole
             "definition": {...},   # the monitor's definition, {} if gone
             "session_id": str}     # "" when the firing spawned nothing

        A `$0` cached monitor tick and a firing that failed before its agent
        started both have a record and no session. They get the record plus
        the definition of the monitor that produced it - what happened, beside
        what it was asked to do.

        Raises ``UnknownRun`` when no record carries that id.
        """

    @abstractmethod
    def chat_submit(self, name: str, session: str, text: str) -> str:
        """Queue *text* for *session*; returns a message id to poll."""

    @abstractmethod
    def chat_job(self, name: str, message_id: str) -> dict | None:
        """The submit's outcome: pending/done/error. None when the id is
        unknown or belongs to a different team."""

    # --- observability (read-only, #733) --------------------------------
    # Surfaces signals we already capture; no new emitters. One interface,
    # both runtimes, rendered once. The spend vertical folds the per-session
    # cost already written to each session's state.json.

    @abstractmethod
    def spend_summary(self, name: str) -> dict:
        """One team's spend: total plus breakdown by session, role, and model.

        Shape (see ``bobi.costs.CostSummary.to_dict``)::

            {"total_cost_usd", "sessions_counted",
             "by_provider", "by_model", "by_session", "by_role",
             "estimated_cost_usd", "estimated_by_model", "tokens_by_model"}

        ``total_cost_usd``/``by_*`` are provider-reported dollars only;
        ``estimated_*`` is fold-time list-price math over recorded token
        counts for models that report no dollars (#760), kept separate so an
        estimate is never mistaken for a bill. ``tokens_by_model`` carries the
        raw token volumes - the render fallback when no estimate exists.

        Implementations may add ``script_cache`` (see
        ``bobi.webapp.savings``): what the cached-script monitor runner did
        NOT spend. Those are counterfactual dollars and live in their own
        block for that reason - they are never summed into the recorded or
        estimated totals above.
        """

    @abstractmethod
    def overview(self, name: str) -> dict:
        """What a team IS: its description, roles, reach, automations, brain,
        and spend cap - the identity header's read-only view of composition.

        Shape (see ``bobi.webapp.overview.build_overview``)::

            {"name", "description",
             "roles": [{"name", "description"}, ...],
             "chat": {"service", "channels"},
             "services": [{"name", "events", "required"}, ...],
             "automations": {"monitors", "paused_monitors", "workflows"},
             "brain": {"kind", "model", "effort", "max_turns", "gateway"},
             "spend_cap": {"value", "is_default"},
             "entry_role": str}

        Read-only by design: composition is edited in setup, and this exists
        so nobody has to open setup to remember what a team is. Values come
        from the installed package image, never a source directory - the
        runtime runs the image, so the image is the truth.
        """

    @abstractmethod
    def fleet_spend(self) -> dict:
        """Fleet-wide spend for the dashboard: a total plus per-team totals::

            {"total_cost_usd", "estimated_cost_usd", "sessions_counted",
             "teams": [{"name", "total_cost_usd", "estimated_cost_usd",
                        "sessions_counted"}, ...]}

        Same recorded/estimated separation as ``spend_summary``.
        """

    @abstractmethod
    def health_summary(self, name: str) -> dict:
        """One team's system health: manager liveness, session statuses, and
        (where a supervisor records one) the recent lifecycle trail.

        Shape::

            {"reachability": "live" | "stale" | "unreachable",
             "last_heartbeat_at": str | None,
             "state": "running" | "stopped" | "not_responding",
             "detail": str,          # one line of prose for a human
             "segments": [{"key", "label", "kind", "value", "note"}, ...],
             "manager": {"status", "pid", "running", "healthy",
                         "restart_count", "last_restart_reason",
                         "last_restart_at", "idle_seconds"},
             "sessions": [{"name", "role", "status"}, ...],
             "lifecycle": [{"event", "received_at", "at", ...}, ...]}

        ``lifecycle`` is newest first. Fields a runtime cannot know are
        null/empty rather than omitted (a local team has no heartbeats and no
        supervisor trail), so render code branches on value, never on key
        presence.

        ``state`` is the agent's own state, and ``not_responding`` is why it
        exists: a manager process can be alive while the manager is not
        working, and reporting that as running is the failure this surface
        cannot afford. ``segments`` is the status strip's telemetry, ordered
        for display and BEST-EFFORT: a reading this runtime cannot produce is
        omitted rather than faked, so callers must render the list they are
        given rather than expecting a fixed set. ``kind`` says how to read
        ``value`` - ``duration`` (elapsed seconds), ``time`` (epoch seconds),
        ``count``, ``text`` - and formatting belongs to the client, which
        knows the viewer's timezone.

        These three keys are newer than this interface. An implementation
        that predates them may omit them; ``bobi.webapp.health.normalize``
        completes the payload at the response boundary so the null/empty rule
        above holds for every runtime. See ``bobi/webapp/health.py``.
        """

    @abstractmethod
    def session_log(self, name: str) -> dict:
        """One team's session history, newest first: every session the
        registry still holds - active and terminal - with its honest outcome.

        Shape (each row is the ``serialize_session`` view: the roster card
        fields plus ``session_id``/``error``/``terminal_at``)::

            {"sessions": [{"name", "session_id", "role", "title", "phase",
                           "project", "status",  # starting|running|idle|
                                                 # completed|failed|crashed
                                                 # (+ "done" = completed,
                                                 #  "error" = failed)
                           "error",              # terminal failure message,
                                                 # "" otherwise
                           "ended",              # bool: status has left the
                                                 # active vocabulary
                           "model", "provider", "total_cost_usd", "run_key",
                           "started_at", "last_activity",  # epoch seconds
                           "terminal_at",        # epoch seconds | None
                           "is_manager"}, ...],
             "counts": {"active", "completed", "failed", "crashed"},
             "truncated": bool}

        ``sessions`` is newest first by last activity. ``counts`` covers the
        whole history even when an implementation caps ``sessions`` for
        transport (``truncated`` flags that cap). Transcripts drill in via
        ``messages(name, session)`` - a terminal session's transcript stays
        readable as long as its registry entry exists.
        """

    @abstractmethod
    def runs(self, name: str, *, status: str = "", query: str = "",
             offset: int = 0, limit: int | None = None) -> dict:
        """One team's runs: sessions, workflow runs, and monitor runs merged
        into one list, newest first with live runs at the top.

        Shape (see ``bobi.webapp.runs.build_runs``)::

            {"runs": [{"kind",        # session | workflow | monitor
                       "key",         # stable row identity
                       "status",      # running|idle|done|failed|crashed|
                                      # awaiting_action|closed
                       "title", "origin",       # what it is, what kicked it off
                       "started_at",            # ISO 8601, "" if unknown
                       "duration_seconds",      # float | None
                       "tokens", "cost_usd", "est_cost_usd",
                       "error",                 # "" unless it failed
                       "session_id", "run_id",  # "" when the row has neither
                       "detail"}, ...],         # kind-specific extras
             "counts": {"all", "running", "awaiting_action", "failed"},
             "total": int, "offset": int, "limit": int, "query": str,
             "truncated": bool}

        ``status`` and ``query`` filter before ``offset`` / ``limit`` paginate;
        ``failed`` covers terminal failures; human gates use
        ``awaiting_action``.
        ``counts`` describes the whole set and ``total`` the filtered set.
        """

    @abstractmethod
    def resume_run(self, name: str, run_id: str, *, verdict: str = "",
                   reply: str = "") -> dict:
        """Resume one suspended workflow run, answering its gate.

        ``verdict`` is one of ``bobi.workflow.schema.GATE_VERDICTS`` and
        ``reply`` is the human's own words. Both reach the workflow as the
        ``event`` scope, so a route step after the await takes the branch the
        answer chose. An unrecognised verdict raises ``TeamLifecycleError``
        (409) instead of resuming; an absent one is not an approval.

        Returns
        ``{"ok", "accepted", "run_id", "workflow", "await_event", "verdict"}``.
        ``accepted`` is the honest word: a workflow run takes as long as it
        takes, so this returns once the resume is under way and the caller
        watches ``runs`` for the status to move - the same submit-then-poll
        discipline chat uses. No request is ever held open for a workflow.

        Raises ``UnknownRun`` when no run carries that id, and
        ``TeamLifecycleError`` (409) when the run is not in a resumable
        state. Resuming is single-winner: exactly one resume of a given run
        proceeds even if two arrive together.
        """

    @abstractmethod
    def remind_run(self, name: str, run_id: str) -> dict:
        """Resend a waiting workflow's user-facing gate notification.

        Returns ``{"ok", "delivered", "run_id", "workflow", "await_event"}``.
        The run is untouched - this nudges the human, it does not advance the
        workflow. Same error vocabulary as ``resume_run``, plus
        ``TeamLifecycleError`` when the notification could not be delivered.
        """

    @abstractmethod
    def close_run(self, name: str, run_id: str) -> dict:
        """Close a waiting workflow without advancing its gate.

        Returns ``{"ok", "closed", "run_id", "workflow"}``, and marks the
        run's session cancelled. Same error vocabulary as ``resume_run``.
        """


# --- Local implementation ----------------------------------------------------

def _first_paragraph(md: str) -> str:
    """First prose paragraph of an agent.md - the card description."""
    para: list[str] = []
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            if para:
                break
            continue
        para.append(s)
    return " ".join(para)


def _describe(agent_dir: Path) -> str:
    try:
        return _first_paragraph((agent_dir / "agent.md").read_text())[:160]
    except OSError:
        return ""


def _manager_pid(root: Path) -> int:
    """The manager pid when alive, else 0. A pure filesystem+signal check -
    the dashboard read path never touches a runtime bind."""
    from bobi.sdk import pid_alive, read_pid

    pid = read_pid(paths.manager_pid_path(root))
    return pid if pid > 0 and pid_alive(pid) else 0


def agent_card(name: str) -> dict:
    """One dashboard card: an installed agent slot and its runtime state."""
    root = paths.agent_run_root(name)
    pid = _manager_pid(root)
    return {
        "name": name,
        "installed": True,
        "running": bool(pid),
        "pid": pid,
        "description": _describe(paths.package_dir(root)),
    }


def design_card(name: str) -> dict:
    """A source-only slot (designed, never installed) - dashboard shows it so
    the library and the runtime roster share one home."""
    return {
        "name": name,
        "installed": False,
        "running": False,
        "pid": 0,
        "description": _describe(paths.agent_source_dir(name)),
    }


def session_usage(root: Path, session: str) -> dict:
    """The transcript slab header's duration/tokens/cost for one session.

    Module-level, like the serializers below, because the supervisor answers
    the hosted `transcript` command from the box and must produce the SAME
    header the local UI does. An unknown session yields the zero envelope
    rather than raising: the entries are the payload, and a transcript that
    outlived its registry entry still renders.
    """
    from bobi.sdk import SessionRegistry

    entry = SessionRegistry(root).get(session)
    if entry is None:
        return {"started_at": 0.0, "ended_at": 0.0, "tokens": 0,
                "cost_usd": 0.0, "status": ""}
    return {
        "started_at": entry.started_at or 0.0,
        "ended_at": entry.terminal_at or 0.0,
        "tokens": sum(
            int(u.get("input_tokens", 0) or 0)
            + int(u.get("output_tokens", 0) or 0)
            for u in (entry.model_usage or {}).values()
            if isinstance(u, dict)),
        "cost_usd": round(entry.total_cost_usd or 0.0, 6),
        "status": entry.status,
    }


def serialize_subagent(entry, *, manager_name: str = "") -> dict:
    """A session's card view. Mirrors the fields a `SessionEntry`
    (`bobi.sdk.SessionEntry`) exposes; `is_manager` flags the entry-point
    session so the UI can badge it."""
    return {
        "name": entry.name,
        "role": entry.role,
        "title": entry.title,
        "phase": entry.phase,
        "project": entry.project,
        "status": entry.status,
        "model": entry.model,
        "provider": entry.provider,
        "total_cost_usd": round(entry.total_cost_usd or 0.0, 4),
        "run_key": entry.run_key,
        "started_at": entry.started_at,
        "last_activity": entry.last_activity,
        "is_manager": bool(manager_name) and entry.name == manager_name,
    }


def ordered_subagents(entries, *, manager_name: str = "") -> list:
    return sorted(entries,
                  key=lambda e: (0 if manager_name and e.name == manager_name
                                 else 1, e.started_at or 0))


def serialize_session(entry, *, manager_name: str = "") -> dict:
    """A session-log row (#733 vertical 3): the roster card view plus the
    honest terminal outcome. ``status`` already carries the MDS-65 vocabulary
    (completed/failed/crashed); ``error``/``terminal_at`` say why and when.
    ``ended`` derives from the ACTIVE vocabulary (not the terminal one) so
    render code never has to enumerate every word a writer may record -
    stopped/cancelled/legacy words are all honestly "over". The hosted
    supervisor builds the identical row."""
    from bobi.sdk import ACTIVE_STATUSES

    row = serialize_subagent(entry, manager_name=manager_name)
    row["session_id"] = entry.session_id
    row["error"] = entry.error
    row["terminal_at"] = entry.terminal_at or None
    row["ended"] = entry.status not in ACTIVE_STATUSES
    return row


def session_outcome_counts(entries) -> dict:
    """Outcome buckets over a team's whole history. Legacy ``done`` records
    (pre-MDS-65 successes) count as completed; ``error`` counts as failed
    (session.py/subagent.py still write it for turn-level failures -
    rotation-recovery death, monitor timeouts, unparseable verdicts).
    Statuses outside the vocabulary (stopped/cancelled) stay listed but
    uncounted."""
    from bobi.sdk import (
        ACTIVE_STATUSES,
        TERMINAL_COMPLETED,
        TERMINAL_CRASHED,
        TERMINAL_FAILED,
    )

    counts = {"active": 0, "completed": 0, "failed": 0, "crashed": 0}
    for e in entries:
        if e.status in ACTIVE_STATUSES:
            counts["active"] += 1
        elif e.status in (TERMINAL_COMPLETED, "done"):
            counts["completed"] += 1
        elif e.status in (TERMINAL_FAILED, "error"):
            counts["failed"] += 1
        elif e.status == TERMINAL_CRASHED:
            counts["crashed"] += 1
    return counts


def ordered_session_log(entries) -> list:
    """Session-log order: newest activity first (a log, not a roster)."""
    return sorted(entries, key=lambda e: e.last_activity or 0.0, reverse=True)


class LocalRuntime(TeamRuntime):
    """Today's behavior: this machine's agents tree, explicit-root service
    calls, chat delivered by a background thread per submit."""

    def __init__(self) -> None:
        # Submit-then-poll job store. Carries only status and errors; the
        # reply itself reaches the transcript via the messages poll. Guarded
        # by a lock: submits and finishing worker threads share it.
        self._chat_jobs: dict[str, dict] = {}
        self._chat_lock = threading.Lock()
        # Spawning pins process-global brain state for the in-process
        # preflight probe (service.spawn_team, #655) - serialize starts so
        # two teams' preflights never interleave those env writes. Held only
        # for the spawn call, not for stop waits or chat.
        self._spawn_lock = threading.Lock()

    def _resolve(self, name: str) -> Path:
        try:
            return paths.resolve_root_for_agent(name)
        except RuntimeError:
            raise UnknownTeam(name) from None

    def dashboard(self) -> dict:
        """Every agent slot on this machine: installed (with run state)
        first, then design-only sources."""
        installed = paths.list_agents()
        cards = [agent_card(name) for name in installed]

        agents_root = paths.agents_root()
        if agents_root.is_dir():
            for d in sorted(agents_root.iterdir()):
                if (d.is_dir() and d.name not in installed
                        and (d / "src" / "agent.yaml").is_file()):
                    cards.append(design_card(d.name))
        return {"agents": cards, "home": str(paths.home_dir())}

    def team_status(self, name: str) -> dict:
        self._resolve(name)
        return agent_card(name)

    def start_team(self, name: str) -> dict:
        from bobi import service

        root = self._resolve(name)
        try:
            with self._spawn_lock:
                result = service.spawn_team(root)
        except service.AlreadyRunning as e:
            raise TeamAlreadyRunning(e.pid) from e
        except service.PreflightFailed as e:
            raise TeamPreflightFailed(e.validation.format()) from e
        except service.ServiceError as e:
            raise TeamLifecycleError(str(e)) from e
        return {"ok": True, "pid": result.startup.pid}

    def stop_team(self, name: str) -> dict:
        from bobi import service

        root = self._resolve(name)
        result = service.stop_team(root)
        return {
            "ok": result.stopped or result.killed or result.stale
                  or result.pid == 0,
            "stopped": result.stopped,
            "pid": result.pid,
            "still_running": result.still_running,
        }

    def restart_team(self, name: str) -> dict:
        from bobi import service

        root = self._resolve(name)
        try:
            result = service.restart_team(root)
        except service.RestartFailed as e:
            raise TeamLifecycleError(e.report()) from e
        return {"ok": True, "pid": result.pid}

    def subagents(self, name: str) -> list[dict]:
        from bobi import service

        root = self._resolve(name)
        mgr = service.manager_session_name(root)
        entries = service.list_agents(root)
        return [serialize_subagent(e, manager_name=mgr)
                for e in ordered_subagents(entries, manager_name=mgr)]

    def messages(self, name: str, session: str) -> list[dict]:
        from bobi.sdk import load_session_brain, load_session_id

        root = self._resolve(name)
        # The durable source of truth is the session transcript; the web-UI
        # chat log is the fallback when no transcript resolves yet. The recorded
        # brain picks the transcript format (Codex rollout vs Claude JSONL).
        # Both are explicit-path reads.
        messages = read_transcript_messages(
            load_session_id(session, root=root),
            brain=load_session_brain(session, root=root),
        )
        if not messages:
            messages = read_chat(root, session)
        return messages

    def transcript(self, name: str, session: str) -> dict:
        """The debugging view of the same transcript ``messages`` reads."""
        from bobi.chat_history import read_transcript_detail
        from bobi.sdk import load_session_brain, load_session_id

        root = self._resolve(name)
        entries = read_transcript_detail(
            load_session_id(session, root=root),
            brain=load_session_brain(session, root=root),
        )
        return {"session": session, "entries": entries,
                "usage": session_usage(root, session)}

    def run_details(self, name: str, run_id: str) -> dict:
        """The Details slab for a run with no transcript to open."""
        from bobi.webapp.details import UnknownRun as DetailsUnknownRun
        from bobi.webapp.details import build_details

        root = self._resolve(name)
        try:
            return build_details(root, run_id)
        except DetailsUnknownRun:
            raise UnknownRun(run_id) from None

    def _prune_jobs(self) -> None:
        # Caller holds _chat_lock.
        if len(self._chat_jobs) <= 500:
            return
        for mid in [m for m, j in self._chat_jobs.items()
                    if j["status"] != "pending"][:250]:
            self._chat_jobs.pop(mid, None)

    def chat_submit(self, name: str, session: str, text: str) -> str:
        root = self._resolve(name)
        message_id = uuid.uuid4().hex
        with self._chat_lock:
            self._prune_jobs()
            self._chat_jobs[message_id] = {"team": name, "status": "pending"}

        def work() -> None:
            from bobi import service

            try:
                # An empty session means the team manager. The route documents
                # `subagent` as optional and the hosted runtime has always read
                # it that way (supervisor/admin.py), so both runtimes resolve
                # it identically rather than one answering `unknown agent ''`.
                target = (session or "").strip() \
                    or service.manager_session_name(root)
                service.ask(root, target, text,
                            timeout=DEFAULT_CHAT_TIMEOUT)
                outcome = {"team": name, "status": "done"}
            except Exception as e:  # noqa: BLE001 - job must resolve
                outcome = {"team": name, "status": "error", "error": str(e)}
            with self._chat_lock:
                self._chat_jobs[message_id] = outcome

        threading.Thread(target=work, daemon=True,
                         name=f"chat-{message_id[:8]}").start()
        return message_id

    def chat_job(self, name: str, message_id: str) -> dict | None:
        with self._chat_lock:
            job = self._chat_jobs.get(message_id)
        # A job is only visible under the team it was submitted for.
        if job is None or job.get("team") != name:
            return None
        return {k: v for k, v in job.items() if k != "team"}

    # --- observability (read-only, #733) --------------------------------

    def spend_summary(self, name: str) -> dict:
        from bobi.costs import rollup_costs
        from bobi.webapp.savings import script_cache_savings

        root = self._resolve(name)
        # sessions_path (not sessions_dir): a read endpoint must not mkdir.
        payload = rollup_costs(paths.sessions_path(root)).to_dict()
        # Counterfactual dollars, kept in their own block so nothing can
        # accidentally add them to a bill.
        payload["script_cache"] = script_cache_savings(root)
        return payload

    def overview(self, name: str) -> dict:
        from bobi.webapp.overview import build_overview

        return build_overview(self._resolve(name))

    def fleet_spend(self) -> dict:
        """Roll up spend across every installed team. Offline: reads each
        team's session state files directly, one rollup per team."""
        from bobi.costs import rollup_costs

        total = 0.0
        estimated = 0.0
        counted = 0
        teams: list[dict] = []
        for team in paths.list_agents():
            root = paths.agent_run_root(team)
            summary = rollup_costs(paths.sessions_path(root))
            # Sum the rounded per-team totals so the dashboard header always
            # equals the sum of the visible tiles (no sub-cent drift where the
            # header shows spend no tile accounts for).
            team_total = round(summary.total_cost_usd, 4)
            team_estimated = round(summary.estimated_cost_usd, 4)
            teams.append({
                "name": team,
                "total_cost_usd": team_total,
                "estimated_cost_usd": team_estimated,
                "sessions_counted": summary.sessions_counted,
            })
            total += team_total
            estimated += team_estimated
            counted += summary.sessions_counted
        teams.sort(key=lambda t: (t["total_cost_usd"]
                                  + t["estimated_cost_usd"]), reverse=True)
        return {
            "total_cost_usd": round(total, 4),
            "estimated_cost_usd": round(estimated, 4),
            "sessions_counted": counted,
            "teams": teams,
        }

    def health_summary(self, name: str) -> dict:
        """Manager liveness + session statuses from this machine's files -
        the same sources the dashboard card and the roster read (manager
        pidfile, session registry) - plus the manager's own health probe,
        which is what makes ``state`` a tri-state rather than pid presence
        rephrased. A local team shares this host, so ``reachability`` is
        "live" by construction; there is no supervisor here, so the restart
        fields are null and the lifecycle trail is empty - the hosted runtime
        fills those from its sidecar."""
        from bobi import service
        from bobi.webapp import health as health_state

        root = self._resolve(name)
        status = service.team_status(root)
        mgr_name = service.manager_session_name(root)
        entries = ordered_subagents(status.active_agents,
                                    manager_name=mgr_name)
        running = status.manager_running
        mgr_entry = next((e for e in entries if e.name == mgr_name), None)
        if not running:
            mgr_status = "stopped"
        elif mgr_entry is not None and mgr_entry.status:
            mgr_status = mgr_entry.status
        else:
            # Manager pid alive but no registered manager session yet: the
            # boot window. Same fail-open verdict the hosted sidecar reports.
            mgr_status = "starting"

        # The strip's SINCE/EXIT/WAS UP come from the manager's last TERMINAL
        # record, which by definition is not in the active list - so the
        # stopped path pays one extra registry read, and only it does.
        strip_entry = mgr_entry
        if not running:
            strip_entry = self._last_manager_record(root, mgr_name)

        # "Last activity" is the agent's last sign of life, not the manager's
        # alone: a subagent still working is the newest thing that happened.
        last_activity = max((e.last_activity or 0.0 for e in entries),
                            default=0.0)
        state = health_state.build_state(
            running=running,
            pid=status.manager_pid,
            root=root,
            manager_entry=strip_entry,
            live_runs=sum(1 for e in entries if e.name != mgr_name),
            last_activity=last_activity,
        )
        return {
            "reachability": "live",
            "last_heartbeat_at": None,
            "state": state["state"],
            "detail": state["detail"],
            "segments": state["segments"],
            "manager": {
                "status": mgr_status,
                "pid": status.manager_pid,
                "running": running,
                # A wedged manager is not healthy, whatever its pid says.
                # This is the #887 defect: process-alive was read as healthy.
                "healthy": running and state["probe_ok"] is not False,
                "restart_count": None,
                "last_restart_reason": None,
                "last_restart_at": None,
                "idle_seconds": state["idle_seconds"],
            },
            "sessions": [{"name": e.name, "role": e.role, "status": e.status}
                         for e in entries],
            "lifecycle": [],
        }

    @staticmethod
    def _last_manager_record(root: Path, mgr_name: str):
        """The manager's registry entry including terminal ones, or None.

        Read with dead-pid reaping so a manager killed without reporting a
        terminal status reads as crashed here too, never as still running.
        """
        from bobi.sdk import SessionRegistry

        return next((e for e in SessionRegistry(root).list_all(reap_dead=True)
                     if e.name == mgr_name), None)

    def session_log(self, name: str) -> dict:
        """The whole registry, terminal sessions included. ``reap_dead``
        runs the same dead-pid crash marking as the roster read, so a
        session whose process died reads ``crashed`` here, never
        ``running``. Local responses are never capped - the history is on
        this disk."""
        from bobi import service
        from bobi.sdk import SessionRegistry

        root = self._resolve(name)
        mgr = service.manager_session_name(root)
        entries = ordered_session_log(
            SessionRegistry(root).list_all(reap_dead=True))
        return {
            "sessions": [serialize_session(e, manager_name=mgr)
                         for e in entries],
            "counts": session_outcome_counts(entries),
            "truncated": False,
        }

    # The three writes delegate to bobi.webapp.run_actions, which the
    # supervisor's admin commands call too - one implementation, both
    # runtimes. Only the error vocabulary is mapped here: the shared module
    # raises its own exceptions so it can stay free of this one.
    def resume_run(self, name: str, run_id: str, *, verdict: str = "",
                   reply: str = "") -> dict:
        return self._run_action("resume_run", name, run_id,
                                verdict=verdict, reply=reply)

    def remind_run(self, name: str, run_id: str) -> dict:
        return self._run_action("remind_run", name, run_id)

    def close_run(self, name: str, run_id: str) -> dict:
        return self._run_action("close_run", name, run_id)

    def _run_action(self, action: str, name: str, run_id: str, **kw) -> dict:
        from bobi.webapp import run_actions

        root = self._resolve(name)
        try:
            return getattr(run_actions, action)(root, run_id, **kw)
        except run_actions.UnknownRun:
            raise UnknownRun(run_id) from None
        except (run_actions.RunNotWaiting, run_actions.ActionFailed) as e:
            raise TeamLifecycleError(str(e)) from None

    def runs(self, name: str, *, status: str = "", query: str = "",
             offset: int = 0, limit: int | None = None) -> dict:
        """Fold this machine's three run stores for one team. Every read takes
        the resolved root explicitly - this process serves every team and
        binds none of them."""
        from bobi import service
        from bobi.webapp.runs import DEFAULT_LIMIT, build_runs

        root = self._resolve(name)
        return build_runs(
            root,
            manager_name=service.manager_session_name(root),
            status=status or "",
            query=query or "",
            offset=max(0, offset),
            limit=limit if limit and limit > 0 else DEFAULT_LIMIT,
        )
