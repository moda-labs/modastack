"""Integration test fixtures — fully isolated Bobi Agent installation.

Every integration test runs against a temporary BOBI_HOME. Nothing touches the
real ~/.bobi directory or any production state.
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

PACKAGE_ROOT = Path(__file__).parent.parent.parent
TEST_GRANTS_SECRET = "bobi-integration-test-grants"


def _free_port() -> int:
    """Pick an OS-assigned free TCP port (bind, read, release).

    The one definition for the integration suite — a fix to the port-picking
    logic (SO_REUSEADDR, retry against the pick-then-close race) lands here
    once instead of in every file (D103).
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(base_url: str, timeout: float = 15) -> bool:
    """Poll an event server's /health until it reports ok, or time out.

    Loops on ``bobi.events.server.health()`` — the product's single
    definition of "what counts as healthy" — instead of a hand-rolled
    urllib poll per file (Q041).
    """
    from bobi.events.server import health

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health(base_url):
            return True
        time.sleep(0.3)
    return False


@dataclass
class BobiEnv:
    """Paths for an isolated Bobi home."""
    home_dir: Path
    agent_name: str
    project_path: Path
    package_dir: Path
    state_dir: Path
    sessions_dir: Path
    workflows_dir: Path
    event_server_url: str
    env: dict[str, str]


def _provision_bobi_env(base: Path, *, agent_name: str, brain: str | None,
                        agents_md: str | None = None):
    """Build + install an isolated Bobi home and return its :class:`BobiEnv`.

    The ONE scaffold both the default (Claude) integration fixture and the
    stub-brain fixture share, parameterized only on the brain: ``brain=None``
    leaves the framework default (Claude), ``brain="stub"`` installs the public
    deterministic stub brain and marks the env so ``make_session`` accepts it.
    Caller owns BOBI_HOME/BOBI_ROOT save/restore around the yielded value.
    """
    home_dir = base / "home"
    project_path = home_dir / "agents" / agent_name / "run"
    package_dir = project_path / "package"
    state_dir = project_path / "state"
    sessions_dir = state_dir / "sessions"
    workflows_dir = package_dir / "workflows"
    event_server_url = f"http://localhost:{_free_port()}"

    for d in [home_dir, package_dir, state_dir, sessions_dir, workflows_dir,
              state_dir / "workflow" / "runs", state_dir / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["BOBI_HOME"] = str(home_dir)
    os.environ.pop("BOBI_ROOT", None)

    # Build a local agent team, then install it via the machine-scoped CLI.
    pack_dir = base / "software_team"
    pack_dir.mkdir()
    agent_yaml = {
        "version": "1.0.0",
        "agent": "software_team",
        "entry_point": "manager",
        "event_server": {"url": event_server_url},
        "services": [
            {"name": "github", "events": True},
        ],
    }
    if brain:
        agent_yaml["brain"] = {"kind": brain}
    (pack_dir / "agent.yaml").write_text(yaml.dump(agent_yaml))
    if agents_md is not None:
        # Team-shipped global instructions (#779), installed to
        # run/package/AGENTS.md and rendered at manager boot.
        (pack_dir / "AGENTS.md").write_text(agents_md)
    for role_name, content in [
        ("manager", "# Manager\n\nYou are a test manager agent.\n"),
        ("engineer", "# Engineer\n\nYou are a test engineer agent. Complete tasks quickly.\n"),
        ("project_lead", "# Project Lead\n\nYou are a test project lead agent.\n"),
    ]:
        (pack_dir / "roles" / role_name).mkdir(parents=True)
        (pack_dir / "roles" / role_name / "ROLE.md").write_text(content)

    (pack_dir / "workflows").mkdir()
    (pack_dir / "workflows" / "adhoc.yaml").write_text(
        "name: adhoc\n"
        "trigger: >\n"
        "  For any ad-hoc task.\n"
        "description: >\n"
        "  Open-ended task.\n"
        "steps:\n"
        "  - name: task\n"
        '    prompt: "${{input.task}}"\n'
    )

    (pack_dir / "workflows" / "two-step.yaml").write_text(
        "name: two-step\n"
        "trigger: >\n"
        "  For integration testing — two quick steps.\n"
        "steps:\n"
        "  - name: analyze\n"
        '    prompt: "Say STEP1_DONE and nothing else."\n'
        "    timeout: 60\n"
        "    handoff:\n"
        "      required: []\n"
        "  - name: summarize\n"
        '    prompt: "Say STEP2_DONE and nothing else."\n'
        "    timeout: 60\n"
    )

    env = {
        **os.environ,
        "BOBI_HOME": str(home_dir),
        "BOBI_EVENT_SERVER": event_server_url,
        "BOBI_ES_TEST_GRANTS_SECRET": TEST_GRANTS_SECRET,
    }
    if brain == "stub":
        # Acknowledge the test-only brain (the make_session gate). BOBI_BRAIN is
        # set too so in-process resolution picks the stub without agent.yaml.
        env["BOBI_STUB_BRAIN"] = "1"
        env["BOBI_BRAIN"] = "stub"

    result = subprocess.run(
        [
            sys.executable, "-m", "bobi.cli",
            "agents", "install", str(pack_dir),
            "--name", agent_name, "--non-interactive",
        ],
        capture_output=True, text=True, timeout=30,
        cwd=str(base), env=env,
    )
    assert result.returncode == 0, f"install failed: {result.stderr}"

    subprocess.run(
        ["git", "init"], cwd=str(project_path),
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(project_path), capture_output=True, check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
             "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
    )
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/test-org/test-repo.git"],
        cwd=str(project_path), capture_output=True, check=True,
    )

    return BobiEnv(
        home_dir=home_dir,
        agent_name=agent_name,
        project_path=project_path,
        package_dir=package_dir,
        state_dir=state_dir,
        sessions_dir=sessions_dir,
        workflows_dir=workflows_dir,
        event_server_url=event_server_url,
        env=env,
    )


@pytest.fixture(scope="session")
def claude_bobi_env(tmp_path_factory):
    """Create a fully isolated Bobi home on the default (Claude) brain.

    Session-scoped: created once, shared across all integration tests.
    Includes a real git repo at the selected run root (for worktree support),
    config files, workflows, and empty credentials.
    """
    old_home = os.environ.get("BOBI_HOME")
    old_root = os.environ.get("BOBI_ROOT")
    env = _provision_bobi_env(tmp_path_factory.mktemp("bobi"),
                              agent_name="test-repo", brain=None)
    try:
        yield env
    finally:
        if old_home is None:
            os.environ.pop("BOBI_HOME", None)
        else:
            os.environ["BOBI_HOME"] = old_home
        if old_root is None:
            os.environ.pop("BOBI_ROOT", None)
        else:
            os.environ["BOBI_ROOT"] = old_root


@pytest.fixture(scope="session")
def bobi_env(claude_bobi_env):
    """Default isolated env (Claude brain) - the name the bulk of the suite uses.

    A thin alias so existing tests keep the ``bobi_env`` name while the session
    scaffold lives under ``claude_bobi_env`` (the sibling of ``stub_bobi_env``).
    Session-scoped like the original so module/session-scoped consumers (e.g.
    ``test_event_server``'s ``event_server`` fixture) can still depend on it.
    """
    return claude_bobi_env


@pytest.fixture(scope="module")
def stub_bobi_env(tmp_path_factory):
    """A sibling of :func:`bobi_env` running the public stub brain.

    The same isolated scaffold, but the installed team selects ``brain: stub``
    so a REAL manager boots to idle with no ``claude`` CLI or credentials -
    letting the runtime-plumbing integration tests (start/stop/status/restart,
    event flow) run deterministically in the fast lane instead of behind
    ``requires_claude``. It is the same stub brain the private deploy-package
    sidecar e2e uses, so both surfaces share one test double.

    Module-scoped (not session): the stub suites each start and churn real
    managers / a local event server, so a per-file home keeps one suite's
    leftover runtime state from polluting the next (the Claude fixture can stay
    session-scoped because those tests are gated and rarely run together).
    """
    old_home = os.environ.get("BOBI_HOME")
    old_root = os.environ.get("BOBI_ROOT")
    # Same agent_name as bobi_env (isolated in its own home) so tests that
    # hardcode "test-repo" session names work against either fixture.
    env = _provision_bobi_env(tmp_path_factory.mktemp("bobi-stub"),
                              agent_name="test-repo", brain="stub")
    try:
        yield env
    finally:
        if old_home is None:
            os.environ.pop("BOBI_HOME", None)
        else:
            os.environ["BOBI_HOME"] = old_home
        if old_root is None:
            os.environ.pop("BOBI_ROOT", None)
        else:
            os.environ["BOBI_ROOT"] = old_root


@pytest.fixture(autouse=True)
def _bind_bobi_env_for_test(request):
    """Bind the shared integration Bobi Agent only for tests that use it.

    Binds whichever isolated home the test requested - the default Claude
    ``bobi_env`` or the ``stub_bobi_env`` - so in-process code (``paths``,
    ``set_project_root``) resolves the same run root the subprocesses use.

    ``BOBI_HOME`` is pinned here, per test, and NOT left to the fixtures'
    own save/restore. The two envs have different lifetimes - ``stub_bobi_env``
    is module-scoped, ``claude_bobi_env`` is session-scoped - and both
    save/restore the same process-global. On a dual-brain module the stub leg
    runs first, so it captures ``BOBI_HOME=None``, and its MODULE teardown then
    pops the variable while the session-scoped Claude env is still live. Every
    later test that spawns a subprocess inherited an unset ``BOBI_HOME`` and
    resolved the developer's real ``~/.bobi`` instead of the isolated one -
    silently, because the child failed with "agent not installed" and the
    parent only saw an unparseable result. Pinning per test makes the binding
    independent of fixture teardown order.
    """
    fixture = next((f for f in ("bobi_env", "stub_bobi_env")
                    if f in request.fixturenames), None)
    if fixture is None:
        yield
        return

    from bobi import paths
    from bobi.sdk import set_project_root

    env = request.getfixturevalue(fixture)
    paths.bind_root(env.project_path)
    set_project_root(env.project_path)
    # Pin the selected home, and (for the stub env) the brain, in os.environ so
    # BOTH in-process resolution and any subprocess the test spawns (which
    # inherits os.environ) land on THIS isolated env. Saved/restored around the
    # test - see the note above on why BOBI_HOME cannot be left to the
    # fixtures' own teardown.
    pins = {k: env.env[k]
            for k in ("BOBI_HOME", "BOBI_BRAIN", "BOBI_STUB_BRAIN")
            if k in env.env}
    saved = {k: os.environ.get(k) for k in pins}
    os.environ.update(pins)
    # BOBI_ROOT needs no handling here: `paths.bind_root` above already wrote
    # this env's run root into it, and `paths.bind_root(None)` in the finally
    # clears it - so it is rebound per test either way.
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        set_project_root(None)
        paths.bind_root(None)


requires_claude = pytest.mark.skipif(
    not shutil.which("claude"),
    reason="claude CLI not installed",
)

requires_codex = pytest.mark.skipif(
    not shutil.which("codex"),
    reason="codex CLI not installed",
)


async def _drain(client):
    """Drain one live brain turn; return (final_text, turn_result).

    Shared by the live-brain integration suites (cross-model resume, gateway,
    effort selection). Deliberately raw: the PRODUCT drain is
    ``bobi.brain.turns.drain_turn``; this helper consumes the protocol
    without the bookkeeping so the suites observe the brain directly.
    """
    from bobi.brain import AssistantText, TurnResult

    text, result = "", None
    async for msg in client.receive_response():
        if isinstance(msg, AssistantText) and msg.text:
            text += msg.text
        elif isinstance(msg, TurnResult):
            result = msg
    return text, result


def _make_cli_run(env_obj: BobiEnv):
    """Return a ``bobi`` CLI runner bound to *env_obj* (Claude or stub)."""
    def _run(*args, timeout=10):
        explicit_top_level = {
            "agent", "agents", "deploy", "destroy", "version",
            "create-slack-bot", "skill", "login-bootstrap",
            "reply", "read-conversation",
            # Removed top-level commands still belong here so removal
            # regression tests assert the actual top-level CLI error instead
            # of routing them through `bobi agent <name>`.
            "supervise", "slack-reply", "slack-upload-file", "slack-read-thread",
        }
        argv = list(args)
        if argv and argv[0] not in explicit_top_level:
            argv = ["agent", env_obj.agent_name, *argv]
        # Rebuild from live os.environ each call (so a test's monkeypatch is
        # honored), then layer the env's fixed overrides - including the
        # stub-brain pins that make the subprocess select the stub.
        env = {
            **os.environ,
            "BOBI_HOME": str(env_obj.home_dir),
            "BOBI_EVENT_SERVER": env_obj.event_server_url,
            "BOBI_ES_TEST_GRANTS_SECRET": TEST_GRANTS_SECRET,
        }
        for key in ("BOBI_BRAIN", "BOBI_STUB_BRAIN"):
            if key in env_obj.env:
                env[key] = env_obj.env[key]
        return subprocess.run(
            [sys.executable, "-m", "bobi.cli", *argv],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(env_obj.project_path), env=env,
        )
    return _run


@pytest.fixture
def cli_run(bobi_env):
    """Run bobi CLI commands against the isolated (Claude) install."""
    return _make_cli_run(bobi_env)


@pytest.fixture
def stub_cli_run(stub_bobi_env):
    """Run bobi CLI commands against the stub-brain install (no claude CLI)."""
    return _make_cli_run(stub_bobi_env)


# The "without and with real claude" tiers as one axis: a runtime-plumbing test
# runs on the stub (fast lane, always) AND on real Claude (gated, so it still
# proves the real manager/subagent path when the CLI is present). Mirrors the
# private sidecar e2e's brain parametrization.
BRAIN_PARAMS = [
    pytest.param("stub", id="stub"),
    pytest.param("claude", id="claude", marks=requires_claude),
]


@pytest.fixture(params=BRAIN_PARAMS)
def dual_brain_env(request):
    """An isolated env parametrized over both brains (stub + claude)."""
    name = "stub_bobi_env" if request.param == "stub" else "claude_bobi_env"
    return request.getfixturevalue(name)


@pytest.fixture
def dual_brain_cli_run(dual_brain_env):
    """CLI runner bound to whichever brain ``dual_brain_env`` selected."""
    return _make_cli_run(dual_brain_env)


def _reap_process_group(pid: int, grace: float = 5.0) -> None:
    """Terminate the process group led by *pid*, then WAIT for it to empty.

    A launch is detached with ``start_new_session=True``
    (``bobi/subagent.py::_launch_detached``), so the agent leads its own
    process group and ``killpg`` takes its node children with it. The wait is
    the load-bearing part, and it is why ``cancel_agent`` is not enough here:
    that SIGTERMs and returns immediately, leaving the agent free to keep
    writing while the caller removes its directory.

    Bounded and best effort. SIGKILL after the grace period, and return
    regardless once it expires: a process that took SIGKILL cannot execute
    another instruction, which is the property the caller actually needs.
    """
    if pid <= 0:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(pid, sig)
        except OSError:
            return  # no such group: already gone, or not ours to signal
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except OSError:
                return
            time.sleep(0.02)


def _drop_session(name):
    """Retire *name* from whichever install this process is bound to.

    Reap, THEN remove. A launch is detached by design, so the agent is still
    writing into its session directory when the test that launched it cleans
    up. Removing first raced that writer and raised ``OSError: [Errno 39]
    Directory not empty`` on roughly 3% of CI runs; killing the writer removes
    the race outright instead of narrowing it, and closes the orphaned-node
    leak the CI runner used to clean up after every run.

    The registry is resolved per call, never cached: the bound root is pinned
    per test, so a registry captured at fixture setup would outlive its env.
    """
    from bobi.sdk import get_registry
    registry = get_registry()
    entry = registry.get(name)
    _reap_process_group(entry.pid if entry else 0)
    registry.mark_done(name)
    session_dir = registry.session_dir(name)
    if session_dir.exists():
        shutil.rmtree(session_dir)


def _clean_session():
    """Shared body for the `clean_session` fixtures - clean on both ends."""
    names = []

    def _register(name):
        names.append(name)
        _drop_session(name)

    yield _register

    for name in names:
        _drop_session(name)


@pytest.fixture
def clean_session(bobi_env):
    """Clean up a named session from the registry."""
    yield from _clean_session()


@pytest.fixture
def stub_clean_session(stub_bobi_env):
    """`clean_session` for the stub-only lane.

    Same body; the DEPENDENCY is the whole point. The autouse root binder
    prefers `bobi_env` wherever it appears in a test's fixture graph, so a
    stub-only test that reached for `clean_session` would pin this process to
    the claude install while its CLI subprocess wrote to the stub one - and
    then clean, and read back, the wrong registry.
    """
    yield from _clean_session()
