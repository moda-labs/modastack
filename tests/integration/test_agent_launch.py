"""Integration tests for agent launching — adhoc and multi-step workflows.

Uses a short 2-step test workflow instead of the full issue-lifecycle so tests
complete quickly. All session state goes into the isolated install. Runs on BOTH
brains (``dual_brain_env``): the public stub (fast lane, always) and real Claude
(gated). These assert launch + completion plumbing (the subagent/workflow
completes and writes its session state/logs), so the stub proves them in CI while
the Claude leg exercises a real subagent locally - the same stub the private
sidecar e2e uses.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from bobi.sdk import _sessions_dir


# Bind this file's ``bobi_env`` / ``cli_run`` to the dual-brain (stub + claude)
# variants (see test_manager_lifecycle for the pattern) so subagents launch once
# per brain while the test bodies stay untouched.
@pytest.fixture
def bobi_env(dual_brain_env):
    return dual_brain_env


@pytest.fixture
def cli_run(dual_brain_cli_run):
    return dual_brain_cli_run


# These tests assert launch ADMISSION, never latency, so the subprocess budget
# is generous on purpose: a cold `bobi` import is seconds on a loaded box, and
# the sibling tests that do assert latency own that concern.
LAUNCH_TIMEOUT_S = 60


@pytest.mark.timeout(600)
class TestUnkeyedLaunchDedup:
    """The #850 incident, end to end through the real CLI.

    A role file documented `subagents launch` without `--id`. Every launch got a
    random run key, so the "already active" guard never fired and the chain ran
    50 deep to the spend cap. With a derived key the chain terminates at N=2.

    Stub brain only, and deliberately. Admission decides before the brain is
    ever called, so this is the brain-agnostic case CLAUDE.md's "one mechanism,
    two brains" rule exempts - a claude leg would spend real money re-proving a
    guard that cannot reach a model. Binding ONE env is also load-bearing: the
    dual-brain runner sends its subprocess to the stub home while `bobi_env`
    pins this process to the claude one, so a registry read here would look in
    an install the CLI never wrote to.
    """

    ROLE = "engineer"

    def _task(self, tag):
        """A task text unique to ONE test, which is the whole point.

        These tests share a role and a project, so a shared task constant would
        derive one session name for all of them - and the first test leaves its
        agent running detached, so it re-registers that name after the next
        test has cleaned it and gets that test's own first launch refused. The
        feature's own advice, applied to its tests: fanning out, give each unit
        its own task string.
        """
        return f"Say 'hello dedup' and exit. Issue #850 ({tag})"

    def _derived_session_name(self, env, task):
        """The name the CLI below will actually register under.

        Mirrors `launch_agent`'s own two lines rather than restating them: the
        SAME project dial feeds both the derivation and the session name. Every
        dial the launch passes has to be passed here too - `--role`, and the
        project just as much. Deriving without `project` while the name still
        carried it computed a key production never produces, so both tests
        asserted against a session that does not exist.

        The dial is the env's agent name because these fixtures install at
        `<home>/agents/<agent>/run`, the runtime-scoped shape that
        `_resolve_project_name` names after the agent. It is not re-derived by
        calling that helper: it answers for the root bound in THIS process,
        which is not the one the CLI subprocess resolves.
        """
        from bobi.subagent import derive_run_key
        from bobi.workflow.orchestrator import make_session_name
        key = derive_run_key("adhoc", task, project=env.agent_name,
                             role=self.ROLE)
        return make_session_name("adhoc", env.agent_name, key)

    def test_an_unkeyed_launch_registers_under_the_derived_name(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        """Half of the guard: both launches have to agree on a name at all.

        This is what was broken. Every un-keyed launch minted a random key, so
        no two ever landed on the same session and the admission check below
        could not fire no matter how it was written.
        """
        cli_run = stub_cli_run
        task = self._task("registers")
        name = self._derived_session_name(stub_bobi_env, task)
        stub_clean_session(name)

        result = cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--task", task,
            timeout=LAUNCH_TIMEOUT_S,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert f"Agent started: {name}" in result.stdout, result.stdout
        # The derivation is announced, so an un-keyed launch is not silent.
        assert "derived" in result.stderr, result.stderr

    def test_identical_unkeyed_launch_is_refused_while_the_first_runs(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        """The other half: the run in flight refuses its own twin.

        The first run is pinned active rather than raced against. A real stub
        agent can finish inside the seconds the second CLI spends starting up,
        and then admitting the second launch is *correct* - so racing it would
        make this test pass or fail on timing rather than on the guard.

        Pinning only holds if nothing else is still writing the entry, so the
        first run is allowed to SETTLE first. Otherwise the agent's own
        terminal write lands after the pin and clobbers it - the same race in
        the other direction.
        """
        from bobi.sdk import get_registry
        cli_run = stub_cli_run
        task = self._task("refused")
        name = self._derived_session_name(stub_bobi_env, task)
        stub_clean_session(name)

        first = cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--task", task,
            timeout=LAUNCH_TIMEOUT_S,
        )
        assert first.returncode == 0, f"stderr: {first.stderr}"

        registry = get_registry()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            entry = registry.get(name)
            if entry and entry.status not in ("starting", "running", "idle"):
                break
            time.sleep(1)
        else:
            pytest.fail(f"first run never settled: {registry.get(name)}")

        # Now that no one else writes this entry, hold it active under a pid
        # that is certainly alive: this process, so admission's crash-close
        # (reconcile.close_dead_run) must NOT clear it.
        registry.update(name, status="running", pid=os.getpid())

        second = cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--task", task,
            timeout=LAUNCH_TIMEOUT_S,
        )
        assert second.returncode != 0, (
            "a byte-identical un-keyed relaunch started a second run - "
            f"stdout: {second.stdout}"
        )
        output = second.stdout + second.stderr
        assert "already active" in output, output
        assert "--id-random" in output, output
        assert "subagents cancel" in output, output
        assert "Traceback" not in output, output

    def test_id_random_opts_back_into_parallel_fan_out(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        cli_run, clean_session = stub_cli_run, stub_clean_session
        task = self._task("id-random")
        started = []
        for _ in range(2):
            result = cli_run(
                "subagents", "launch",
                "-w", "adhoc", "--role", self.ROLE, "--id-random",
                "--task", task,
                timeout=LAUNCH_TIMEOUT_S,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            name = result.stdout.split("Agent started:")[-1].strip()
            clean_session(name)
            started.append(name)

        assert len(set(started)) == 2, started

    # "A different task still launches alongside" is deliberately NOT here.
    # Proving it end to end needs two live agents at once, so it measures the
    # concurrency semaphore rather than dedup and fails on a loaded box for a
    # reason unrelated to what it claims. It is pinned deterministically at
    # tests/test_subagent.py::TestLaunchAgentUnkeyedDedup instead.


@pytest.mark.timeout(120)
class TestSessionCleanupReapsBeforeRemoving:
    """The suite's own teardown must not race a live agent.

    ``_drop_session`` used to ``rmtree`` a session directory milliseconds after
    a detached launch returned, while the agent was still writing into it:
    ``OSError: [Errno 39] Directory not empty``, on roughly 3% of CI runs and
    twice red on main this month. The claim is about the PROCESS, so this
    asserts the group is gone rather than looping a launch until a flake stops
    reproducing.
    """

    # A stand-in for a detached agent, with the launch path's topology.
    # `launcher` starts `leader` in its own session (as `_launch_detached`
    # does) and exits, so the leader is reparented exactly as a real agent is
    # once the `bobi` CLI returns, and is never this process's child. `leader`
    # then starts a child of its own, as an agent starts its node processes.
    # Both write into the session directory continuously, so a teardown that
    # removes without reaping meets the same ENOTEMPTY CI did.
    STUB_AGENT = """\
import os
import subprocess
import sys
import time

session_dir, mode = sys.argv[1], sys.argv[2]
if mode == "launcher":
    subprocess.Popen([sys.executable, __file__, session_dir, "leader"],
                     start_new_session=True)
    raise SystemExit(0)
if mode == "leader":
    subprocess.Popen([sys.executable, __file__, session_dir, "child"])
probe = os.path.join(session_dir, "%s-%d.probe" % (mode, os.getpid()))
while True:
    with open(probe, "w") as handle:
        handle.write("x")
    time.sleep(0.005)
"""

    @staticmethod
    def _await_writers(session_dir, timeout=30.0):
        """Block until leader AND child are both writing into *session_dir*.

        Non-vacuity gate: a teardown that reaps nothing proves nothing unless
        there was something live to reap.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = {}
            for probe in session_dir.glob("*.probe"):
                mode, _, pid = probe.stem.rpartition("-")
                found[mode] = int(pid)
            if {"leader", "child"} <= found.keys():
                return found
            time.sleep(0.05)
        raise AssertionError(
            f"stub agent never wrote into {session_dir}: "
            f"{sorted(p.name for p in session_dir.iterdir())}"
        )

    def test_drop_session_reaps_the_group_before_removing_the_directory(
        self, stub_bobi_env, tmp_path
    ):
        from bobi.sdk import SessionEntry, get_registry

        from .conftest import _drop_session

        registry = get_registry()
        name = "wf-adhoc-test-repo-reap-probe"
        registry.register(SessionEntry(name=name, role="engineer",
                                       status="running"))
        session_dir = registry.session_dir(name)

        script = tmp_path / "stub_agent.py"
        script.write_text(self.STUB_AGENT)
        subprocess.run(
            [sys.executable, str(script), str(session_dir), "launcher"],
            check=True, timeout=30,
        )
        writers = self._await_writers(session_dir)
        registry.update(name, pid=writers["leader"])

        try:
            _drop_session(name)
        finally:
            # Never leave the stand-in spinning, even on a failed assertion.
            with contextlib.suppress(OSError):
                os.killpg(writers["leader"], signal.SIGKILL)

        # The load-bearing assertion: by the time _drop_session returns, the
        # whole group is gone - leader and its child, not just the pid the
        # registry knew about. Signal 0 to an empty group is ESRCH.
        with pytest.raises(ProcessLookupError):
            os.killpg(writers["leader"], 0)
        assert not session_dir.exists(), (
            "the session directory survived teardown: "
            f"{sorted(p.name for p in session_dir.iterdir())}"
        )


@pytest.mark.timeout(240)
class TestWaitRunsThroughTheExecutor:
    """--wait is a one-step workflow execution (#1057).

    The blocking CLI form produces the same run identity, ledger entry and
    session shape as a detached dispatch of the same task - one executor.
    Before #1057 a --wait run went through a second executor (spawn_adhoc)
    with no WorkflowRun entry at all, so the ledger assertion here is the
    feature, not a detail.
    """

    ROLE = "engineer"

    def test_wait_prints_the_final_text_and_writes_a_ledger_entry(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        from bobi.workflow.orchestrator import make_session_name
        from bobi.workflow.state import WorkflowRun

        env = stub_bobi_env
        session_name = make_session_name("adhoc", env.agent_name, "W1057")
        stub_clean_session(session_name)

        result = stub_cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--wait", "--id", "W1057",
            "--task", "please __stub__:reply:executor-said-done",
            timeout=LAUNCH_TIMEOUT_S * 3,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # The run's answer travels to --wait stdout at full fidelity.
        assert "executor-said-done" in result.stdout, result.stdout

        run = WorkflowRun.find_by_run_key("adhoc", "W1057",
                                          repo=env.agent_name)
        assert run is not None, "no WorkflowRun ledger entry for the wait run"
        assert run.status == "completed"
        assert run.session_name == session_name

    def test_an_unkeyed_wait_run_derives_and_ledgers_against_the_real_stack(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        """The delegation-idiom shape (no --id): derivation, implied fresh,
        registry and ledger composing for real - each piece is unit-proven
        with the neighbors mocked, and this is where an interaction between
        them would surface."""
        from bobi.workflow.state import WorkflowRun

        env = stub_bobi_env
        task = "unkeyed wait unit __stub__:reply:derived-leg-done"
        result = stub_cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--wait", "--task", task,
            timeout=LAUNCH_TIMEOUT_S * 3,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "derived-leg-done" in result.stdout, result.stdout
        # The derivation is announced, so the un-keyed launch is not silent.
        assert "derived" in result.stderr, result.stderr

        runs = [r for r in WorkflowRun.list_runs()
                if r.workflow_name == "adhoc"
                and r.run_key.startswith("adhoc-")
                and r.repo == env.agent_name]
        assert runs, "no ledger entry for the derived-key wait run"
        assert runs[0].status == "completed"
        assert runs[0].session_name.startswith(
            f"wf-adhoc-{env.agent_name}-adhoc-")
        stub_clean_session(runs[0].session_name)

    def test_a_failed_wait_run_exits_nonzero_with_the_error(
        self, stub_bobi_env, stub_cli_run, stub_clean_session
    ):
        from bobi.workflow.orchestrator import make_session_name
        from bobi.workflow.state import WorkflowRun

        env = stub_bobi_env
        session_name = make_session_name("adhoc", env.agent_name, "W1057F")
        stub_clean_session(session_name)

        result = stub_cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", self.ROLE, "--wait", "--id", "W1057F",
            "--task", "please __stub__:raise:executor-broke",
            timeout=LAUNCH_TIMEOUT_S * 3,
        )
        assert result.returncode != 0, result.stdout
        assert "executor-broke" in result.stderr, result.stderr
        assert "Traceback" not in result.stdout + result.stderr

        run = WorkflowRun.find_by_run_key("adhoc", "W1057F",
                                          repo=env.agent_name)
        assert run is not None and run.status == "failed"


@pytest.mark.timeout(120)
class TestAdhocAgentLaunch:

    def test_adhoc_cli_returns_immediately(self, bobi_env, cli_run, clean_session):
        clean_session("wf-adhoc-test-repo-101")

        start = time.monotonic()
        result = cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", "engineer",

            "--task", "Say hello #101",
            timeout=10,
        )
        elapsed = time.monotonic() - start

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert elapsed < 5, f"adhoc took {elapsed:.1f}s — should return immediately"

    def test_adhoc_agent_completes(self, bobi_env, cli_run, clean_session):
        """Launch via CLI (subprocess finds repo from cwd) and poll for completion."""
        clean_session("wf-adhoc-test-repo-102")

        result = cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", "engineer", "--id", "102",
            "--task", "Say 'hello world' and exit. Issue #102",
            timeout=10,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        session_dir = _sessions_dir() / "wf-adhoc-test-repo-102"

        deadline = time.monotonic() + 90
        completed = False
        while time.monotonic() < deadline:
            state_path = session_dir / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") == "completed":
                    completed = True
                    break
            time.sleep(2)

        assert completed, "Agent did not complete within 90s"
        assert (session_dir / "state.json").exists()
        assert (session_dir / "log.jsonl").exists()

    def test_adhoc_session_state_fields(self, bobi_env, cli_run, clean_session):
        """Verify the session state file has the expected fields after completion."""
        clean_session("wf-adhoc-test-repo-103")

        cli_run(
            "subagents", "launch",
            "-w", "adhoc", "--role", "engineer", "--id", "103",
            "--task", "Reply with DONE. Issue #103",
            timeout=10,
        )

        session_dir = _sessions_dir() / "wf-adhoc-test-repo-103"

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state_path = session_dir / "state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text())
                if state.get("status") == "completed":
                    break
            time.sleep(2)

        state = json.loads((session_dir / "state.json").read_text())
        assert state["status"] == "completed"
        assert state["pid"] == 0
        assert state["role"] == "engineer"

        log_content = (session_dir / "log.jsonl").read_text()
        assert len(log_content) > 0


@pytest.mark.timeout(180)
class TestMultiStepWorkflowLaunch:

    def test_two_step_cli_returns_immediately(self, bobi_env, cli_run, clean_session):
        clean_session("wf-two-step-test-repo-201")

        start = time.monotonic()
        result = cli_run(
            "subagents", "launch",
            "-w", "two-step", "--role", "engineer",

            "--task", "Run test workflow #201",
            timeout=10,
        )
        elapsed = time.monotonic() - start

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert elapsed < 5

    def test_two_step_workflow_runs_both_steps(self, bobi_env, clean_session):
        from bobi.workflow.schema import load_workflow
        from bobi.workflow.orchestrator import run_workflow, make_session_name

        session_name = make_session_name("two-step", "test-repo", "202")
        clean_session(session_name)

        wf_file = bobi_env.workflows_dir / "two-step.yaml"
        wf = load_workflow(wf_file)

        result = run_workflow(
            wf, task="Run two-step test #202", repo="test-repo",
            cwd=str(bobi_env.project_path), run_key="202",
            timeout=120, interactive=False,
        )

        session_dir = _sessions_dir() / session_name
        assert session_dir.exists(), f"Session dir missing: {session_dir}"
        assert (session_dir / "state.json").exists()
        assert (session_dir / "log.jsonl").exists()
