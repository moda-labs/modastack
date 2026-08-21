"""CLI contract tests for the Bobi Agent command tree."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from bobi.__version__ import __version__
from bobi import paths
from bobi.cli import main
from bobi.subagent import AgentResult, CheckResult, GateResult
from tests.conftest import TEST_AGENT_NAME


def test_version_flag():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "bobi" in result.output
    assert __version__ in result.output


def test_transport_logs_do_not_reach_the_console(monkeypatch):
    """The root logger runs at INFO, and httpx logs every request at INFO.

    `daemon._ping` uses the pooled client (Q123), and `bobi app start` polls
    it every 0.2s while the daemon comes up — so without this the command
    buries its own "running at ..." line under a stack of transport chatter
    that `urllib` never produced.
    """
    import logging

    logging.getLogger("httpx").setLevel(logging.INFO)
    monkeypatch.setattr(
        "bobi.webapp.daemon.start",
        lambda open_browser=True: type("Status", (), {"url": "u", "pid": 1})(),
    )

    CliRunner().invoke(main, [])

    assert logging.getLogger("httpx").level == logging.WARNING


def test_bare_bobi_starts_app(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "bobi.webapp.daemon.start",
        lambda open_browser=True: calls.append(open_browser)
        or type("Status", (), {"url": "http://127.0.0.1:8642/?n=tok",
                               "pid": 1234})(),
    )

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0, result.output
    assert calls == [True]
    assert "bobi app is running at http://127.0.0.1:8642/?n=tok" in result.output


def test_top_level_help_is_machine_scoped():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "agent" in result.output
    assert "agents" in result.output
    assert "setup" in result.output
    for removed in [" start", " stop", " status", " workflows", " monitors", " otel"]:
        assert removed not in result.output


def test_agents_help_lists_machine_commands():
    result = CliRunner().invoke(main, ["agents", "--help"])
    assert result.exit_code == 0
    assert "setup" not in result.output
    for cmd in ["install", "list", "browse", "add-registry"]:
        assert cmd in result.output


def test_agent_help_lists_runtime_commands(bobi_install):
    result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ["start", "stop", "status", "workflows", "monitors",
                "subagents", "event-server", "login-bootstrap", "otel"]:
        assert cmd in result.output


def test_agent_group_pins_team_brain_for_cli_process(bobi_install, monkeypatch):
    """`bobi agent <name> ...` must select the team's brain for sessions the
    CLI process itself runs - a gateway team's `--as-check` monitor check
    otherwise hits real Anthropic with the gateway's token (#655)."""
    import os
    import yaml

    for var in ("BOBI_BRAIN", "BOBI_BRAIN_MODEL",
                "BOBI_GATEWAY_BASE_URL", "BOBI_GATEWAY_SMALL_MODEL",
                "BOBI_GATEWAY_WIRE_API",
                "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    agent_yaml = bobi_install.repo_path / "package" / "agent.yaml"
    cfg = yaml.safe_load(agent_yaml.read_text())
    cfg["brain"] = {"kind": "gateway", "base_url": "http://localhost:4000",
                    "model": "qwen3:14b"}
    agent_yaml.write_text(yaml.dump(cfg))
    (bobi_install.repo_path / ".env").write_text(
        "ANTHROPIC_AUTH_TOKEN=from-runtime-dotenv\n")

    result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "status"])

    assert result.exit_code == 0, result.output
    assert os.environ.get("BOBI_BRAIN") == "gateway"
    assert os.environ.get("BOBI_BRAIN_MODEL") == "qwen3:14b"
    assert os.environ.get("BOBI_GATEWAY_BASE_URL") == "http://localhost:4000"
    assert os.environ.get("ANTHROPIC_AUTH_TOKEN") == "from-runtime-dotenv"


def test_agent_group_pins_gateway_openai_brain_for_cli_process(
    bobi_install, monkeypatch,
):
    import os
    import yaml

    for var in ("BOBI_BRAIN", "BOBI_BRAIN_MODEL",
                "BOBI_GATEWAY_BASE_URL", "BOBI_GATEWAY_SMALL_MODEL",
                "BOBI_GATEWAY_WIRE_API"):
        monkeypatch.delenv(var, raising=False)
    agent_yaml = bobi_install.repo_path / "package" / "agent.yaml"
    cfg = yaml.safe_load(agent_yaml.read_text())
    cfg["brain"] = {
        "kind": "gateway-openai",
        "base_url": "http://localhost:9000/v1",
        "model": "gpt-5.5",
        "wire_api": "responses",
        "small_model": "must-not-pin",
    }
    agent_yaml.write_text(yaml.dump(cfg))

    result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "status"])

    assert result.exit_code == 0, result.output
    assert os.environ.get("BOBI_BRAIN") == "gateway-openai"
    assert os.environ.get("BOBI_BRAIN_MODEL") == "gpt-5.5"
    assert os.environ.get("BOBI_GATEWAY_BASE_URL") == "http://localhost:9000/v1"
    assert os.environ.get("BOBI_GATEWAY_WIRE_API") == "responses"
    assert "BOBI_GATEWAY_SMALL_MODEL" not in os.environ


def test_missing_agent_errors_without_cwd_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBI_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(main, ["agent", "missing", "status"])
    assert result.exit_code != 0
    assert "Bobi Agent 'missing' is not installed" in result.output
    assert "package/agent.yaml" in result.output


def test_agent_ui_deployment_mode_is_removed(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBI_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(
        main, ["agent", "canary", "ui", "ci-canary"])

    assert result.exit_code != 0
    assert "`bobi agent <name> ui <deployment>` was removed" in result.output
    assert "control plane" in result.output


def test_agent_ui_removed_app_flag_reports_control_plane(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBI_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(
        main, ["agent", "canary", "ui", "--app", "ci-canary", "--check"])

    assert result.exit_code != 0
    assert "`bobi agent <name> ui <deployment>` was removed" in result.output
    assert "control plane" in result.output


def test_agent_ui_local_deep_links_unified_app(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBI_HOME", str(tmp_path / "home"))
    opened = {}

    monkeypatch.setattr(
        "bobi.webapp.daemon.start",
        lambda open_browser=True: type(
            "Status", (), {"url": "http://127.0.0.1:8642/?n=tok",
                           "pid": 1234})(),
    )
    monkeypatch.setattr("webbrowser.open",
                        lambda url: opened.setdefault("url", url))

    result = CliRunner().invoke(main, ["agent", "canary", "ui"])

    assert result.exit_code == 0, result.output
    assert opened["url"] == "http://127.0.0.1:8642/?n=tok#/agents/canary"


def test_agents_list_without_installs_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBI_HOME", str(tmp_path / "home"))
    result = CliRunner().invoke(main, ["agents", "list"])
    assert result.exit_code == 0
    assert "No Bobi Agents installed" in result.output


def test_agents_list_shows_installed_agent(bobi_install):
    result = CliRunner().invoke(main, ["agents", "list"])
    assert result.exit_code == 0, result.output
    assert TEST_AGENT_NAME in result.output
    assert str(bobi_install.repo_path) in result.output


def test_workflow_list_shows_installed_workflows(bobi_install):
    result = CliRunner().invoke(
        main, ["agent", TEST_AGENT_NAME, "workflows", "list"])
    assert result.exit_code == 0, result.output
    assert "adhoc" in result.output


def test_workflow_validate_is_agent_scoped(bobi_install, tmp_path):
    wf_file = tmp_path / "test.yaml"
    wf_file.write_text(
        "name: test-wf\ntrigger: manual\nsteps:\n"
        "  - name: s1\n    type: prompt\n    prompt: hello\n"
    )
    result = CliRunner().invoke(
        main, ["agent", TEST_AGENT_NAME, "workflows", "validate", str(wf_file)])
    assert result.exit_code == 0, result.output
    assert "Valid" in result.output


class TestWorkflowResume:
    """`workflows resume` is how a human gate gets answered (#987).

    Two contracts. The verdict has to REACH the workflow - it lands as the
    ``event`` scope, which a route step after the await reads back - and the
    command has to report the run's real ending: ``resume_workflow`` returns
    True both when a run finishes and when it parks on a LATER await step, so
    a bare truthiness check calls a re-gated run "completed."
    """

    def _waiting_run(self):
        from bobi.workflow.state import WorkflowRun
        run = WorkflowRun.create("adhoc", {"data": {"run_key": "42"}})
        run.status = "waiting"
        run.suspended_at_step = 1
        run.await_event = "approval"
        run.run_key = "42"
        run.save()
        return run

    def _resume(self, run_id, fake_resume, *args):
        with patch("bobi.workflow.orchestrator.resume_workflow", fake_resume):
            return CliRunner().invoke(
                main,
                ["agent", TEST_AGENT_NAME, "workflows", "resume", run_id,
                 *args],
            )

    def _completes(self, seen):
        def fake_resume(run, wf, **kwargs):
            seen.append(kwargs)
            run.status = "completed"
            run.save()
            return True
        return fake_resume

    def test_the_verdict_reaches_the_workflow_as_the_event_scope(
            self, bobi_install):
        """The inlet the whole design turns on. ``event`` already becomes the
        run's ``event`` scope inside ``resume_workflow``; this is what finally
        puts something in it."""
        run = self._waiting_run()
        seen: list = []

        result = self._resume(run.run_id, self._completes(seen),
                              "--verdict", "approve", "--reply", "ship it")

        assert result.exit_code == 0, result.output
        assert seen[0]["event"] == {
            "data": {"verdict": "approve", "reply": "ship it"}}

    def test_a_resume_with_no_verdict_still_populates_the_scope(
            self, bobi_install):
        """An empty verdict and a MISSING scope both resolve to "" in a
        condition, but only the first does it without a warning about an
        unknown scope. The route reads either as "not an approval"."""
        run = self._waiting_run()
        seen: list = []

        result = self._resume(run.run_id, self._completes(seen))

        assert result.exit_code == 0, result.output
        assert seen[0]["event"] == {"data": {"verdict": "", "reply": ""}}

    def test_a_verdict_outside_the_vocabulary_is_refused(self, bobi_install):
        """Refused before anything is claimed or resumed. The route would fail
        closed on it anyway, but a typo that silently reworks a spec is a
        worse answer than one that says the word was not understood."""
        run = self._waiting_run()
        called: list = []

        def fake_resume(run, wf, **kwargs):
            called.append(kwargs)
            return True

        result = self._resume(run.run_id, fake_resume, "--verdict", "approved")

        assert result.exit_code != 0
        assert not called, "a malformed verdict reached the workflow"
        from bobi.workflow.state import WorkflowRun
        assert WorkflowRun.load(run.run_id).status == "waiting"

    def test_suspended_again_is_not_reported_as_completed(self, bobi_install):
        """A rejected gate reworks and re-gates, so this is the ordinary
        ending of every reject - not an edge case."""
        run = self._waiting_run()

        def fake_resume(run, wf, **kwargs):
            # What the orchestrator does when a later await step parks the
            # run: the SAME ledger entry goes back to waiting (#1048).
            run.status = "waiting"
            run.save()
            return True

        result = self._resume(run.run_id, fake_resume, "--verdict", "approve")

        assert result.exit_code == 0, result.output
        assert "Workflow completed." not in result.output
        assert "suspended" in result.output.lower()

    def test_a_rejection_the_workflow_cannot_honour_is_refused(
            self, bobi_install):
        """`reject` means "do NOT run the next step". Without a route on the
        verdict in the slot the resume lands on, that is exactly what resuming
        would do, so the command refuses rather than doing the opposite of
        what it was told. The `adhoc` workflow has one prompt step and no
        route."""
        run = self._waiting_run()
        called: list = []

        def fake_resume(run, wf, **kwargs):
            called.append(kwargs)
            return True

        result = self._resume(run.run_id, fake_resume, "--verdict", "reject")

        assert result.exit_code == 1
        assert "no route on the gate's verdict" in result.output
        assert not called

    def test_a_refusal_never_leaves_the_run_claimed(self, bobi_install):
        """`claim()` renames <id>.json to <id>.resuming.json and nothing
        renames it back, so a refusal after claiming would leave the run
        findable forever and resumable never (state.py's D071). Every check
        that can refuse resolves before the claim."""
        from bobi.workflow.state import WorkflowRun
        run = self._waiting_run()

        self._resume(run.run_id, lambda *a, **k: True, "--verdict", "reject")

        assert WorkflowRun.load(run.run_id).status == "waiting"

    def test_completed_run_is_reported_as_completed(self, bobi_install):
        run = self._waiting_run()
        result = self._resume(run.run_id, self._completes([]))

        assert result.exit_code == 0, result.output
        assert "Workflow completed." in result.output

    def test_failed_run_exits_nonzero(self, bobi_install):
        run = self._waiting_run()

        def fake_resume(run, wf, **kwargs):
            run.status = "failed"
            run.save()
            return False

        result = self._resume(run.run_id, fake_resume)

        assert result.exit_code == 1


class TestSubagents:
    def test_launch_adhoc_workflow(self, bobi_install):
        with patch("bobi.subagent.launch_agent", return_value="wf-adhoc-42") as mock:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--task", "Fix #42",
            ])
        assert result.exit_code == 0, result.output
        assert "wf-adhoc-42" in result.output
        mock.assert_called_once()
        assert mock.call_args[1]["workflow_name"] == "adhoc"
        assert mock.call_args[1]["task"] == "Fix #42"
        assert mock.call_args[1]["role"] == "engineer"
        assert mock.call_args[1]["cwd"] == str(bobi_install.repo_path)

    def test_id_random_is_passed_through(self, bobi_install):
        with patch("bobi.subagent.launch_agent", return_value="wf-adhoc-x") as mock:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--id-random",
                "--task", "Fan out",
            ])
        assert result.exit_code == 0, result.output
        assert mock.call_args[1]["random_key"] is True

    def test_id_random_reaches_the_wait_path_too(self, bobi_install):
        """--wait needs its OWN --id-random passthrough (#850).

        Asserting only the detached branch leaves `--wait --id-random` free to
        fall back to a derived key, which is the opposite of what was asked
        for: two deliberate parallel runs would land on one name.
        """
        with patch("bobi.subagent.launch_agent") as mock:
            mock.return_value = MagicMock(final_text="", success=True, error="")
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--wait", "--id-random",
                "--task", "Fan out",
            ])
        assert result.exit_code == 0, result.output
        assert mock.call_args[1]["random_key"] is True
        assert mock.call_args[1]["wait"] is True

    def test_wait_refusal_renders_readably_not_as_a_traceback(
            self, bobi_install):
        """--wait can raise DuplicateRunError since #1057 (the old executor
        had no active-run guard). The reader is an LLM mid-delegation: a raw
        traceback reads as a transient crash and teaches it to retry, so the
        refusal must land through _launch_refusal_is_readable like every
        other launch."""
        from bobi.subagent import DuplicateRunError
        exc = DuplicateRunError(
            "A run is already active: wf-adhoc-r-adhoc-abc123",
            session_name="wf-adhoc-r-adhoc-abc123", status="running",
            derived_key=True)
        with patch("bobi.subagent.launch_agent", side_effect=exc):
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--wait",
                "--task", "Investigate X",
            ])
        assert result.exit_code == 1
        assert "Launch refused" in result.output
        assert "Traceback" not in result.output

    def test_wait_routes_through_the_one_launch_path(self, bobi_install):
        """--wait is launch_agent's synchronous mode (#1057): derivation,
        admission and the run ledger are launch_agent's own, so the CLI
        passes the raw launch through rather than resolving a name first."""
        with patch("bobi.subagent.launch_agent") as mock:
            mock.return_value = MagicMock(final_text="", success=True, error="")
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--wait",
                "--task", "Investigate X",
            ])
        assert result.exit_code == 0, result.output
        assert mock.call_args[1]["wait"] is True
        assert mock.call_args[1]["run_key"] is None
        assert mock.call_args[1]["workflow_name"] == "adhoc"

    def test_id_and_id_random_are_mutually_exclusive(self, bobi_install):
        with patch("bobi.subagent.launch_agent") as mock:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer",
                "--id", "42", "--id-random", "--task", "X",
            ])
        assert result.exit_code != 0
        assert "--id-random" in result.output
        mock.assert_not_called()

    def test_duplicate_launch_reports_cleanly(self, bobi_install):
        """Refusing an un-keyed duplicate is a common path now, not a crash."""
        from bobi.subagent import DuplicateRunError
        refusal = DuplicateRunError(
            "A run is already active: wf-adhoc-x (status=running). Its run key "
            "was derived from the task; pass --id-random to run both.",
            session_name="wf-adhoc-x", status="running", derived_key=True,
        )
        with patch("bobi.subagent.launch_agent", side_effect=refusal):
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--task", "Fix #42",
            ])
        assert result.exit_code == 1
        assert "already active" in result.output
        assert "Traceback" not in result.output
        # The remediation must be runnable as printed, not a <name> placeholder.
        assert f"bobi agent {TEST_AGENT_NAME} subagents cancel wf-adhoc-x" \
            in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_dependency_failure_is_not_reported_as_a_duplicate(self, bobi_install):
        """`launch_agent` raises RuntimeError for the requires preflight and the
        spend governor too; a blanket catch would relabel those."""
        with patch("bobi.subagent.launch_agent",
                   side_effect=RuntimeError("Required dependency check failed: gh")):
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer", "--task", "Fix #42",
            ])
        assert result.exit_code != 0
        assert "Launch refused" not in result.output

    def test_workflow_required(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "--role", "engineer", "--task", "X",
        ])
        assert result.exit_code != 0
        assert "--workflow" in result.output

    def test_role_required(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "-w", "adhoc", "--task", "X",
        ])
        assert result.exit_code != 0
        assert "--role" in result.output

    def test_invalid_role(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "-w", "adhoc", "--role", "nonexistent", "--task", "X",
        ])
        assert result.exit_code != 0
        assert "Unknown role" in result.output

    def test_wait_mode_runs_agent(self, bobi_install):
        agent = AgentResult(
            session_id="sess-1", run_key="run-1", phase="adhoc",
            success=True, final_text="done",
        )
        with patch("bobi.subagent.launch_agent", return_value=agent) as spawn, \
             patch("bobi.subagent.run_check_blocking") as check:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer",
                "--wait", "--task", "Fix prod URL",
            ])
        assert result.exit_code == 0, result.output
        assert "done" in result.output
        check.assert_not_called()
        spawn.assert_called_once()
        assert spawn.call_args.kwargs["task"] == "Fix prod URL"
        assert spawn.call_args.kwargs["role"] == "engineer"

    def test_as_check_runs_check(self, bobi_install):
        check = CheckResult(success=True, finding=False,
                            session="monitor-check-abc-check")
        with patch("bobi.subagent.run_check_blocking", return_value=check) as run_check, \
             patch("bobi.subagent.launch_agent") as spawn:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer",
                "--as-check", "--task", "Check prod URL",
            ])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output.strip()) == {
            "success": True,
            "finding": False,
            "summary": "",
            "details": {},
            # The monitor scheduler records this on the run so a check's row
            # can open its transcript.
            "session": "monitor-check-abc-check",
        }
        run_check.assert_called_once()
        spawn.assert_not_called()

    def test_subagents_launch_help_separates_wait_and_check(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch", "--help",
        ])
        assert result.exit_code == 0, result.output
        assert "--wait" in result.output
        assert "Block until the launched agent completes" in result.output
        assert "--as-check" in result.output
        assert "monitoring check" in result.output

    def test_agent_wait_alias_is_not_supported(self, bobi_install):
        with patch("bobi.subagent.launch_agent") as spawn, \
             patch("bobi.subagent.run_check_blocking") as check:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer",
                "--agent-wait", "--task", "Fix prod URL",
            ])
        assert result.exit_code != 0
        assert "No such option '--agent-wait'" in result.output
        spawn.assert_not_called()
        check.assert_not_called()

    def test_as_check_rejects_wait_flags(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "-w", "adhoc", "--role", "engineer",
            "--as-check", "--wait", "--task", "Check prod URL",
        ])
        assert result.exit_code != 0
        assert "--as-check cannot be combined with --wait" in result.output

    def test_post_event_requires_as_check(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "-w", "adhoc", "--role", "engineer",
            "--post-event", "monitor/test", "--task", "Check prod URL",
        ])
        assert result.exit_code != 0
        assert "--post-event requires --as-check" in result.output

    def test_wait_non_adhoc_fails_loudly(self, bobi_install):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "subagents", "launch",
            "-w", "issue-lifecycle", "--role", "engineer",
            "--wait", "--task", "Fix #1",
        ])
        assert result.exit_code != 0
        # Names the limit, the workflow that tripped it, WHY it exists, and the
        # pattern to use instead - an agent that hits this must be able to
        # recover without reading the source (#845).
        assert "--wait requires '-w adhoc'" in result.output
        assert "issue-lifecycle" in result.output
        assert "fan out" in result.output
        assert "wait" in result.output

    def test_passes_requested_by(self, bobi_install):
        req = '{"requester":"Alice","source":{"kind":"test"},"ids":[1,2]}'
        with patch("bobi.subagent.launch_agent", return_value="wf-adhoc-1") as mock:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "subagents", "launch",
                "-w", "adhoc", "--role", "engineer",
                "--task", "Fix #1", "--requested-by", req,
            ])
        assert result.exit_code == 0, result.output
        assert mock.call_args[1]["task"] == "Fix #1"
        assert mock.call_args[1]["requested_by"] == {
            "requester": "Alice",
            "source": {"kind": "test"},
            "ids": [1, 2],
        }


class TestMonitorGate:
    """`monitors gate` is the scheduler's relevance-gate plumbing (#630):
    read the request file, run the gate agent, print the verdict line."""

    def _request(self, tmp_path, **overrides):
        payload = {"criterion": "about billing", "name": "gate-billing",
                   "items": [{"key": "m1", "data": {"subject": "refund"}}]}
        payload.update(overrides)
        req = tmp_path / "req.json"
        req.write_text(json.dumps(payload))
        return req

    def test_prints_verdict_line(self, bobi_install, tmp_path):
        gate = GateResult(success=True, relevant=["m1"])
        with patch("bobi.subagent.run_gate_blocking", return_value=gate) as mock:
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "monitors", "gate",
                "--request", str(self._request(tmp_path)),
            ])
        assert result.exit_code == 0, result.output
        verdict = json.loads(result.output.strip().splitlines()[-1])
        assert verdict == {"success": True, "relevant": ["m1"]}
        assert mock.call_args[0][0] == "about billing"
        assert mock.call_args[0][1] == [{"key": "m1",
                                         "data": {"subject": "refund"}}]
        assert mock.call_args[1]["name"] == "gate-billing"

    def test_gate_failure_exits_nonzero_with_verdict(self, bobi_install, tmp_path):
        gate = GateResult(success=False, error="no verdict")
        with patch("bobi.subagent.run_gate_blocking", return_value=gate):
            result = CliRunner().invoke(main, [
                "agent", TEST_AGENT_NAME, "monitors", "gate",
                "--request", str(self._request(tmp_path)),
            ])
        assert result.exit_code == 1
        # The verdict line still prints so the scheduler parses "success": false.
        assert '"success": false' in result.output

    def test_missing_request_file_fails(self, bobi_install, tmp_path):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "monitors", "gate",
            "--request", str(tmp_path / "nope.json"),
        ])
        assert result.exit_code == 1

    def test_empty_items_rejected(self, bobi_install, tmp_path):
        result = CliRunner().invoke(main, [
            "agent", TEST_AGENT_NAME, "monitors", "gate",
            "--request", str(self._request(tmp_path, items=[])),
        ])
        assert result.exit_code == 1


class TestEventsCommand:
    def _run_events(self, bobi_install):
        return CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "events"])

    def test_skips_malformed_lines_in_events_jsonl(self, bobi_install):
        good = {"timestamp": "2026-01-01T00:00:00", "source": "github",
                "type": "push", "data": {}}
        (bobi_install.state_dir / "events-default.jsonl").write_text(
            json.dumps(good) + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({**good, "type": "pr"}) + "\n"
        )
        result = self._run_events(bobi_install)
        assert result.exit_code == 0, result.output
        assert "push" in result.output
        assert "pr" in result.output
        assert "1 malformed" in result.output

    def test_skips_malformed_lines_in_decisions_jsonl(self, bobi_install):
        good = {"timestamp": "2026-01-01T00:00:00",
                "actions": [{"type": "deploy"}], "reasoning": "ship it"}
        (bobi_install.state_dir / "decisions.jsonl").write_text(
            json.dumps(good) + "\nCORRUPTED\n"
        )
        result = self._run_events(bobi_install)
        assert result.exit_code == 0, result.output
        assert "deploy" in result.output
        assert "1 malformed" in result.output

    def test_orders_mixed_era_timestamps_by_instant_not_string(self, bobi_install,
                                                                monkeypatch):
        # An events file spans the aware-UTC upgrade: pre-upgrade lines carry
        # naive LOCAL timestamps, post-upgrade lines aware UTC. On a UTC+9
        # host the newer UTC string sorts lexicographically BEFORE the older
        # local one; ordering (and --tail selection) must go by instant.
        import time as _time
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        _time.tzset()
        try:
            older_local = {"timestamp": "2026-01-01T18:00:00",  # 09:00 UTC
                           "source": "github", "type": "old-evt", "data": {}}
            newer_aware = {"timestamp": "2026-01-01T10:00:00+00:00",
                           "source": "github", "type": "new-evt", "data": {}}
            (bobi_install.state_dir / "events-default.jsonl").write_text(
                json.dumps(older_local) + "\n" + json.dumps(newer_aware) + "\n")
            result = self._run_events(bobi_install)
            assert result.exit_code == 0, result.output
            assert result.output.index("old-evt") < result.output.index("new-evt")
        finally:
            monkeypatch.undo()
            _time.tzset()

    def test_deduplicates_events_by_seq_deployment(self, bobi_install):
        ev = {"timestamp": "2026-01-01T00:00:01", "source": "github",
              "type": "push", "seq": 5, "deployment_id": "d1"}
        (bobi_install.state_dir / "events-sess-a.jsonl").write_text(json.dumps(ev) + "\n")
        (bobi_install.state_dir / "events-sess-b.jsonl").write_text(json.dumps(ev) + "\n")
        result = self._run_events(bobi_install)
        assert result.exit_code == 0, result.output
        assert result.output.count("push") == 1

    def test_payload_event_renders_text(self, bobi_install):
        ev = {
            "timestamp": "2026-01-01T00:00:01",
            "source": "inbox",
            "type": "message",
            "payload": {"sender": "alice", "text": "hello world"},
        }
        (bobi_install.state_dir / "events-sess-a.jsonl").write_text(json.dumps(ev) + "\n")
        result = self._run_events(bobi_install)
        assert result.exit_code == 0, result.output
        assert "alice" in result.output
        assert "hello world" in result.output

    def test_ignores_legacy_events_jsonl(self, bobi_install):
        legacy = {"timestamp": "2026-01-01T00:00:01", "source": "github",
                  "type": "legacy_push"}
        session = {"timestamp": "2026-01-01T00:00:02", "source": "github",
                   "type": "new_pr", "seq": 1, "deployment_id": "d1"}
        (bobi_install.state_dir / "events.jsonl").write_text(json.dumps(legacy) + "\n")
        (bobi_install.state_dir / "events-sess-a.jsonl").write_text(json.dumps(session) + "\n")
        result = self._run_events(bobi_install)
        assert result.exit_code == 0, result.output
        assert "legacy_push" not in result.output
        assert "new_pr" in result.output

    def test_publish_reads_payload_from_stdin(self, bobi_install):
        with patch("bobi.events.publish.post_event", return_value=True) as post:
            result = CliRunner().invoke(
                main,
                ["agent", TEST_AGENT_NAME, "events", "publish", "alert/firing"],
                input='{"title":"x"}',
            )

        assert result.exit_code == 0, result.output
        assert "Published alert/firing" in result.output
        post.assert_called_once_with(
            "alert/firing",
            {"title": "x"},
            project_path=bobi_install.repo_path,
        )

    def test_publish_reads_payload_from_json_option(self, bobi_install):
        with patch("bobi.events.publish.post_event", return_value=True) as post:
            result = CliRunner().invoke(
                main,
                [
                    "agent", TEST_AGENT_NAME, "events", "publish",
                    "alert/firing", "--json", '{"title":"x"}',
                ],
            )

        assert result.exit_code == 0, result.output
        post.assert_called_once_with(
            "alert/firing",
            {"title": "x"},
            project_path=bobi_install.repo_path,
        )

    def test_publish_rejects_non_object_payload(self, bobi_install):
        result = CliRunner().invoke(
            main,
            ["agent", TEST_AGENT_NAME, "events", "publish", "alert/firing"],
            input='["x"]',
        )

        assert result.exit_code != 0
        assert "Payload must be a JSON object" in result.output

    def test_publish_rejects_bare_topic(self, bobi_install):
        result = CliRunner().invoke(
            main,
            [
                "agent", TEST_AGENT_NAME, "events", "publish",
                "firing", "--json", '{"title":"x"}',
            ],
        )

        assert result.exit_code != 0
        assert "source/type" in result.output

    def test_publish_rejects_global_topic_prefixes(self, bobi_install):
        for topic in [
            "github:org/repo",
            "linear:TEAM/firing",
            "slack:T123/firing",
            "alert/github:org",
        ]:
            result = CliRunner().invoke(
                main,
                [
                    "agent", TEST_AGENT_NAME, "events", "publish",
                    topic, "--json", '{"title":"x"}',
                ],
            )

            assert result.exit_code != 0
            assert "reserved for webhooks" in result.output

    def test_publish_rejects_webhook_source_labels(self, bobi_install):
        for topic in [
            "github/firing",
            "linear/firing",
            "slack/firing",
        ]:
            result = CliRunner().invoke(
                main,
                [
                    "agent", TEST_AGENT_NAME, "events", "publish",
                    topic, "--json", '{"title":"x"}',
                ],
            )

            assert result.exit_code != 0
            assert "sources are reserved for webhooks" in result.output

    def test_publish_without_payload_does_not_read_interactive_stdin(
        self,
        bobi_install,
        monkeypatch,
    ):
        class TtyStdin:
            @staticmethod
            def isatty():
                return True

            @staticmethod
            def read():
                raise AssertionError("interactive stdin should not be read")

        monkeypatch.setattr("click.get_text_stream", lambda name: TtyStdin())
        result = CliRunner().invoke(
            main,
            ["agent", TEST_AGENT_NAME, "events", "publish", "alert/firing"],
        )

        assert result.exit_code != 0
        assert "Provide payload with --json or stdin" in result.output

    def test_publish_reports_rejected_publish(self, bobi_install):
        with patch("bobi.events.publish.post_event", return_value=False):
            result = CliRunner().invoke(
                main,
                [
                    "agent", TEST_AGENT_NAME, "events", "publish",
                    "alert/firing", "--json", '{"title":"x"}',
                ],
            )

        assert result.exit_code != 0
        assert "Publish failed" in result.output
        assert "bubble credentials" in result.output


class TestEventServerCommand:
    def test_status_uses_selected_runtime_port_file(self, bobi_install, monkeypatch):
        (bobi_install.state_dir / "event-server.pid").write_text("12345")
        (bobi_install.state_dir / "event-server.port").write_text("58405")

        seen = []

        def fake_health(url):
            seen.append(url)
            if url == "http://localhost:58405":
                return {"status": "ok", "mode": "local", "deployments": 2}
            return None

        monkeypatch.setattr("bobi.events.server.health", fake_health)

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "event-server", "status"])

        assert result.exit_code == 0, result.output
        assert "running on port 58405" in result.output
        assert seen == ["http://localhost:58405"]

    def test_start_uses_configured_local_event_server_port(self, bobi_install, monkeypatch):
        paths.agent_yaml_path(bobi_install.repo_path).write_text(
            "agent: test-agent\n"
            "entry_point: director\n"
            "event_server: http://localhost:17777\n"
        )
        called = {}

        def fake_ensure_running(port, project_path=None):
            called["port"] = port
            called["project_path"] = project_path
            return "connected"

        monkeypatch.setattr("bobi.events.server.ensure_running", fake_ensure_running)

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "event-server", "start"])

        assert result.exit_code == 0, result.output
        assert called == {"port": 17777, "project_path": bobi_install.repo_path}
        assert "port 17777" in result.output

    def test_start_surfaces_packaged_artifact_reinstall_guidance(
        self, bobi_install, monkeypatch,
    ):
        from bobi.events.server import PackagedEventServerArtifactError

        monkeypatch.setattr(
            "bobi.events.server.ensure_running",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                PackagedEventServerArtifactError(
                    "The installed local event-server artifact is corrupt. "
                    "Reinstall or upgrade Bobi."
                )
            ),
        )

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "event-server", "start"]
        )

        assert result.exit_code != 0
        assert "Reinstall or upgrade Bobi" in result.output
        assert "event-server artifact is corrupt" in result.output

    def test_stop_warning_uses_selected_runtime_port(self, bobi_install, monkeypatch):
        (bobi_install.state_dir / "event-server.port").write_text("58405")

        def fake_health(url):
            assert url == "http://localhost:58405"
            return {"status": "ok", "mode": "local", "deployments": 1}

        monkeypatch.setattr("bobi.events.server.health", fake_health)

        result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "stop"])

        assert result.exit_code == 0, result.output
        assert "Event server is still running on port 58405" in result.output

    # D018 — a pid file is written by another process and can be truncated by a
    # crash mid-write. `int(pid_file.read_text())` sat outside the try, so stop
    # raised ValueError, printed a traceback, and left the stale files in
    # place — making every subsequent stop fail the same way. The manager stop
    # path (cli.py `Invalid PID file — cleaning up.`) already defends against
    # exactly this.

    @pytest.mark.parametrize("contents", ["", "   ", "not-a-pid", "12345\n67890"])
    def test_stop_cleans_up_a_corrupt_pid_file(self, bobi_install, contents):
        pid_file = bobi_install.state_dir / "event-server.pid"
        port_file = bobi_install.state_dir / "event-server.port"
        pid_file.write_text(contents)
        port_file.write_text("58405")

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "event-server", "stop"])

        assert result.exception is None or isinstance(result.exception, SystemExit), \
            f"unhandled traceback: {result.exception!r}"
        assert "Traceback" not in result.output
        assert not pid_file.exists(), "stale pid file left behind"
        assert not port_file.exists(), "stale port file left behind"

    def test_stop_reports_a_pid_it_may_not_signal(self, bobi_install, monkeypatch):
        # A pid owned by another user: os.kill raises PermissionError, which was
        # uncaught. Report it and still clear our own stale files.
        (bobi_install.state_dir / "event-server.pid").write_text("4242")
        (bobi_install.state_dir / "event-server.port").write_text("58405")

        def deny(pid, sig):
            raise PermissionError(1, "Operation not permitted")
        monkeypatch.setattr("os.kill", deny)

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "event-server", "stop"])

        assert result.exception is None or isinstance(result.exception, SystemExit), \
            f"unhandled traceback: {result.exception!r}"
        assert "Traceback" not in result.output


class TestSetupCommand:
    def _home(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("BOBI_HOME", str(home))
        return home

    def _patch_app(self, monkeypatch):
        seen = {}

        monkeypatch.setattr(
            "bobi.webapp.daemon.start",
            lambda open_browser=True: seen.setdefault("open_browser", open_browser)
            or type("Status", (), {"url": "http://127.0.0.1:8642/?n=tok",
                                   "pid": 1234})(),
        )
        monkeypatch.setattr("webbrowser.open",
                            lambda url: seen.setdefault("url", url))
        return seen

    def test_setup_opens_named_unified_app_route(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        seen = self._patch_app(monkeypatch)

        result = CliRunner().invoke(main, ["setup", "alpha"])

        assert result.exit_code == 0, result.output
        assert seen["open_browser"] is False
        assert seen["url"] == "http://127.0.0.1:8642/?n=tok#/setup/alpha"
        assert "bobi setup is open at" in result.output

    def test_help(self):
        result = CliRunner().invoke(main, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        # Q122/D064 — `--resume` was parsed, advertised in help, documented in
        # QUICKSTART, then discarded with `del resume`. Setup reopens through
        # the webapp, which resumes an unfinished session unconditionally
        # (webapp/server.py), so the flag named the default and promised a
        # choice that did not exist. Removed rather than left as dead surface.
        assert "--resume" not in result.output
        assert "resumes where you left off" in result.output

    def test_resume_flag_is_gone(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        self._patch_app(monkeypatch)

        result = CliRunner().invoke(main, ["setup", "alpha", "--resume"])

        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_setup_without_name_opens_create_route(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        seen = self._patch_app(monkeypatch)

        result = CliRunner().invoke(main, ["setup"])

        assert result.exit_code == 0, result.output
        assert seen["url"] == "http://127.0.0.1:8642/?n=tok#/setup"

    def test_model_option_reaches_the_setup_url(self, tmp_path, monkeypatch):
        self._home(tmp_path, monkeypatch)
        seen = self._patch_app(monkeypatch)

        result = CliRunner().invoke(
            main, ["setup", "alpha", "--model", "sonnet"])

        assert result.exit_code == 0, result.output
        assert seen["url"] == (
            "http://127.0.0.1:8642/?n=tok#/setup/alpha?model=sonnet")


class TestMonitorAdd:
    def _add(self, bobi_install, args):
        return CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "monitors", "add", *args])

    def _written(self, bobi_install):
        import yaml
        path = paths.package_dir(bobi_install.repo_path) / "monitors.yaml"
        return yaml.safe_load(path.read_text())["monitors"]

    def test_interval_monitor_still_works(self, bobi_install):
        result = self._add(bobi_install, [
            "pr check", "--interval", "15m", "--description", "check PRs"])
        assert result.exit_code == 0, result.output
        rec = self._written(bobi_install)[0]
        assert rec["name"] == "pr-check"
        assert rec["interval"] == "15m"
        assert "at" not in rec

    def test_weekly_notify_monitor_writes_at_days_tz(self, bobi_install):
        result = self._add(bobi_install, [
            "weekly-prep-doc", "--at", "21:00", "--days", "sun",
            "--tz", "America/Los_Angeles", "--notify",
            "--event", "monitor/prep.weekly_due",
            "--description", "Generate my prep doc for the upcoming week",
        ])
        assert result.exit_code == 0, result.output
        rec = self._written(bobi_install)[0]
        assert rec["name"] == "weekly-prep-doc"
        assert rec["at"] == ["21:00"]
        assert rec["days"] == ["sun"]
        assert rec["tz"] == "America/Los_Angeles"
        assert rec["notify"] is True
        assert rec["event"] == "monitor/prep.weekly_due"
        assert "interval" not in rec

    def test_interval_and_at_are_mutually_exclusive(self, bobi_install):
        result = self._add(bobi_install, ["x", "--interval", "5m", "--at", "21:00"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_days_without_at_is_rejected(self, bobi_install):
        result = self._add(bobi_install, ["x", "--days", "sun"])
        assert result.exit_code != 0
        assert "--days only applies to --at" in result.output

    def test_invalid_at_time_is_rejected(self, bobi_install):
        result = self._add(bobi_install, ["x", "--at", "25:00"])
        assert result.exit_code != 0
        assert "at-time" in result.output

    def test_invalid_weekday_is_rejected(self, bobi_install):
        result = self._add(bobi_install, ["x", "--at", "21:00", "--days", "funday"])
        assert result.exit_code != 0
        assert "weekday" in result.output.lower()


class TestAgentsUpdateAndBrowse:
    """D066/D065 — the machine-facing `agents` surface."""

    def test_update_all_exits_nonzero_when_every_pack_fails(self, bobi_install,
                                                            monkeypatch):
        # D066: the named-pack path exits 1 on this exact failure, the
        # update-all path returned 0, so the two forms reported contradictory
        # exit codes for identical failures and CI could not detect it.
        monkeypatch.setattr(
            "bobi.registry.list_cached",
            lambda: [{"name": "alpha"}, {"name": "beta"}])

        def boom(*a, **k):
            raise RuntimeError("registry unreachable")
        monkeypatch.setattr("bobi.registry.check_update", boom)

        result = CliRunner().invoke(main, ["agents", "update"])

        assert result.exit_code == 1, result.output
        assert "alpha — failed" in result.output
        assert "beta — failed" in result.output

    def test_update_all_still_exits_zero_when_a_pack_succeeds(self, bobi_install,
                                                              monkeypatch):
        monkeypatch.setattr(
            "bobi.registry.list_cached",
            lambda: [{"name": "alpha"}, {"name": "beta"}])

        def check(name, *a, **k):
            if name == "beta":
                raise RuntimeError("registry unreachable")
            return ("1.0.0", "1.0.0")
        monkeypatch.setattr("bobi.registry.check_update", check)

        result = CliRunner().invoke(main, ["agents", "update"])

        assert result.exit_code == 1, "one failure is still a failure"
        assert "alpha v1.0.0 — up to date" in result.output

    def test_browse_survives_an_unquoted_numeric_version(self, bobi_install,
                                                         monkeypatch):
        # D065: a third-party registry.yaml with `version: 1.0` parses as a
        # float, and `f"v{version:8s}"` dies with "Unknown format code 's'",
        # taking down the whole listing instead of one row.
        monkeypatch.setattr(
            "bobi.registry.list_remote",
            lambda *a, **k: [{"name": "numeric", "version": 1.0,
                              "description": "unquoted version"},
                             {"name": "ordinary", "version": "2.1.0",
                              "description": "quoted version"}])
        monkeypatch.setattr("bobi.registry.list_cached", lambda: [])

        result = CliRunner().invoke(main, ["agents", "browse"])

        assert result.exception is None or isinstance(result.exception, SystemExit), \
            f"unhandled traceback: {result.exception!r}"
        assert result.exit_code == 0, result.output
        assert "numeric" in result.output and "ordinary" in result.output

    def test_browse_matches_a_local_version_against_a_numeric_remote(
            self, bobi_install, monkeypatch):
        # The same coercion fixes the silent str-vs-float mismatch that made an
        # installed pack read as an available upgrade to itself.
        monkeypatch.setattr(
            "bobi.registry.list_remote",
            lambda *a, **k: [{"name": "numeric", "version": 1.0}])
        monkeypatch.setattr(
            "bobi.registry.list_cached",
            lambda: [{"name": "numeric", "version": "1.0"}])

        result = CliRunner().invoke(main, ["agents", "browse"])

        assert result.exit_code == 0, result.output
        assert "[installed]" in result.output
        assert "available" not in result.output


class TestFindTranscript:
    """`bobi logs`' transcript fallback, on the shared locator (Q027).

    This fallback was the fourth hand-rolled copy of "find <session>.jsonl"
    and the only one that ignored CLAUDE_CONFIG_DIR, so `bobi logs` printed
    "No session" for exactly the agents bobi itself runs under a per-team
    config dir (#779).
    """

    def _bind(self, tmp_path, monkeypatch, session_id):
        from bobi import paths
        from bobi.cli import _find_transcript

        monkeypatch.setattr("bobi.cli._detect_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            "bobi.sdk._sessions_dir",
            lambda root=None: paths.sessions_dir(tmp_path))
        sessions = paths.sessions_dir(tmp_path)
        sessions.mkdir(parents=True, exist_ok=True)
        (sessions / "worker.id").write_text(session_id + "\n")
        return _find_transcript

    def test_finds_transcript_under_claude_config_dir(self, tmp_path, monkeypatch):
        find = self._bind(tmp_path, monkeypatch, "sess-cli")
        target = tmp_path / "cfg" / "projects" / "proj" / "sess-cli.jsonl"
        target.parent.mkdir(parents=True)
        target.write_text("{}\n")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

        assert find("worker") == target

    def test_missing_transcript_returns_none(self, tmp_path, monkeypatch):
        find = self._bind(tmp_path, monkeypatch, "sess-absent")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

        assert find("worker") is None

    def test_blank_session_id_returns_none(self, tmp_path, monkeypatch):
        find = self._bind(tmp_path, monkeypatch, "")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

        assert find("worker") is None


class TestRestartCommand:
    def test_restart_delegates_without_stopping_in_caller(
        self, bobi_install, monkeypatch,
    ):
        from bobi import service

        seen = {}

        def fake_restart(project_path, *, fresh=False, **kwargs):
            seen["project_path"] = project_path
            seen["fresh"] = fresh
            return service.RestartResult(
                pid=4242,
                log_file=bobi_install.state_dir / "restart.log",
                output="Restart worker finished.\n",
            )

        monkeypatch.setattr("bobi.cli._has_systemd_service", lambda: False)
        monkeypatch.setattr(service, "restart_team", fake_restart)
        monkeypatch.setattr(
            service,
            "stop_team",
            lambda *args, **kwargs: pytest.fail("restart stopped in the caller"),
        )

        result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "restart"])

        assert result.exit_code == 0, result.output
        assert seen == {"project_path": bobi_install.repo_path, "fresh": False}
        assert "Restart worker finished." in result.output

    def test_restart_forwards_fresh(self, bobi_install, monkeypatch):
        from bobi import service

        seen = {}

        def fake_restart(project_path, *, fresh=False, **kwargs):
            seen["fresh"] = fresh
            return service.RestartResult(
                pid=4242,
                log_file=bobi_install.state_dir / "restart.log",
            )

        monkeypatch.setattr("bobi.cli._has_systemd_service", lambda: False)
        monkeypatch.setattr(service, "restart_team", fake_restart)

        result = CliRunner().invoke(
            main, ["agent", TEST_AGENT_NAME, "restart", "--fresh"]
        )

        assert result.exit_code == 0, result.output
        assert seen["fresh"] is True

    def test_restart_reports_worker_failure(self, bobi_install, monkeypatch):
        from bobi import service

        def fail_restart(project_path, **kwargs):
            raise service.RestartFailed(
                "restart failed (worker exit 1)",
                bobi_install.state_dir / "restart.log",
                "missing SLACK_BOT_TOKEN",
            )

        monkeypatch.setattr("bobi.cli._has_systemd_service", lambda: False)
        monkeypatch.setattr(service, "restart_team", fail_restart)

        result = CliRunner().invoke(main, ["agent", TEST_AGENT_NAME, "restart"])

        assert result.exit_code == 1
        assert "worker exit 1" in result.output
        assert "missing SLACK_BOT_TOKEN" in result.output

    def test_detached_worker_runs_stop_then_start(self, bobi_install, monkeypatch):
        from bobi import service

        calls = []
        monkeypatch.setattr("bobi.cli._has_systemd_service", lambda: False)
        monkeypatch.setattr(
            service,
            "stop_team",
            lambda root, **kwargs: calls.append("stop")
            or service.StopResult(pid=42, stopped=True),
        )
        monkeypatch.setattr(
            service,
            "spawn_team",
            lambda root, **kwargs: calls.append(("start", kwargs["fresh"]))
            or SimpleNamespace(
                startup=SimpleNamespace(pid=43, log_file=Path("manager.log")),
                validation=SimpleNamespace(ok=True, checks=[]),
                image_rotated=False,
            ),
        )
        monkeypatch.setattr(
            service,
            "restart_team",
            lambda *args, **kwargs: pytest.fail("worker delegated recursively"),
        )
        monkeypatch.setattr("bobi.events.server.health", lambda url: None)
        monkeypatch.setattr("bobi.cli._print_startup_info", lambda *args: None)

        result = CliRunner().invoke(
            main,
            ["agent", TEST_AGENT_NAME, "restart", "--detached-worker", "--fresh"],
        )

        assert result.exit_code == 0, result.output
        assert calls == ["stop", ("start", True)]
        assert "Restart worker finished." in result.output
