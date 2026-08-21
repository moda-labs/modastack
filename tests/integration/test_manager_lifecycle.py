"""Integration tests for manager start/stop lifecycle.

Exercises the full named start → status → stop cycle via the CLI against the
isolated install. Runs on BOTH brains (``dual_brain_env``): the public stub
(fast lane, always) and real Claude (gated on the ``claude`` CLI). These are
runtime-plumbing assertions (pid/log/status/restart, drain-loop readiness), so
the stub proves them deterministically in CI while the Claude leg still exercises
a real manager locally. The same stub brain drives the private sidecar e2e.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from bobi.sdk import DEAD_STATUSES


# Bind this file's ``bobi_env`` / ``cli_run`` to the dual-brain (stub + claude)
# variants, so every lifecycle test below runs once per brain without touching
# its body. The autouse binder in conftest still sees "bobi_env" in the fixture
# graph, resolves it to the selected env, and pins the stub brain on that leg.
@pytest.fixture
def bobi_env(dual_brain_env):
    return dual_brain_env


@pytest.fixture
def cli_run(dual_brain_cli_run):
    return dual_brain_cli_run


@pytest.mark.timeout(120)
class TestManagerStartStop:
    def test_launch_team_service_starts_manager(self, bobi_env):
        from bobi.service import launch_team, stop_team

        pid_file = bobi_env.state_dir / "manager.pid"
        try:
            entry = launch_team(bobi_env.project_path, wait_timeout=60)
            assert entry.name == f"bobi-{bobi_env.agent_name}-manager"
            assert entry.status in ("starting", "running", "idle")
            assert pid_file.exists(), "PID file not created after service launch"
        finally:
            stop_team(bobi_env.project_path)
            _wait_for_exit_file(pid_file)

    def test_start_creates_pid_file(self, bobi_env, cli_run):
        result = cli_run("start", timeout=15)
        assert result.returncode == 0, f"start failed: {result.stderr}"

        pid_file = bobi_env.state_dir / "manager.pid"

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                break
            time.sleep(0.5)

        assert pid_file.exists(), "PID file not created after start"
        pid = int(pid_file.read_text().strip())
        assert pid > 0

        # Clean up
        os.kill(pid, signal.SIGTERM)
        _wait_for_exit(pid)
        pid_file.unlink(missing_ok=True)

    def test_start_writes_log(self, bobi_env, cli_run):
        result = cli_run("start", timeout=15)
        assert result.returncode == 0

        log_file = bobi_env.state_dir / "manager.log"

        deadline = time.monotonic() + 15
        has_log = False
        while time.monotonic() < deadline:
            if log_file.exists() and log_file.stat().st_size > 0:
                has_log = True
                break
            time.sleep(0.5)

        assert has_log, "Manager log file not written"
        content = log_file.read_text()
        assert "Bobi" in content or "starting" in content.lower()

        # Clean up
        pid_file = bobi_env.state_dir / "manager.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
            _wait_for_exit_file(pid_file)
            pid_file.unlink(missing_ok=True)

    def test_stop_removes_pid_file(self, bobi_env, cli_run):
        cli_run("start", timeout=45)

        pid_file = bobi_env.state_dir / "manager.pid"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                break
            time.sleep(0.5)
        assert pid_file.exists()

        result = cli_run("stop", timeout=15)
        assert result.returncode == 0
        assert "stopped" in result.stdout.lower() or "stopping" in result.stdout.lower()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not pid_file.exists():
                break
            time.sleep(0.3)

        assert not pid_file.exists(), "PID file not cleaned up after stop"

    def test_stop_when_not_running(self, bobi_env, cli_run):
        pid_file = bobi_env.state_dir / "manager.pid"
        pid_file.unlink(missing_ok=True)

        result = cli_run("stop", timeout=5)
        assert result.returncode == 0
        assert "not running" in result.stdout.lower()

    def test_start_rejects_double_start(self, bobi_env, cli_run):
        cli_run("start", timeout=45)

        pid_file = bobi_env.state_dir / "manager.pid"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                break
            time.sleep(0.5)

        result = cli_run("start", timeout=5)
        assert "already running" in result.stdout.lower()

        # Clean up
        cli_run("stop", timeout=15)
        _wait_for_exit_file(pid_file)

    def test_status_shows_running_after_start(self, bobi_env, cli_run):
        cli_run("start", timeout=45)

        pid_file = bobi_env.state_dir / "manager.pid"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                break
            time.sleep(0.5)

        result = cli_run("status", timeout=5)
        assert result.returncode == 0

        cli_run("stop", timeout=15)
        _wait_for_exit_file(pid_file)

    def test_restart_swaps_pid(self, bobi_env, cli_run):
        cli_run("start", timeout=45)

        pid_file = bobi_env.state_dir / "manager.pid"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if pid_file.exists():
                break
            time.sleep(0.5)

        old_pid = int(pid_file.read_text().strip())

        cli_run("restart", timeout=30)

        deadline = time.monotonic() + 15
        new_pid = old_pid
        while time.monotonic() < deadline:
            if pid_file.exists():
                try:
                    new_pid = int(pid_file.read_text().strip())
                    if new_pid != old_pid:
                        break
                except (ValueError, OSError):
                    pass
            time.sleep(0.5)

        assert new_pid != old_pid, "PID should change after restart"

        cli_run("stop", timeout=15)
        _wait_for_exit_file(pid_file)


@pytest.mark.timeout(120)
def test_restart_completes_when_caller_dies(stub_bobi_env, stub_cli_run):
    pid_file = stub_bobi_env.state_dir / "manager.pid"
    caller = None
    try:
        started = stub_cli_run("start", timeout=45)
        assert started.returncode == 0, started.stderr
        old_pid = _wait_for_pid(pid_file)

        caller_env = {
            **os.environ,
            "BOBI_HOME": str(stub_bobi_env.home_dir),
            "BOBI_EVENT_SERVER": stub_bobi_env.event_server_url,
            "BOBI_BRAIN": "stub",
            "BOBI_STUB_BRAIN": "1",
        }
        caller = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bobi.cli",
                "agent",
                stub_bobi_env.agent_name,
                "restart",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(stub_bobi_env.project_path),
            env=caller_env,
        )

        assert _kill_caller_when_manager_stops(caller, pid_file, old_pid), (
            "restart caller exited before the manager stopped"
        )
        new_pid = _wait_for_pid(pid_file, timeout=60, other_than=old_pid)
        os.kill(new_pid, 0)

        record = (stub_bobi_env.state_dir / "restart.log").read_text()
        assert "Restart worker finished." in record
    finally:
        if caller is not None and caller.poll() is None:
            caller.kill()
            caller.wait(timeout=10)
        stub_cli_run("stop", timeout=30)
        _wait_for_exit_file(pid_file)


@pytest.mark.timeout(180)
class TestManagerMessaging:
    """Tests that require a fully booted manager with drain loop active."""

    @pytest.fixture(autouse=True)
    def _start_and_stop(self, bobi_env, cli_run):
        log_file = bobi_env.state_dir / "manager.log"
        pid_file = bobi_env.state_dir / "manager.pid"

        # Record log position before start so we only check new output
        log_pos = log_file.stat().st_size if log_file.exists() else 0

        cli_run("start", timeout=45)

        deadline = time.monotonic() + 60
        ready = False
        while time.monotonic() < deadline:
            if pid_file.exists() and log_file.exists():
                new_content = log_file.read_text()[log_pos:]
                if "Drain loop active" in new_content:
                    ready = True
                    break
            time.sleep(1)

        if not ready:
            new_content = log_file.read_text()[log_pos:] if log_file.exists() else "(no log)"
            # fail, not skip: these are the only checks in the suite that the
            # drain loop actually starts, and a skip turns a boot regression
            # into green CI (D012).
            pytest.fail(
                "Manager did not become ready (needs Node 20 + `npm ci` in "
                f"event-server/ for the drain loop to start): {new_content[-300:]}"
            )

        yield

        cli_run("stop", timeout=15)
        _wait_for_exit_file(pid_file)

    def test_message_and_ask(self, cli_run):
        result = cli_run("message", "hello from integration test", timeout=30)
        assert result.returncode == 0
        assert "sent" in result.stdout.lower()

        result = cli_run("ask", "Reply with just: INTEGRATION_OK", "--timeout", "90", timeout=120)
        assert result.returncode == 0, f"ask failed: stderr={result.stderr}"
        assert len(result.stdout.strip()) > 0


# Addressing a manager that is not running fails through one of two honest
# shapes: no addressable session at all, or a registry entry that is already
# terminal - and an orderly teardown drops the pid, so `deliver()` names the
# terminal STATUS rather than the dead process. Deriving the terminal phrasings
# from DEAD_STATUSES (the same constant deliver() reports from) keeps this
# assertion pinned to the contract instead of to one incidental wording.
_NOT_RUNNING_MESSAGES = (
    "not running", "no active session", "cannot reach", "process is dead",
    *(f"is {status}" for status in DEAD_STATUSES),
)


@pytest.mark.timeout(30)
class TestManagerNotRunning:
    """Tests for message/ask when the manager is stopped."""

    def test_message_when_not_running(self, bobi_env, cli_run):
        pid_file = bobi_env.state_dir / "manager.pid"
        pid_file.unlink(missing_ok=True)

        result = cli_run("message", "should fail", timeout=5)
        output = (result.stdout + result.stderr).lower()
        assert result.returncode != 0
        assert any(msg in output for msg in _NOT_RUNNING_MESSAGES), output

    def test_ask_when_not_running(self, bobi_env, cli_run):
        pid_file = bobi_env.state_dir / "manager.pid"
        pid_file.unlink(missing_ok=True)

        result = cli_run("ask", "should fail", timeout=5)
        assert result.returncode != 0


def _wait_for_pid(pid_file, timeout: float = 15, other_than: int = 0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, OSError, ProcessLookupError):
            pid = 0
        if pid and pid != other_than:
            return pid
        time.sleep(0.1)
    raise TimeoutError(f"{pid_file} never held a live pid other than {other_than}")


def _kill_caller_when_manager_stops(proc, pid_file, old_pid: int,
                                    timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            current_pid = int(pid_file.read_text().strip())
        except (ValueError, OSError):
            current_pid = 0
        if current_pid != old_pid:
            if proc.poll() is not None:
                return False
            proc.kill()
            proc.wait(timeout=10)
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    raise TimeoutError(f"manager pid file never changed from {old_pid}")


def _wait_for_exit(pid: int, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.3)
        except ProcessLookupError:
            return
    raise TimeoutError(f"Process {pid} did not exit within {timeout}s")


def _wait_for_exit_file(pid_file, timeout: float = 10):
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
            time.sleep(0.3)
        except ProcessLookupError:
            return
