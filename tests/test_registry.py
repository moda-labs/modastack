"""Tests for session registry — file-per-worker model."""

import builtins
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import bobi.sdk as sdk
from bobi import paths
from bobi.sdk import SessionEntry, SessionRegistry


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    paths.bind_root(None)
    paths.package_dir(tmp_path).mkdir(parents=True)
    paths.agent_yaml_path(tmp_path).write_text("agent: test\n")
    paths.bind_root(tmp_path)
    yield SessionRegistry()
    paths.bind_root(None)


class TestSessionEntry:
    def test_defaults(self):
        e = SessionEntry(name="test")
        assert e.role == ""
        assert e.status == "starting"
        assert e.session_id == ""

    def test_manager_role(self):
        e = SessionEntry(name="moda-manager", role="manager")
        assert e.role == "manager"

    def test_requested_by_defaults_empty(self):
        e = SessionEntry(name="test")
        assert e.requested_by == {}

    def test_requested_by_roundtrips(self, tmp_registry):
        requester = {"from": "Alice", "user_id": "U1", "channel": "C1"}
        tmp_registry.register(SessionEntry(name="adhoc-x", requested_by=requester))
        got = tmp_registry.get("adhoc-x")
        assert got.requested_by == requester


class TestSessionRegistry:
    def test_register_and_get(self, tmp_registry):
        entry = SessionEntry(name="agent-42", run_key="42", phase="pickup")
        tmp_registry.register(entry)
        got = tmp_registry.get("agent-42")
        assert got is not None
        assert got.run_key == "42"

    def test_update(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="agent-1", status="starting"))
        tmp_registry.update("agent-1", status="running")
        got = tmp_registry.get("agent-1")
        assert got.status == "running"

    def test_state_writes_never_expose_partial_json(
            self, tmp_registry, tmp_path, monkeypatch):
        """A published state file is always one complete JSON document."""
        state_path = (
            tmp_path / "state" / "sessions" / "agent-1" / "state.json"
        )
        original_builtin_open = builtins.open
        original_open = Path.open
        original_replace = os.replace
        opened_for_write = []
        observed = []

        def observe_open(path, mode):
            if "w" in mode and path.parent == state_path.parent:
                published = (
                    json.loads(state_path.read_text())
                    if state_path.exists()
                    else None
                )
                opened_for_write.append((path, published))

        def builtin_open_and_observe(file, *args, **kwargs):
            handle = original_builtin_open(file, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            observe_open(Path(file), mode)
            return handle

        def open_and_observe(path, *args, **kwargs):
            handle = original_open(path, *args, **kwargs)
            mode = args[0] if args else kwargs.get("mode", "r")
            observe_open(path, mode)
            return handle

        def replace_and_observe(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            staged = json.loads(source_path.read_text())
            published = (
                json.loads(state_path.read_text())
                if state_path.exists()
                else None
            )
            observed.append((source_path, destination_path, staged, published))
            original_replace(source, destination)

        monkeypatch.setattr(builtins, "open", builtin_open_and_observe)
        monkeypatch.setattr(Path, "open", open_and_observe)
        monkeypatch.setattr(os, "replace", replace_and_observe)

        def assert_complete_during(action):
            before_open = len(opened_for_write)
            before = len(observed)
            action()
            writes = opened_for_write[before_open:]
            snapshots = observed[before:]
            assert len(writes) == 1, (
                "state writer bypassed single-file staging"
            )
            staged_path, published_at_open = writes[0]
            assert staged_path.parent == state_path.parent
            assert staged_path != state_path
            if published_at_open is not None:
                assert published_at_open["name"] == "agent-1"
            assert len(snapshots) == 1, (
                "state writer bypassed atomic publication"
            )
            source, destination, staged, published = snapshots[0]
            assert source.parent == state_path.parent
            assert source != state_path
            assert destination == state_path
            assert staged["name"] == "agent-1"
            if published is not None:
                assert published["name"] == "agent-1"

            return json.loads(state_path.read_text())

        state = assert_complete_during(
            lambda: tmp_registry.register(
                SessionEntry(name="agent-1", status="starting")
            )
        )
        assert state["status"] == "starting"

        state = assert_complete_during(
            lambda: tmp_registry.update("agent-1", status="running")
        )
        assert state["status"] == "running"

        state = assert_complete_during(
            lambda: tmp_registry.record_cost("agent-1", 0.25)
        )
        assert state["total_cost_usd"] == 0.25

    def test_state_writers_share_state_lock(
            self, tmp_registry, tmp_path, monkeypatch):
        """Every state writer joins one serialization domain."""
        tmp_registry.register(
            SessionEntry(name="agent-1", status="starting", pid=0)
        )
        state_path = (
            tmp_path / "state" / "sessions" / "agent-1" / "state.json"
        )

        first_at_write = threading.Event()
        cost_attempting_lock = threading.Event()
        release_first = threading.Event()
        serialization_lock = threading.Lock()
        lock_holders = set()
        lock_attempts = []
        lock_paths = []
        original_write = SessionRegistry._write_state
        errors = []

        @contextmanager
        def coordinated_lock(path):
            writer = threading.current_thread().name
            lock_attempts.append(writer)
            lock_paths.append(path)
            if writer == "cost-writer":
                cost_attempting_lock.set()
            with serialization_lock:
                lock_holders.add(writer)
                try:
                    yield
                finally:
                    lock_holders.remove(writer)

        def controlled_write(path, data):
            writer = threading.current_thread().name
            assert writer in lock_holders, "state write escaped the shared lock"
            if writer == "status-writer":
                first_at_write.set()
                if not release_first.wait(5):
                    raise AssertionError("timed out releasing first state writer")
            original_write(path, data)

        monkeypatch.setattr(sdk, "_state_file_lock", coordinated_lock)
        monkeypatch.setattr(
            SessionRegistry, "_write_state", staticmethod(controlled_write)
        )

        tmp_registry.register(
            SessionEntry(name="agent-1", status="starting", pid=0)
        )

        def write_status():
            try:
                tmp_registry.update("agent-1", status="running")
            except Exception as exc:
                errors.append(exc)

        def write_cost():
            try:
                tmp_registry.record_cost("agent-1", 0.25)
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(
            target=write_status,
            name="status-writer",
            daemon=True,
        )
        second = threading.Thread(
            target=write_cost,
            name="cost-writer",
            daemon=True,
        )

        first.start()
        try:
            assert first_at_write.wait(5), (
                "status writer did not reach publication"
            )
            second.start()
            assert cost_attempting_lock.wait(5), (
                "record_cost did not attempt the shared state lock"
            )
        finally:
            release_first.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []
        assert lock_attempts == [
            "MainThread",
            "status-writer",
            "cost-writer",
        ]
        assert lock_paths == [state_path, state_path, state_path]

        state = tmp_registry.get("agent-1")
        assert state.status == "running"
        assert state.total_cost_usd == 0.25

    def test_state_lock_is_exclusive_across_processes(
            self, tmp_registry, tmp_path):
        """The serialization lock excludes an independent process."""
        tmp_registry.register(SessionEntry(name="agent-1"))
        state_path = (
            tmp_path / "state" / "sessions" / "agent-1" / "state.json"
        )
        lock_path = state_path.with_suffix(".lock")
        probe = """
import fcntl
import sys
from pathlib import Path

with Path(sys.argv[1]).open("a+") as lock_file:
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(0)
raise SystemExit(1)
"""

        with sdk._state_file_lock(state_path):
            result = subprocess.run(
                [sys.executable, "-c", probe, str(lock_path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        assert result.returncode == 0, result.stderr

    def test_update_nonexistent_is_noop(self, tmp_registry):
        tmp_registry.update("does-not-exist", status="running")

    def test_update_leaves_no_lock_when_the_entry_vanishes_mid_call(
            self, tmp_registry, tmp_path):
        """A no-op update must not leave state.lock in a dying directory.

        `update` re-checks the state file under the lock and returns when the
        entry has already gone, but it took the lock to get there - creating
        `state.lock` inside a session directory that a concurrent teardown was
        removing. `shutil.rmtree` had already scandir'd that directory, so the
        file it never saw made the final `os.rmdir` raise ENOTEMPTY.

        The vanishing is injected in exactly the window the inner re-check
        guards: after `update` saw the file, before it holds the lock.
        """
        tmp_registry.register(SessionEntry(name="agent-1", status="running"))
        session_dir = tmp_path / "state" / "sessions" / "agent-1"
        state_path = session_dir / "state.json"
        real_lock = sdk._state_file_lock

        @contextmanager
        def vanish_then_lock(path):
            path.unlink(missing_ok=True)
            with real_lock(path):
                yield

        with patch.object(sdk, "_state_file_lock", vanish_then_lock):
            tmp_registry.update("agent-1", status="done")

        assert not state_path.exists()
        leftover = sorted(p.name for p in session_dir.iterdir())
        assert leftover == [], f"update left files behind: {leftover}"

    def test_mark_done_keeps_entry(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="agent-1", status="running"))
        tmp_registry.mark_done("agent-1")
        assert tmp_registry.get("agent-1") is not None
        assert tmp_registry.get("agent-1").status == "done"

    def test_list_active(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="a", status="running"))
        tmp_registry.register(SessionEntry(name="b", status="done"))
        tmp_registry.register(SessionEntry(name="c", status="idle"))
        active = tmp_registry.list_active()
        names = {e.name for e in active}
        assert names == {"a", "c"}

    def test_list_all(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="a", status="running"))
        tmp_registry.register(SessionEntry(name="b", status="done"))
        assert len(tmp_registry.list_all()) == 2

    def test_get_by_role(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="mgr", role="manager"))
        tmp_registry.register(SessionEntry(name="agent-1", role="engineer"))
        tmp_registry.register(SessionEntry(name="agent-2", role="engineer"))
        engineers = tmp_registry.get_by_role("engineer")
        assert len(engineers) == 2

    def test_dir_per_session(self, tmp_registry, tmp_path):
        """Each session gets its own directory."""
        tmp_registry.register(SessionEntry(name="agent-42", run_key="42"))
        assert (tmp_path / "state" / "sessions" / "agent-42" / "state.json").exists()

    def test_mark_done_clears_pid(self, tmp_registry):
        tmp_registry.register(SessionEntry(name="agent-1", pid=12345, status="running"))
        tmp_registry.mark_done("agent-1")
        got = tmp_registry.get("agent-1")
        assert got.status == "done"
        assert got.pid == 0

    def test_handoff_path(self, tmp_path, monkeypatch):
        paths.bind_root(tmp_path)
        path = SessionRegistry().handoff_path("agent-42", "setup")
        assert path == tmp_path / "state" / "sessions" / "agent-42" / "handoff-setup.yaml"

    def test_cross_process_visibility(self, tmp_path, monkeypatch):
        """Two registry instances see each other's entries."""
        paths.bind_root(tmp_path)

        r1 = SessionRegistry()
        r1.register(SessionEntry(name="agent-42", run_key="42", status="running"))

        r2 = SessionRegistry()
        got = r2.get("agent-42")
        assert got is not None
        assert got.run_key == "42"

    def test_list_all_reap_dead_marks_crashed(self, tmp_registry):
        """History reads are honest too (#733 vertical 3): with reap_dead an
        active status with a dead pid reads crashed, never running."""
        tmp_registry.register(
            SessionEntry(name="a", status="running", pid=999999999))
        [entry] = tmp_registry.list_all(reap_dead=True)
        assert entry.status == "crashed"
        assert entry.pid == 0
        assert entry.terminal_at > 0
        assert "died" in entry.error
        # durably written, not just the returned view
        assert tmp_registry.get("a").status == "crashed"

    def test_list_all_raw_by_default(self, tmp_registry):
        # The reconciler sweeps list_all() and owns crash-closing there
        # (reconciled_at + emit) - the default read must not preempt it.
        tmp_registry.register(
            SessionEntry(name="a", status="running", pid=999999999))
        assert tmp_registry.list_all()[0].status == "running"
        assert tmp_registry.get("a").status == "running"

    def test_list_active_marks_and_excludes_dead_pid(self, tmp_registry):
        tmp_registry.register(
            SessionEntry(name="a", status="running", pid=999999999))
        assert tmp_registry.list_active() == []
        assert tmp_registry.get("a").status == "crashed"

    def test_live_pid_stays_active(self, tmp_registry):
        tmp_registry.register(
            SessionEntry(name="a", status="running", pid=os.getpid()))
        assert [e.name for e in tmp_registry.list_active()] == ["a"]
        assert tmp_registry.list_all(reap_dead=True)[0].status == "running"

    def test_terminal_entry_never_remarked(self, tmp_registry):
        # A terminal record is settled, whatever its pid field says.
        tmp_registry.register(
            SessionEntry(name="a", status="completed", pid=999999999))
        assert tmp_registry.list_all(reap_dead=True)[0].status == "completed"

    def test_completed_session_stays_for_history(self, tmp_path, monkeypatch):
        paths.bind_root(tmp_path)

        r = SessionRegistry()
        r.register(SessionEntry(name="agent-42", status="running"))
        r.mark_done("agent-42")

        assert r.get("agent-42") is not None
        assert r.get("agent-42").status == "done"
        assert len(r.list_active()) == 0
        assert len(r.list_all()) == 1
