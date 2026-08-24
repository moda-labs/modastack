"""Tests for the unified web app server — security, the dashboard snapshot,
and the per-agent lifecycle endpoints. Most classes monkeypatch service-core
calls and read a real (temp) BOBI_HOME via the bobi_install fixture;
TestMultiAgentRealService runs the real service layer on purpose (#706)."""

import json

import pytest
import yaml
from fastapi.testclient import TestClient

from bobi import service
from bobi.webapp import server

TOKEN = "test-token-123"


def _testclient():
    # The Host guard only allows loopback; TestClient defaults to "testserver".
    app = server.build_app(token=TOKEN)
    return TestClient(app, base_url="http://127.0.0.1")


def _client():
    c = _testclient()
    c.headers.update({"x-bobi-webui-token": TOKEN})
    return c


def _add_design_slot(agents_dir, name, description="An idea, not installed."):
    src = agents_dir / name / "src"
    src.mkdir(parents=True)
    (src / "agent.yaml").write_text(yaml.dump({"agent": name}))
    (src / "agent.md").write_text(f"# {name}\n\n{description}\n")


# --- security ------------------------------------------------------------

class TestSecurity:
    def test_api_requires_token(self, bobi_install):
        c = _testclient()
        assert c.get("/api/dashboard").status_code == 403
        c.headers.update({"x-bobi-webui-token": "wrong"})
        assert c.get("/api/dashboard").status_code == 403
        c.headers.update({"x-bobi-webui-token": TOKEN})
        assert c.get("/api/dashboard").status_code == 200

    def test_host_guard(self, bobi_install):
        app = server.build_app(token=TOKEN)
        c = TestClient(app, base_url="http://evil.example")
        c.headers.update({"x-bobi-webui-token": TOKEN})
        assert c.get("/api/dashboard").status_code == 403

    def test_page_is_open_and_embeds_token(self, bobi_install):
        r = _testclient().get("/")   # no token header
        assert r.status_code == 200
        assert TOKEN in r.text


# --- dashboard -----------------------------------------------------------

class TestDashboard:
    def test_lists_installed_agent(self, bobi_install):
        r = _client().get("/api/dashboard")
        assert r.status_code == 200
        agents = r.json()["agents"]
        names = [a["name"] for a in agents]
        assert bobi_install.agent_name in names
        card = agents[names.index(bobi_install.agent_name)]
        assert card["installed"] is True
        assert card["running"] is False
        assert card["pid"] == 0

    def test_lists_design_only_slot(self, bobi_install):
        _add_design_slot(bobi_install.agents_dir, "ideas")
        agents = _client().get("/api/dashboard").json()["agents"]
        card = next(a for a in agents if a["name"] == "ideas")
        assert card["installed"] is False
        assert card["running"] is False
        assert "not installed" in card["description"]

    def test_running_agent_shows_pid(self, bobi_install):
        import os

        from bobi import paths
        pid_path = paths.manager_pid_path(bobi_install.repo_path)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))   # a live pid: this test process
        agents = _client().get("/api/dashboard").json()["agents"]
        card = next(a for a in agents if a["name"] == bobi_install.agent_name)
        assert card["running"] is True
        assert card["pid"] == os.getpid()

    def test_stale_pid_reads_stopped(self, bobi_install):
        from bobi import paths
        pid_path = paths.manager_pid_path(bobi_install.repo_path)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text("999999999")
        agents = _client().get("/api/dashboard").json()["agents"]
        card = next(a for a in agents if a["name"] == bobi_install.agent_name)
        assert card["running"] is False


# --- per-agent status ------------------------------------------------------

class TestStatus:
    def test_unknown_agent_404(self, bobi_install):
        assert _client().get("/api/agents/nope/status").status_code == 404

    def test_known_agent(self, bobi_install):
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/status")
        assert r.status_code == 200
        assert r.json()["installed"] is True


# --- spend (observability #733) --------------------------------------------

def _seed_session(sessions_dir, name, *, cost, role="engineer",
                  model_usage=None):
    d = sessions_dir / name
    d.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "role": role, "total_cost_usd": cost}
    if model_usage is not None:
        data["model_usage"] = model_usage
    (d / "state.json").write_text(json.dumps(data))


class TestSpend:
    def test_team_spend_folds_sessions(self, bobi_install):
        sd = bobi_install.sessions_dir
        _seed_session(sd, "director", cost=0.60, role="director",
                      model_usage={"anthropic:opus": {"cost_usd": 0.60}})
        _seed_session(sd, "eng", cost=0.40, role="engineer",
                      model_usage={"anthropic:sonnet": {"cost_usd": 0.40}})
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/spend")
        assert r.status_code == 200
        body = r.json()
        assert body["total_cost_usd"] == 1.0
        assert body["sessions_counted"] == 2
        assert body["by_role"] == {"director": 0.6, "engineer": 0.4}
        # by_model ranked highest-first
        assert list(body["by_model"]) == ["anthropic:opus", "anthropic:sonnet"]

    def test_team_spend_empty(self, bobi_install):
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/spend")
        assert r.status_code == 200
        assert r.json()["total_cost_usd"] == 0.0
        assert r.json()["sessions_counted"] == 0

    def test_team_spend_unknown_agent_404(self, bobi_install):
        assert _client().get("/api/agents/nope/spend").status_code == 404

    def test_team_spend_estimates_codex_tokens(self, bobi_install):
        # A codex-brained session records tokens with the cached split but
        # cost 0 (#760): the wire carries a fold-time estimate + raw token
        # volume on the additive fields, and recorded dollars stay 0.
        _seed_session(bobi_install.sessions_dir, "dev", role="dev", cost=0.0,
                      model_usage={"openai:gpt-5.6": {
                          "cost_usd": 0.0, "input_tokens": 1_000_000,
                          "cached_input_tokens": 900_000,
                          "output_tokens": 10_000}})
        body = _client().get(
            f"/api/agents/{bobi_install.agent_name}/spend").json()
        assert body["total_cost_usd"] == 0.0
        assert body["estimated_cost_usd"] == 1.25
        assert body["estimated_by_model"] == {"openai:gpt-5.6": 1.25}
        assert body["tokens_by_model"]["openai:gpt-5.6"] == {
            "input_tokens": 1_000_000, "cached_input_tokens": 900_000,
            "output_tokens": 10_000}

    def test_fleet_spend_totals(self, bobi_install):
        _seed_session(bobi_install.sessions_dir, "director", cost=1.25)
        _seed_session(bobi_install.sessions_dir, "dev", role="dev", cost=0.0,
                      model_usage={"openai:gpt-5.6": {
                          "cost_usd": 0.0, "input_tokens": 1_000_000,
                          "cached_input_tokens": 900_000,
                          "output_tokens": 10_000}})
        r = _client().get("/api/fleet/spend")
        assert r.status_code == 200
        body = r.json()
        assert body["total_cost_usd"] == 1.25
        assert body["estimated_cost_usd"] == 1.25
        assert body["sessions_counted"] == 2
        team = next(t for t in body["teams"]
                    if t["name"] == bobi_install.agent_name)
        assert team["total_cost_usd"] == 1.25
        assert team["estimated_cost_usd"] == 1.25
        assert team["sessions_counted"] == 2

    def test_spend_read_does_not_create_sessions_dir(self, bobi_install):
        import shutil
        # A read endpoint must not mutate disk: remove the sessions dir the
        # fixture pre-creates and confirm a spend GET leaves it absent.
        shutil.rmtree(bobi_install.sessions_dir)
        c = _client()
        assert c.get(f"/api/agents/{bobi_install.agent_name}/spend").status_code == 200
        assert c.get("/api/fleet/spend").status_code == 200
        assert not bobi_install.sessions_dir.exists()

    def test_null_cost_session_does_not_500(self, bobi_install):
        _seed_session(bobi_install.sessions_dir, "broken", cost=None,
                      model_usage={"anthropic:opus": {"cost_usd": None}})
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/spend")
        assert r.status_code == 200
        assert r.json()["total_cost_usd"] == 0.0

    def test_fleet_spend_empty_install(self, bobi_install):
        body = _client().get("/api/fleet/spend").json()
        assert body["total_cost_usd"] == 0.0
        # the installed test agent is listed even with no spend
        names = [t["name"] for t in body["teams"]]
        assert bobi_install.agent_name in names


# --- system health (observability #733) --------------------------------------

def _seed_active_session(sessions_dir, name, *, status, role="engineer"):
    """A registry entry list_active() keeps: an active status and pid 0
    (no liveness check), so the health fold sees it."""
    d = sessions_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(
        {"name": name, "role": role, "status": status, "pid": 0}))


class TestHealth:
    # manager_session_name: bobi-<agent>-<entry_role>; the fixture's
    # agent.yaml declares entry_point "director".
    MGR = "bobi-test-agent-director"

    def _get(self, bobi_install):
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/health")
        assert r.status_code == 200
        return r.json()

    def test_stopped_team(self, bobi_install):
        body = self._get(bobi_install)
        # Local teams share the webapp's host: live by construction, no
        # heartbeats, no supervisor trail.
        assert body["reachability"] == "live"
        assert body["last_heartbeat_at"] is None
        assert body["lifecycle"] == []
        mgr = body["manager"]
        assert mgr["status"] == "stopped"
        assert mgr["running"] is False
        assert mgr["healthy"] is False
        assert mgr["pid"] == 0
        assert mgr["restart_count"] is None
        assert body["sessions"] == []

    def test_running_manager_reports_registry_status(self, bobi_install):
        import os

        from bobi import paths
        pid_path = paths.manager_pid_path(bobi_install.repo_path)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))   # a live pid: this process
        _seed_active_session(bobi_install.sessions_dir, self.MGR,
                             status="idle", role="director")
        _seed_active_session(bobi_install.sessions_dir, "eng",
                             status="running")
        body = self._get(bobi_install)
        mgr = body["manager"]
        assert mgr["running"] is True
        assert mgr["pid"] == os.getpid()
        assert mgr["status"] == "idle"     # the registry's word, not just "up"
        assert mgr["healthy"] is True
        # manager-first ordering, roles and statuses carried through
        sessions = body["sessions"]
        assert sessions[0]["name"] == self.MGR
        assert {"name": "eng", "role": "engineer",
                "status": "running"} in sessions

    def test_boot_window_reads_starting(self, bobi_install):
        import os

        from bobi import paths
        pid_path = paths.manager_pid_path(bobi_install.repo_path)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))
        # pid alive but no registered manager session yet: fail open to
        # "starting" (the same verdict the hosted sidecar reports pre-spawn).
        body = self._get(bobi_install)
        assert body["manager"]["status"] == "starting"
        assert body["manager"]["running"] is True

    def test_stale_pid_reads_stopped(self, bobi_install):
        from bobi import paths
        pid_path = paths.manager_pid_path(bobi_install.repo_path)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text("999999999")
        _seed_active_session(bobi_install.sessions_dir, self.MGR,
                             status="idle", role="director")
        body = self._get(bobi_install)
        assert body["manager"]["status"] == "stopped"
        assert body["manager"]["running"] is False

    def test_terminal_sessions_not_listed(self, bobi_install):
        d = bobi_install.sessions_dir / "done-run"
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps(
            {"name": "done-run", "role": "engineer", "status": "completed"}))
        body = self._get(bobi_install)
        assert body["sessions"] == []

    def test_unknown_agent_404(self, bobi_install):
        assert _client().get("/api/agents/nope/health").status_code == 404


# --- session logs (observability #733 vertical 3) ---------------------------

def _seed_history_session(sessions_dir, name, *, status, role="engineer",
                          error="", terminal_at=0.0, last_activity=0.0,
                          pid=0, cost=0.0):
    d = sessions_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "name": name, "role": role, "status": status, "pid": pid,
        "error": error, "terminal_at": terminal_at,
        "last_activity": last_activity, "total_cost_usd": cost,
        "session_id": f"sid-{name}",
    }))


class TestSessionLog:
    MGR = "bobi-test-agent-director"

    def _get(self, bobi_install):
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/sessions")
        assert r.status_code == 200
        return r.json()

    def test_empty_history(self, bobi_install):
        assert self._get(bobi_install) == {
            "sessions": [],
            "counts": {"active": 0, "completed": 0, "failed": 0, "crashed": 0},
            "truncated": False,
        }

    def test_outcomes_listed_newest_first(self, bobi_install):
        import os

        sd = bobi_install.sessions_dir
        _seed_history_session(sd, "old-done", status="completed",
                              terminal_at=100.0, last_activity=100.0)
        _seed_history_session(sd, "boom", status="failed",
                              error="turn errored", terminal_at=200.0,
                              last_activity=200.0, cost=0.25)
        _seed_history_session(sd, "live", status="running", pid=os.getpid(),
                              last_activity=300.0)
        body = self._get(bobi_install)
        assert [s["name"] for s in body["sessions"]] == \
            ["live", "boom", "old-done"]
        boom = body["sessions"][1]
        assert boom["status"] == "failed"
        assert boom["error"] == "turn errored"
        assert boom["terminal_at"] == 200.0
        assert boom["session_id"] == "sid-boom"
        assert boom["total_cost_usd"] == 0.25
        assert boom["ended"] is True
        live = body["sessions"][0]
        assert live["error"] == ""
        assert live["terminal_at"] is None   # null, never omitted
        assert live["ended"] is False
        assert body["counts"] == {"active": 1, "completed": 1,
                                  "failed": 1, "crashed": 0}
        assert body["truncated"] is False

    def test_dead_pid_reads_crashed_not_running(self, bobi_install):
        _seed_history_session(bobi_install.sessions_dir, "zombie",
                              status="running", pid=999999999,
                              last_activity=100.0)
        body = self._get(bobi_install)
        [z] = body["sessions"]
        assert z["status"] == "crashed"
        assert z["error"]                     # the honest-status message
        assert z["terminal_at"] is not None
        assert z["ended"] is True
        assert body["counts"] == {"active": 0, "completed": 0,
                                  "failed": 0, "crashed": 1}

    def test_manager_flagged(self, bobi_install):
        _seed_history_session(bobi_install.sessions_dir, self.MGR,
                              status="completed", role="director",
                              terminal_at=1.0, last_activity=1.0)
        [row] = self._get(bobi_install)["sessions"]
        assert row["is_manager"] is True

    def test_legacy_done_counts_completed(self, bobi_install):
        _seed_history_session(bobi_install.sessions_dir, "old",
                              status="done", last_activity=1.0)
        assert self._get(bobi_install)["counts"]["completed"] == 1

    def test_error_status_counts_failed(self, bobi_install):
        # "error" is still written for turn-level failures (rotation death,
        # monitor timeouts) - the log counts it as a failure.
        _seed_history_session(bobi_install.sessions_dir, "curator",
                              status="error", last_activity=1.0)
        assert self._get(bobi_install)["counts"]["failed"] == 1

    def test_unknown_agent_404(self, bobi_install):
        assert _client().get("/api/agents/nope/sessions").status_code == 404


# --- lifecycle actions -----------------------------------------------------

class _FakeStartup:
    pid = 4242


class _FakeSpawn:
    startup = _FakeStartup()


# --- hosted onboarding (the setup app mounted under /setup) ----------------

class TestClaudeAvailable:
    def test_absolute_resolution_counts(self, tmp_path, monkeypatch):
        cli = tmp_path / "claude"
        cli.write_text("#!/bin/sh\n")
        monkeypatch.setattr("bobi.brain.claude.shutil.which",
                            lambda name: str(cli))
        assert server._claude_available() is True

    def test_cwd_relative_fallback_is_not_availability(self, tmp_path,
                                                       monkeypatch):
        # On Linux with no CLI on PATH the resolver falls back to the bare
        # name "claude" (correct for exec). A stray file of that name in the
        # server's CWD must not read as an installed CLI.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "claude").mkdir()
        monkeypatch.setattr("bobi.brain.claude.shutil.which", lambda name: None)
        monkeypatch.setattr("bobi.brain.claude.platform.system", lambda: "Linux")
        assert server._claude_available() is False


class TestSetupHosting:
    def _open(self, client, monkeypatch, name="new-team"):
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        return client.post("/api/setup/open", json={"name": name})

    def test_no_session_redirects_to_shell(self, bobi_install):
        c = _client()
        r = c.get("/setup/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/#/setup"

    def test_open_creates_session(self, bobi_install, monkeypatch):
        c = _client()
        r = self._open(c, monkeypatch)
        assert r.status_code == 200
        assert r.json() == {"url": "/setup/new-team/", "name": "new-team",
                            "resumed": False}
        cur = c.get("/api/setup/current").json()
        assert cur == {"active": True, "name": "new-team",
                       "sessions": ["new-team"]}
        # setup state persisted under the slot's run root
        from bobi.setup.state import SetupState
        from bobi import paths
        state = SetupState.load(paths.agent_run_root("new-team"))
        assert state is not None
        assert state.team_name == "new-team"

    def test_open_resumes_unfinished_session(self, bobi_install, monkeypatch):
        c = _client()
        assert self._open(c, monkeypatch).json()["resumed"] is False
        assert self._open(c, monkeypatch).json()["resumed"] is True

    def test_open_passes_model_to_hosted_setup_app(self, bobi_install,
                                                   monkeypatch):
        from fastapi import FastAPI

        seen = {}

        def fake_build_app(*args, **kwargs):
            seen["model"] = kwargs.get("model")
            return FastAPI()

        monkeypatch.setattr(server, "_claude_available", lambda: True)
        monkeypatch.setattr("bobi.setup.webui.server.build_app", fake_build_app)

        c = _client()
        r = c.post("/api/setup/open",
                   json={"name": "new-team", "model": "sonnet"})

        assert r.status_code == 200
        assert seen["model"] == "sonnet"

    def test_hosted_page_serves_with_base_and_token(self, bobi_install,
                                                    monkeypatch):
        c = _client()
        self._open(c, monkeypatch)
        r = c.get("/setup/new-team/")
        assert r.status_code == 200
        assert TOKEN in r.text
        assert '"/setup/new-team/static/app.js"' in r.text

    def test_hosted_setup_supports_concurrent_sessions(
            self, bobi_install, monkeypatch):
        c = _client()
        assert self._open(c, monkeypatch, name="alpha").json()["url"] \
            == "/setup/alpha/"
        assert self._open(c, monkeypatch, name="beta").json()["url"] \
            == "/setup/beta/"
        cur = c.get("/api/setup/current").json()
        assert cur == {"active": True, "name": "alpha",
                       "sessions": ["alpha", "beta"]}
        assert c.get("/setup/alpha/api/state").json()["team_name"] == "alpha"
        assert c.get("/setup/beta/api/state").json()["team_name"] == "beta"

    def test_hosted_api_requires_token(self, bobi_install, monkeypatch):
        # The mounted sub-app must still enforce its /api token check even
        # though its url.path carries the /setup prefix (the root_path fix
        # in webui_common.security). Both clients share one app so they see
        # the same active onboarding session.
        app = server.build_app(token=TOKEN)
        c = TestClient(app, base_url="http://127.0.0.1")
        c.headers.update({"x-bobi-webui-token": TOKEN})
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        assert c.post("/api/setup/open",
                      json={"name": "new-team"}).status_code == 200
        bare = TestClient(app, base_url="http://127.0.0.1")
        assert bare.get("/setup/new-team/api/state").status_code == 403
        assert c.get("/setup/new-team/api/state").status_code == 200

    def test_hosted_finish_returns_home_without_launching(self, bobi_install,
                                                          monkeypatch):
        # Finish returns to the dashboard; launching stays a deliberate
        # action on the agent's card/dashboard (never automatic).
        def fail_start(root, **kw):
            raise AssertionError("finish must not launch")

        monkeypatch.setattr(service, "start_team", fail_start)
        monkeypatch.setattr(service, "spawn_team", fail_start)
        c = _client()
        self._open(c, monkeypatch)
        body = c.post("/setup/new-team/api/finish").json()
        assert body["finished"] is True
        assert body["redirect"] == "/#/"
        # The onboarding slot is released: current reports inactive and
        # /setup/ redirects back to the shell.
        assert c.get("/api/setup/current").json()["active"] is False
        assert c.get("/setup/",
                     follow_redirects=False).status_code == 307

    def test_open_mode_deep_links_the_editor(self, bobi_install, monkeypatch):
        import yaml as _yaml

        from bobi import paths
        src = paths.agent_source_dir(bobi_install.agent_name)
        src.mkdir(parents=True, exist_ok=True)
        (src / "agent.yaml").write_text(_yaml.dump(
            {"agent": bobi_install.agent_name}))
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        r = _client().post("/api/setup/open",
                           json={"name": bobi_install.agent_name,
                                 "mode": "open"})
        assert r.status_code == 200
        assert r.json()["url"].startswith(
            f"/setup/{bobi_install.agent_name}/?open=")
        assert str(src) in r.json()["url"]

    def test_open_mode_requires_source(self, bobi_install, monkeypatch):
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        r = _client().post("/api/setup/open",
                           json={"name": "no-src-slot", "mode": "open"})
        assert r.status_code == 404

    def test_open_mode_resolves_nested_source(self, bobi_install, monkeypatch):
        # An older flow could land a template in a src/ SUBFOLDER
        # (src/eng-team/); a single team child still resolves as the source.
        import yaml as _yaml

        from bobi import paths
        nested = paths.agent_source_dir("legacy-slot") / "eng-team"
        nested.mkdir(parents=True)
        (nested / "agent.yaml").write_text(_yaml.dump({"agent": "eng-team"}))
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        r = _client().post("/api/setup/open",
                           json={"name": "legacy-slot", "mode": "open"})
        assert r.status_code == 200
        assert str(nested) in r.json()["url"]

    def test_finish_renames_slot_to_team_name(self, bobi_install, monkeypatch):
        # The slot opens under a placeholder name; the team gets its real
        # name during setup (template pick / auto-name). Finish moves the
        # whole slot dir to match (#526: a slot IS its team).
        from bobi import paths

        c = _client()
        self._open(c, monkeypatch)   # slot "new-team"
        # Name the team through the real flow (mutates the parked state).
        r = c.post("/setup/new-team/api/rename", json={"name": "eng-team"})
        assert r.status_code == 200
        body = c.post("/setup/new-team/api/finish").json()
        assert body["redirect"] == "/#/"
        assert not paths.agent_dir("new-team").exists()
        assert paths.agent_run_root("eng-team").is_dir()

    def test_create_rename_moves_placeholder_source_and_run_slot(
            self, bobi_install, monkeypatch):
        # The local web app starts a from-scratch team in a placeholder slot
        # (usually new-agent). Renaming during setup should move the editable
        # source out of that slot immediately, and Finish should move the
        # run/ state beside it so the final folder is the chosen name.
        from bobi import paths

        c = _client()
        self._open(c, monkeypatch, name="new-agent")
        old_src = paths.agent_source_dir("new-agent")
        r = c.post("/setup/new-agent/api/start",
                   json={"mode": "create", "location": str(old_src)})
        assert r.status_code == 200
        old_src.mkdir(parents=True, exist_ok=True)
        (old_src / "agent.yaml").write_text("agent: new-agent\n")
        r = c.post("/setup/new-agent/api/rename",
                   json={"name": "Field Ops"})
        assert r.status_code == 200
        assert not old_src.exists()
        assert paths.agent_source_dir("field-ops").is_dir()

        body = c.post("/setup/new-agent/api/finish").json()
        assert body["redirect"] == "/#/"
        assert not paths.agent_dir("new-agent").exists()
        assert paths.agent_source_dir("field-ops").is_dir()
        assert paths.agent_run_root("field-ops").is_dir()

    def test_finish_does_not_merge_custom_source_into_existing_slot(
            self, bobi_install, monkeypatch, tmp_path):
        from bobi import paths

        c = _client()
        self._open(c, monkeypatch, name="new-agent")
        custom_src = tmp_path / "field-ops-src"
        existing_src = paths.agent_source_dir("field-ops")
        existing_src.mkdir(parents=True)
        (existing_src / "agent.yaml").write_text("agent: field-ops\n")
        r = c.post("/setup/new-agent/api/start",
                   json={"mode": "create", "location": str(custom_src)})
        assert r.status_code == 200
        r = c.post("/setup/new-agent/api/rename",
                   json={"name": "Field Ops"})
        assert r.status_code == 200

        body = c.post("/setup/new-agent/api/finish").json()
        assert body["redirect"] == "/#/"
        assert paths.agent_run_root("new-agent").is_dir()
        assert not paths.agent_run_root("field-ops").exists()

    def test_open_requires_claude(self, bobi_install, monkeypatch):
        monkeypatch.setattr(server, "_claude_available", lambda: False)
        r = _client().post("/api/setup/open", json={"name": "x"})
        assert r.status_code == 409

    def test_open_rejects_bad_name(self, bobi_install, monkeypatch):
        monkeypatch.setattr(server, "_claude_available", lambda: True)
        r = _client().post("/api/setup/open", json={"name": "../evil"})
        assert r.status_code == 400


# --- subagents + chat ------------------------------------------------------

def _entry(name, role="engineer", **kw):
    from bobi.sdk import SessionEntry
    return SessionEntry(name=name, role=role, **kw)


class TestSubagents:
    def test_unknown_agent_404(self, bobi_install):
        assert _client().get("/api/agents/nope/subagents").status_code == 404

    def test_roster_with_manager_badge(self, bobi_install, monkeypatch):
        mgr = f"bobi-{bobi_install.agent_name}-director"
        monkeypatch.setattr(
            service, "list_agents",
            lambda root: [_entry("bobi-worker-1"), _entry(mgr, role="director")])
        r = _client().get(f"/api/agents/{bobi_install.agent_name}/subagents")
        assert r.status_code == 200
        subs = r.json()["subagents"]
        # The manager orders first and carries the badge.
        assert subs[0]["name"] == mgr
        assert subs[0]["is_manager"] is True
        assert subs[1]["is_manager"] is False

    def test_messages_from_chat_log(self, bobi_install):
        from bobi.chat_history import append_chat
        append_chat(bobi_install.repo_path, "bobi-worker-1", "user", "hi")
        append_chat(bobi_install.repo_path, "bobi-worker-1", "agent", "hello!")
        r = _client().get(
            f"/api/agents/{bobi_install.agent_name}"
            "/subagents/bobi-worker-1/messages")
        assert r.status_code == 200
        assert r.json()["messages"] == [
            {"role": "user", "text": "hi"},
            {"role": "agent", "text": "hello!"},
        ]

    def test_messages_bad_session_name_404(self, bobi_install):
        r = _client().get(
            f"/api/agents/{bobi_install.agent_name}"
            "/subagents/..%2Fetc/messages")
        assert r.status_code == 404

    def test_messages_from_codex_rollout(self, bobi_install, monkeypatch,
                                         tmp_path):
        """A codex-brained session renders its rollout, not a blank panel —
        the fleet-UI regression this fixes. The runtime dispatches on the
        recorded ``.brain`` to the Codex reader instead of the Claude one."""
        import json

        from bobi import paths

        sessions = paths.sessions_dir(bobi_install.repo_path)
        (sessions / "bobi-worker-1.id").write_text("codex-thread-42")
        (sessions / "bobi-worker-1.brain").write_text("codex")

        codex_home = tmp_path / "codex"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))
        rollout = (codex_home / "sessions" / "2026" / "07" / "09"
                   / "rollout-2026-07-09T05-12-25-codex-thread-42.jsonl")
        rollout.parent.mkdir(parents=True)
        rollout.write_text("\n".join(json.dumps(r) for r in [
            {"type": "response_item", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "are you alive?"}]}},
            {"type": "response_item", "payload": {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Yes, standing by."}]}},
        ]) + "\n")

        r = _client().get(
            f"/api/agents/{bobi_install.agent_name}"
            "/subagents/bobi-worker-1/messages")
        assert r.status_code == 200
        assert r.json()["messages"] == [
            {"role": "user", "text": "are you alive?"},
            {"role": "agent", "text": "Yes, standing by."},
        ]


class TestChat:
    """Chat is submit-then-poll: POST returns a message id right away; the
    deliver runs in a background thread and its outcome lands on the job
    status endpoint (the reply itself reaches the transcript)."""

    def _await_job(self, client, agent, message_id, tries=100):
        import time
        for _ in range(tries):
            r = client.get(f"/api/agents/{agent}/chat/{message_id}")
            assert r.status_code == 200
            job = r.json()
            if job["status"] != "pending":
                return job
            time.sleep(0.02)
        raise AssertionError("chat job never resolved")

    def test_chat_submits_and_resolves(self, bobi_install, monkeypatch):
        seen = {}

        def fake_ask(root, agent, text, **kw):
            seen.update(root=root, agent=agent, text=text)
            return service.MessageResult(address=agent, response="done!")

        monkeypatch.setattr(service, "ask", fake_ask)
        c = _client()
        r = c.post(
            f"/api/agents/{bobi_install.agent_name}/chat",
            json={"subagent": "bobi-worker-1", "text": "go"})
        assert r.status_code == 200
        mid = r.json()["message_id"]
        assert mid
        job = self._await_job(c, bobi_install.agent_name, mid)
        assert job == {"status": "done"}
        assert seen["root"] == bobi_install.repo_path
        assert seen["agent"] == "bobi-worker-1"

    def test_an_empty_subagent_means_the_manager_on_this_runtime_too(
            self, bobi_install, monkeypatch):
        """#987 - one route, two runtimes, one meaning.

        The route documents `subagent` as optional (only `text` is required),
        and the hosted runtime has always read an empty one as "the team
        manager" (`supervisor/admin.py`). `LocalRuntime` passed the empty
        string straight to `service.ask`, which rejected it on its membership
        guard as `unknown agent ''`, so the same request behaved differently
        on `bobi app` than on the fleet.
        """
        seen = {}

        def fake_ask(root, agent, text, **kw):
            seen.update(root=root, agent=agent, text=text)
            return service.MessageResult(address=agent, response="ack")

        monkeypatch.setattr(service, "ask", fake_ask)
        c = _client()
        r = c.post(f"/api/agents/{bobi_install.agent_name}/chat",
                   json={"subagent": "", "text": "continue this"})
        assert r.status_code == 200
        job = self._await_job(c, bobi_install.agent_name, r.json()["message_id"])
        assert job == {"status": "done"}
        # The same name the supervisor resolves, not a hand-built string.
        assert seen["agent"] == service.manager_session_name(
            bobi_install.repo_path)

    def test_an_absent_subagent_key_resolves_the_same_way(self, bobi_install,
                                                          monkeypatch):
        seen = {}
        monkeypatch.setattr(service, "ask", lambda root, agent, text, **kw: (
            seen.update(agent=agent),
            service.MessageResult(address=agent, response="ack"))[1])
        c = _client()
        r = c.post(f"/api/agents/{bobi_install.agent_name}/chat",
                   json={"text": "continue this"})
        assert r.status_code == 200
        assert self._await_job(c, bobi_install.agent_name,
                               r.json()["message_id"]) == {"status": "done"}
        assert seen["agent"] == service.manager_session_name(
            bobi_install.repo_path)

    def test_chat_empty_message_400(self, bobi_install):
        r = _client().post(
            f"/api/agents/{bobi_install.agent_name}/chat",
            json={"subagent": "x", "text": "  "})
        assert r.status_code == 400

    def test_chat_delivery_failure_lands_on_job(self, bobi_install,
                                                monkeypatch):
        def fake_ask(root, agent, text, **kw):
            raise service.MessageDeliveryError("session 'x' process is dead")

        monkeypatch.setattr(service, "ask", fake_ask)
        c = _client()
        r = c.post(
            f"/api/agents/{bobi_install.agent_name}/chat",
            json={"subagent": "x", "text": "hi"})
        assert r.status_code == 200
        job = self._await_job(c, bobi_install.agent_name, r.json()["message_id"])
        assert job["status"] == "error"
        assert "dead" in job["error"]

    def test_chat_unknown_agent_404(self, bobi_install):
        r = _client().post("/api/agents/nope/chat",
                           json={"subagent": "x", "text": "hi"})
        assert r.status_code == 404

    def test_chat_status_unknown_message_404(self, bobi_install):
        r = _client().get(
            f"/api/agents/{bobi_install.agent_name}/chat/deadbeef")
        assert r.status_code == 404


class TestLifecycle:
    def test_start_spawns(self, bobi_install, monkeypatch):
        seen = {}

        def fake_spawn(root, **kw):
            seen["root"] = root
            return _FakeSpawn()

        monkeypatch.setattr(service, "spawn_team", fake_spawn)
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/start")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "pid": 4242}
        assert seen["root"] == bobi_install.repo_path

    def test_start_unknown_404(self, bobi_install):
        assert _client().post("/api/agents/nope/start").status_code == 404

    def test_start_already_running(self, bobi_install, monkeypatch):
        def fake_spawn(root, **kw):
            raise service.AlreadyRunning(77)

        monkeypatch.setattr(service, "spawn_team", fake_spawn)
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/start")
        assert r.status_code == 409
        assert r.json()["pid"] == 77

    def test_start_preflight_failed(self, bobi_install, monkeypatch):
        class FakeValidation:
            def format(self):
                return "missing SLACK_BOT_TOKEN"

        def fake_spawn(root, **kw):
            raise service.PreflightFailed(FakeValidation())

        monkeypatch.setattr(service, "spawn_team", fake_spawn)
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/start")
        assert r.status_code == 409
        assert "SLACK_BOT_TOKEN" in r.json()["report"]

    def test_stop(self, bobi_install, monkeypatch):
        monkeypatch.setattr(
            service, "stop_team",
            lambda root, **kw: service.StopResult(pid=42, stopped=True))
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/stop")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["stopped"] is True

    def test_stop_not_running_is_ok(self, bobi_install, monkeypatch):
        monkeypatch.setattr(
            service, "stop_team", lambda root, **kw: service.StopResult(pid=0))
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/stop")
        assert r.json()["ok"] is True

    def test_restart(self, bobi_install, monkeypatch):
        calls = []
        monkeypatch.setattr(
            service, "stop_team",
            lambda root, **kw: calls.append("stop")
            or service.StopResult(pid=42, stopped=True))
        monkeypatch.setattr(
            service, "spawn_team",
            lambda root, **kw: calls.append("spawn") or _FakeSpawn())
        r = _client().post(f"/api/agents/{bobi_install.agent_name}/restart")
        assert r.status_code == 200
        assert calls == ["stop", "spawn"]


class TestMultiAgentRealService:
    """Regression tests for #706: one webapp process serves service-backed
    endpoints for MULTIPLE agents.

    No service monkeypatching here on purpose. The bug was the process-global
    runtime bind (`service._bind` -> `paths.bind_root`, which refuses to rebind
    to a second root), and only real service calls exercise it - the mocked
    tests above never see the bind."""

    @pytest.fixture
    def two_agents(self, tmp_path, monkeypatch):
        from bobi import paths

        home = tmp_path / "home"
        monkeypatch.setenv("BOBI_HOME", str(home))
        # The webapp daemon process never binds a runtime root; start unbound.
        paths.bind_root(None)
        names = ["alpha-team", "beta-team"]
        for name in names:
            pkg = home / "agents" / name / "run" / "package"
            pkg.mkdir(parents=True)
            (pkg / "agent.yaml").write_text(yaml.dump({
                "version": "0.0.1",
                "agent": name,
                "entry_point": "director",
            }))
        yield names
        paths.bind_root(None)

    def test_roster_serves_both_agents(self, two_agents):
        c = _client()
        for name in two_agents:
            r = c.get(f"/api/agents/{name}/subagents")
            assert r.status_code == 200, f"{name}: {r.text}"
            assert r.json()["subagents"] == []

    def test_stop_serves_both_agents(self, two_agents):
        c = _client()
        for name in two_agents:
            r = c.post(f"/api/agents/{name}/stop")
            assert r.status_code == 200, f"{name}: {r.text}"
            assert r.json()["ok"] is True

    def test_webapp_process_stays_unbound(self, two_agents):
        from bobi import paths

        c = _client()
        for name in two_agents:
            assert c.get(f"/api/agents/{name}/subagents").status_code == 200
        assert paths.bound_root() is None
