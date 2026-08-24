"""Plain service core for Bobi runtime lifecycle and interaction primitives."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bobi import launch_stamp, paths
from bobi.__version__ import __version__
from bobi.fsutil import atomic_write_text
from bobi.sdk import SessionEntry


log = logging.getLogger(__name__)

RESTART_TIMEOUT = 180.0
MANAGER_PID_TIMEOUT = 30.0


class ServiceError(Exception):
    """Base class for service-core failures."""


class NoAgentInstalled(ServiceError):
    def __init__(self, available: list[tuple[str, str]]) -> None:
        super().__init__("no agent installed")
        self.available = available


class PreflightFailed(ServiceError):
    def __init__(self, validation) -> None:
        super().__init__("preflight failed")
        self.validation = validation


class AlreadyRunning(ServiceError):
    def __init__(self, pid: int) -> None:
        super().__init__(f"already running: {pid}")
        self.pid = pid


class NestedRuntimeError(ServiceError):
    def __init__(self, ancestor: Path, pid: int) -> None:
        super().__init__(f"manager already running at {ancestor}")
        self.ancestor = ancestor
        self.pid = pid


class LaunchTimeout(ServiceError):
    def __init__(self, manager_name: str, timeout: float) -> None:
        super().__init__(
            f"manager session '{manager_name}' did not register within {timeout:g}s"
        )
        self.manager_name = manager_name
        self.timeout = timeout


class TransportReadyTimeout(ServiceError):
    def __init__(self, manager_name: str, timeout: float) -> None:
        super().__init__(
            f"manager session '{manager_name}' transport did not register within {timeout:g}s"
        )
        self.manager_name = manager_name
        self.timeout = timeout


class MessageDeliveryError(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RestartFailed(ServiceError):
    def __init__(self, reason: str, log_file: Path, log_tail: str = "") -> None:
        super().__init__(f"{reason} (see {log_file})")
        self.log_file = Path(log_file)
        self.log_tail = log_tail

    def report(self) -> str:
        return f"{self}\n{self.log_tail}" if self.log_tail else str(self)


@dataclass(frozen=True)
class StartupInfo:
    version: str
    agent_name: str
    project_path: Path
    pid: int
    package: str
    event_server_url: str
    event_server_label: str
    ingress_warning: str
    ingress_hint: str
    workflows: list[str]
    monitors: list[str]
    log_file: Path


@dataclass(frozen=True)
class SpawnResult:
    startup: StartupInfo
    validation: object
    image_rotated: bool = False
    process: subprocess.Popen | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RestartHandle:
    pid: int
    log_file: Path
    process: subprocess.Popen = field(repr=False, compare=False)


@dataclass(frozen=True)
class RestartResult:
    pid: int
    log_file: Path
    output: str = ""


@dataclass(frozen=True)
class LaunchResult:
    entry: SessionEntry
    startup: StartupInfo
    validation: object
    image_rotated: bool = False


@dataclass(frozen=True)
class StopResult:
    pid: int = 0
    stopped: bool = False
    killed: bool = False
    stale: bool = False
    invalid_pid: bool = False
    permission_denied: bool = False
    still_running: bool = False
    event_server_running: bool = False
    event_server_port: int = 0


@dataclass(frozen=True)
class TeamStatus:
    manager_running: bool
    manager_pid: int
    active_agents: list[SessionEntry]


@dataclass(frozen=True)
class MessageResult:
    address: str
    response: str = ""


def manager_session_name(project_path: Path, role: str | None = None) -> str:
    """Return the entry-point session name for a runtime root."""
    if role is None:
        from bobi.config import Config

        try:
            role = Config.load(project_path).entry_role
        except Exception:
            role = "manager"
    return f"bobi-{paths.agent_name_for_root(project_path)}-{role}"


def clear_manager_session(project_path: Path) -> None:
    """Clear persisted manager conversation and event-server bubble state."""
    import shutil

    from bobi.events.state import clear_bubble_state
    from bobi.sdk import save_session_id

    save_session_id(manager_session_name(project_path), "", root=project_path)
    clear_bubble_state(project_path)
    for sub in ("deployments", "cursors"):
        shutil.rmtree(paths.state_path(project_path) / sub, ignore_errors=True)


def build_startup_info(
    project_path: Path,
    pid: int,
    log_file: Path,
    *,
    extra_subscriptions: Iterable[str] = (),
) -> StartupInfo:
    """Collect the startup summary data a CLI or web adapter can format."""
    from bobi.config import Config

    cfg = Config.load(project_path)
    agent_name = paths.agent_name_for_root(project_path)

    if cfg.event_server_url:
        event_server_url = cfg.event_server_url
        event_server_label = (
            "local" if cfg.event_server_url.startswith("http://localhost") else "remote"
        )
    else:
        event_server_url = "localhost:8080"
        event_server_label = "auto"

    ingress_warning = ""
    ingress_hint = ""
    try:
        from bobi.ingress import check_ingress_reachability

        warning = check_ingress_reachability(
            project_path,
            extra_subscriptions=list(extra_subscriptions),
        )
        if warning:
            ingress_warning = warning.detail
            ingress_hint = warning.hint
    except Exception as exc:
        log.debug("Ingress reachability check skipped: %s", exc)

    workflows: list[str] = []
    try:
        import logging as _logging

        _logging.getLogger("bobi.workflow").setLevel(_logging.WARNING)
        from bobi.workflow.triggers import WorkflowDispatcher

        dispatcher = WorkflowDispatcher()
        dispatcher.load_all_workflows(project_path, agent_name=cfg.agent)
        workflows = sorted(set(wf.name for wf, _ in dispatcher.workflows))
    except Exception:
        pass

    monitors: list[str] = []
    try:
        from bobi.monitors.registry import MonitorRegistry

        registry = MonitorRegistry.load(project_path=project_path)
        monitors = sorted(m.name for m in registry.all_monitors())
    except Exception:
        pass

    return StartupInfo(
        version=__version__,
        agent_name=agent_name,
        project_path=project_path,
        pid=pid,
        package=cfg.agent,
        event_server_url=event_server_url,
        event_server_label=event_server_label,
        ingress_warning=ingress_warning,
        ingress_hint=ingress_hint,
        workflows=workflows,
        monitors=monitors,
        log_file=log_file,
    )


def _bind(project_path: Path) -> Path:
    """Bind this process's identity to *project_path* (see paths.bind_root).

    Only the run-in-this-process entry points (`run_team_foreground`,
    `run_manager_from_config`) bind: they ARE the team process. Everything
    else takes the root as an explicit argument so one process (the webapp)
    can service any number of teams without mutating process state (#706).
    """
    from bobi.sdk import set_project_root

    project_path = Path(project_path)
    set_project_root(project_path)
    return project_path


def _load_config_or_raise(project_path: Path):
    from bobi.config import Config

    cfg = Config.load(project_path)
    if cfg.agent:
        return cfg
    raise NoAgentInstalled(_list_agent_packs(project_path))


def _list_agent_packs(project_path: Path) -> list[tuple[str, str]]:
    packs: dict[str, str] = {}
    for agents_dir, label in [
        (paths.agent_cache_dir(), "cached"),
        (paths.agents_root(), "installed"),
        (project_path / "agents", "local"),
    ]:
        if agents_dir.is_dir():
            for d in sorted(agents_dir.iterdir()):
                if (d / "agent.yaml").is_file() or (d / "src" / "agent.yaml").is_file():
                    packs[d.name] = label
    return [(name, source) for name, source in sorted(packs.items())]


def _validate_or_raise(project_path: Path):
    from bobi.validate import validate_config

    validation = validate_config(project_path)
    if getattr(validation, "checks", None) and not validation.ok:
        raise PreflightFailed(validation)
    return validation


def _read_pid(pid_path: Path) -> int:
    from bobi.sdk import read_pid

    return read_pid(pid_path)


def _pid_alive(pid: int) -> bool:
    from bobi.sdk import pid_alive

    return pid_alive(pid)


def _check_nested_runtime(project_path: Path) -> None:
    from bobi.sdk import find_runtime_root

    ancestor = find_runtime_root(project_path.parent)
    if ancestor and ancestor != project_path:
        pid = _read_pid(paths.manager_pid_path(ancestor))
        raise NestedRuntimeError(ancestor, pid)


def _wait_for_manager_entry(
    project_path: Path,
    manager_name: str,
    timeout: float,
) -> SessionEntry:
    from bobi.sdk import SessionRegistry

    deadline = time.monotonic() + timeout
    registry = SessionRegistry(project_path)
    while time.monotonic() < deadline:
        for entry in registry.list_active():
            if entry.name == manager_name:
                return entry
        time.sleep(0.1)
    raise LaunchTimeout(manager_name, timeout)


def _wait_for_manager_transport(
    project_path: Path,
    manager_name: str,
    timeout: float,
) -> None:
    from bobi.events.state import load_bubble_state, load_deployment_state

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bubble = load_bubble_state(project_path)
        deployment = load_deployment_state(project_path, manager_name)
        if (
            bubble.get("bubble_id")
            and bubble.get("bubble_key")
            and deployment.get("deployment_id")
            and deployment.get("api_key")
        ):
            return
        time.sleep(0.1)
    raise TransportReadyTimeout(manager_name, timeout)


def _cli_child_env(project_path: Path) -> dict[str, str]:
    from bobi.env import child_agent_env

    env = child_agent_env(project_path)
    venv_bin = str(Path(sys.executable).parent)
    local_bin = str(Path.home() / ".local" / "bin")
    env["PATH"] = f"{venv_bin}:{local_bin}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_team(
    project_path: Path,
    *,
    fresh: bool = False,
    subscribe: Iterable[str] = (),
) -> SpawnResult:
    """Spawn the manager detached and return without waiting for registration."""
    project_path = Path(project_path)
    cfg = _load_config_or_raise(project_path)
    # Select the team's brain before the in-process preflight below: the MCP
    # probe builds a real brain session in THIS process (#655), which may be
    # the webapp daemon or another host that never ran the CLI agent binding.
    from bobi.brain import set_process_brain_from_config
    set_process_brain_from_config(cfg)
    validation = _validate_or_raise(project_path)

    pid_path = paths.manager_pid_path(project_path)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        pid = _read_pid(pid_path)
        if _pid_alive(pid):
            raise AlreadyRunning(pid)
        pid_path.unlink(missing_ok=True)

    _check_nested_runtime(project_path)

    image_rotated = False
    if fresh:
        clear_manager_session(project_path)
    else:
        from bobi.sdk import check_image_rotation

        image_rotated = check_image_rotation(
            manager_session_name(project_path), project_path
        )

    log_file = paths.manager_log_path(project_path)
    env = _cli_child_env(project_path)
    cmd = [
        sys.executable,
        "-m",
        "bobi.cli",
        "agent",
        paths.agent_name_for_root(project_path),
        "start",
        "--foreground",
    ]
    if fresh:
        cmd.append("--fresh")
    for item in subscribe:
        cmd.extend(["--subscribe", item])

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            cwd=str(project_path),
            env=env,
            start_new_session=True,
        )

    return SpawnResult(
        startup=build_startup_info(
            project_path,
            proc.pid,
            log_file,
            extra_subscriptions=subscribe,
        ),
        validation=validation,
        image_rotated=image_rotated,
        process=proc,
    )


def start_team(
    project_path: Path,
    *,
    fresh: bool = False,
    subscribe: Iterable[str] = (),
    wait_timeout: float = 30,
) -> LaunchResult:
    """Spawn the manager detached, wait for registration, and return its entry."""
    spawned = spawn_team(project_path, fresh=fresh, subscribe=subscribe)
    project_path = Path(project_path)
    cfg = _load_config_or_raise(project_path)
    role = cfg.entry_role
    manager_name = manager_session_name(project_path, role)
    deadline = time.monotonic() + wait_timeout
    try:
        remaining = max(0.0, deadline - time.monotonic())
        entry = _wait_for_manager_entry(
            project_path, manager_name, remaining
        )
        remaining = max(0.0, deadline - time.monotonic())
        _wait_for_manager_transport(project_path, manager_name, remaining)
    except (LaunchTimeout, TransportReadyTimeout):
        proc = spawned.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    return LaunchResult(
        entry=entry,
        startup=spawned.startup,
        validation=spawned.validation,
        image_rotated=spawned.image_rotated,
    )


def launch_team(
    project_path: Path,
    *,
    fresh: bool = False,
    subscribe: Iterable[str] = (),
    wait_timeout: float = 30,
) -> SessionEntry:
    """Spawn the manager detached and return its active registry entry."""
    return start_team(
        project_path,
        fresh=fresh,
        subscribe=subscribe,
        wait_timeout=wait_timeout,
    ).entry


def run_team_foreground(
    project_path: Path,
    *,
    fresh: bool = False,
    subscribe: Iterable[str] = (),
) -> None:
    """Run the selected manager in the current process."""
    project_path = _bind(project_path)
    from bobi.config import load_dotenv
    load_dotenv(project_path)
    cfg = _load_config_or_raise(project_path)
    # Pin before the preflight: the MCP probe builds a brain session here.
    from bobi.brain import set_process_brain_from_config
    set_process_brain_from_config(cfg)
    _validate_or_raise(project_path)
    _check_nested_runtime(project_path)
    if fresh:
        clear_manager_session(project_path)
    else:
        from bobi.sdk import check_image_rotation

        check_image_rotation(manager_session_name(project_path), project_path)
    run_manager_from_config(
        project_path, cfg, extra_subscribe=list(subscribe), foreground=True
    )


def run_manager_from_config(
    project_path: Path,
    cfg,
    extra_subscribe: list[str] | None = None,
    foreground: bool = False,
) -> None:
    """Start an agent from a loaded Config object in the current process."""
    import atexit
    import signal

    from bobi.sdk import set_project_root

    set_project_root(project_path)

    from bobi.brain import set_process_brain_from_config

    set_process_brain_from_config(cfg)

    # Render the team's global instructions (package AGENTS.md) into the paths
    # the active brain auto-loads (#779) - once per boot, before any brain
    # session spawns, and (in a container) after the deploy entrypoint has
    # re-linked the brain config dirs onto the durable volume. Errors propagate
    # for the same reason the codex MCP render's do: a team that ships house
    # rules and can't render them would silently build without the standards.
    from bobi.brain.instructions import render_team_instructions

    render_team_instructions(project_path)

    agent_name = cfg.agent
    role = cfg.entry_role

    from bobi.events.subscriptions import discover_subscriptions

    subscribe = discover_subscriptions(project_path)
    subscribe += [s for s in (extra_subscribe or []) if s not in subscribe]

    from bobi.events.subscriptions import monitor_subscription_keys
    from bobi.monitors.registry import MonitorRegistry

    monitor_events = [
        m.event for m in MonitorRegistry.load(project_path=project_path).effective_monitors()
    ]
    for key in monitor_subscription_keys(monitor_events):
        if key not in subscribe:
            subscribe.append(key)

    from bobi.events.subscriptions import lifecycle_subscription_keys

    for key in lifecycle_subscription_keys():
        if key not in subscribe:
            subscribe.append(key)

    state_dir = paths.state_dir(project_path)

    from bobi.state_version import ensure_state_version

    ensure_state_version(project_path)

    pid_str = str(os.getpid())
    pid_path = paths.manager_pid_path(project_path)
    atomic_write_text(pid_path, pid_str)
    # Record which bobi this manager is running, so an in-place upgrade that
    # replaces the framework underneath it is visible to doctor (#928).
    launch_stamp.record_launch(project_path, launch_stamp.MANAGER, os.getpid())

    def _cleanup():
        try:
            if pid_path.exists() and pid_path.read_text().strip() == pid_str:
                pid_path.unlink(missing_ok=True)
        except OSError:
            pass
        launch_stamp.clear_launch(project_path, launch_stamp.MANAGER,
                                  pid=os.getpid())
        from bobi import http as pooled_http
        from bobi import manager_health

        manager_health.stop()
        pooled_http.close()

    atexit.register(_cleanup)

    def _handle_term(signum, frame):
        log.info("Received SIGTERM - shutting down gracefully")
        _cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handle_term)

    from bobi import manager_health

    health_port = manager_health.start(
        state_dir,
        paths.agent_name_for_root(project_path),
        manager_session=manager_session_name(project_path, role),
    )
    log.info("Manager health endpoint on port %d", health_port)

    if os.environ.get("BOBI_UI"):
        try:
            from bobi.webapp.server import build_app
            from bobi.webui_common.launcher import serve_container

            ui_port = serve_container(
                lambda token: build_app(token=token),
                state_dir=state_dir,
            )
            log.info("Bobi web app on port %d (reach it with `fly proxy`)", ui_port)
        except Exception as e:
            log.warning("Bobi web app failed to start: %s", e)

    log.info(
        "Bobi starting for %s (role=%s)",
        paths.agent_name_for_root(project_path),
        role,
    )

    session_name = manager_session_name(project_path, role)

    # Root the launch chain (#849) BEFORE anything can launch. The manager is
    # a chain root by construction - it boots its own session, and inheriting
    # whatever agent ran `bobi agent <x> restart` would park this daemon at
    # depth N for days. Ordering is load-bearing: the monitor scheduler starts
    # just below and its checks spawn from this process, so a stamp after it
    # would let those launches read as roots of their own.
    from bobi.launch_lineage import LineageLink, stamp_root
    stamp_root(os.environ, LineageLink(
        session=session_name, workflow="", run_key=session_name,
    ))

    has_monitors = paths.monitors_dir(project_path).is_dir() or cfg.monitors
    if has_monitors:
        from bobi.monitors.scheduler import MonitorScheduler

        monitor_scheduler = MonitorScheduler(project_path=project_path)
        monitor_scheduler.start()
        log.info("Monitor scheduler started")

    from bobi.prompts.resolver import build_startup_prompt
    from bobi.subagent import run_persistent_agent

    task = build_startup_prompt(
        role, project_path, agent_name=agent_name, session_name=session_name
    )

    try:
        from bobi.reconcile import reconcile_sessions

        reconciled = reconcile_sessions(exclude_names={session_name})
        if reconciled:
            log.info(
                "Reconciled %d stranded run(s) on startup: %s",
                len(reconciled),
                [r["name"] for r in reconciled],
            )
    except Exception:
        log.debug("Startup reconcile failed", exc_info=True)

    log.info(
        "Bobi launching manager session for %s",
        paths.agent_name_for_root(project_path),
    )
    run_persistent_agent(
        cwd=str(project_path),
        task=task,
        name=session_name,
        role=role,
        mcp_servers=cfg.mcp_servers or None,
        subscribe=subscribe,
    )


def stop_team(project_path: Path, *, force: bool = False) -> StopResult:
    """Stop the selected manager process."""
    import signal

    project_path = Path(project_path)
    pid_path = paths.manager_pid_path(project_path)
    result_kwargs: dict[str, object] = {}

    if not pid_path.exists():
        result_kwargs["pid"] = 0
    else:
        pid = _read_pid(pid_path)
        result_kwargs["pid"] = pid
        if not pid:
            pid_path.unlink(missing_ok=True)
            result_kwargs["invalid_pid"] = True
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pid_path.unlink(missing_ok=True)
                result_kwargs["stale"] = True
            except PermissionError:
                result_kwargs["permission_denied"] = True
            else:
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.kill(pid, sig)
                for _ in range(30):
                    time.sleep(0.2)
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pid_path.unlink(missing_ok=True)
                        result_kwargs["stopped"] = True
                        break
                else:
                    if force:
                        pid_path.unlink(missing_ok=True)
                        result_kwargs["killed"] = True
                    else:
                        result_kwargs["still_running"] = True

    from bobi.kb.embedder import stop as stop_embedder

    stop_embedder(project_path)

    from bobi.events.server import health, resolve_local_port

    es_port = resolve_local_port(project_path)
    result_kwargs["event_server_port"] = es_port
    result_kwargs["event_server_running"] = bool(health(f"http://localhost:{es_port}"))
    return StopResult(**result_kwargs)


def spawn_restart(project_path: Path, *, fresh: bool = False) -> RestartHandle:
    """Detach the stop+start owner before it enters the manager's process tree."""
    project_path = Path(project_path)
    log_file = paths.restart_log_path(project_path)
    cmd = [
        sys.executable,
        "-m",
        "bobi.cli",
        "agent",
        paths.agent_name_for_root(project_path),
        "restart",
        "--detached-worker",
    ]
    if fresh:
        cmd.append("--fresh")

    with open(log_file, "w") as output:
        process = subprocess.Popen(
            cmd,
            stdout=output,
            stderr=output,
            cwd=str(project_path),
            env=_cli_child_env(project_path),
            start_new_session=True,
        )
    return RestartHandle(process.pid, log_file, process)


def restart_team(
    project_path: Path,
    *,
    fresh: bool = False,
    timeout: float = RESTART_TIMEOUT,
) -> RestartResult:
    """Restart through a detached worker that survives its caller."""
    project_path = Path(project_path)
    pid_path = paths.manager_pid_path(project_path)
    previous_pid = _read_pid(pid_path)
    handle = spawn_restart(project_path, fresh=fresh)

    try:
        code = handle.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RestartFailed(
            f"restart is still running as pid {handle.pid} after {timeout:g}s",
            handle.log_file,
            _log_tail(handle.log_file),
        ) from exc

    output = _read_restart_log(handle.log_file)
    if code != 0:
        raise RestartFailed(
            f"restart failed (worker exit {code})",
            handle.log_file,
            _log_tail(handle.log_file),
        )

    pid = _wait_for_manager_pid(
        pid_path,
        timeout=MANAGER_PID_TIMEOUT,
        other_than=previous_pid,
    )
    if not pid:
        if previous_pid and _pid_alive(previous_pid):
            reason = f"restart left manager pid {previous_pid} running"
        else:
            reason = "restart finished but no manager is running"
        raise RestartFailed(reason, handle.log_file, _log_tail(handle.log_file))
    return RestartResult(pid=pid, log_file=handle.log_file, output=output)


def _wait_for_manager_pid(
    pid_path: Path,
    *,
    timeout: float,
    other_than: int = 0,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = _read_pid(pid_path)
        if pid and pid != other_than and _pid_alive(pid):
            return pid
        time.sleep(0.1)
    return 0


def _read_restart_log(log_file: Path) -> str:
    try:
        return log_file.read_text(errors="replace")
    except OSError:
        return ""


def _log_tail(log_file: Path, lines: int = 40) -> str:
    return "\n".join(_read_restart_log(log_file).splitlines()[-lines:])


def team_status(project_path: Path) -> TeamStatus:
    """Return manager and active-agent status without formatting."""
    project_path = Path(project_path)
    pid_path = paths.manager_pid_path(project_path)
    pid = _read_pid(pid_path) if pid_path.exists() else 0
    manager_running = bool(pid and _pid_alive(pid))

    from bobi.sdk import SessionRegistry

    return TeamStatus(
        manager_running=manager_running,
        manager_pid=pid if manager_running else 0,
        active_agents=SessionRegistry(project_path).list_active(),
    )


def list_agents(project_path: Path) -> list[SessionEntry]:
    return team_status(project_path).active_agents


def resolve_address(project_path: Path, to: str | None = None) -> str | None:
    """Resolve a friendly session address to an actual session name."""
    project_path = Path(project_path)

    from bobi.config import Config
    from bobi.sdk import SessionRegistry

    if to is not None and to != "manager":
        return to

    roles = list(dict.fromkeys([Config.load(project_path).entry_role, "manager"]))
    registry = SessionRegistry(project_path)
    for role in roles:
        managers = registry.get_by_role(role)
        active = [m for m in managers if m.status in ("idle", "running", "starting")]
        if active:
            return active[0].name
        if managers:
            return managers[0].name
    return None


def send_message(
    project_path: Path,
    text: str,
    *,
    wait: bool = False,
    session: str | None = None,
    timeout: int = 300,
    sender: str = "cli",
) -> MessageResult:
    """Send a message through the shared inbox transport."""
    project_path = Path(project_path)
    address = resolve_address(project_path, session)
    if not address:
        target = session or "manager"
        raise MessageDeliveryError(f"No active session found for '{target}'.")

    from bobi.inbox import deliver

    ok, response = deliver(address, text, sender=sender, wait=wait,
                           timeout=timeout, root=project_path)
    if not ok:
        raise MessageDeliveryError(response)
    return MessageResult(address=address, response=response)


def ask(
    project_path: Path,
    agent: str,
    text: str,
    *,
    sender: str = "web-ui",
    timeout: int = 300,
) -> MessageResult:
    """Blocking chat with one live session, persisted to its web-UI chat log.

    Only addresses sessions that are actually live — a caller (e.g. a web UI)
    can never fan a message at an arbitrary name. The exchange is appended to
    the session's ``webui-chat.jsonl`` so a chat panel survives refresh."""
    project_path = Path(project_path)
    if agent not in {e.name for e in list_agents(project_path)}:
        raise MessageDeliveryError(f"unknown agent '{agent}'")

    result = send_message(project_path, text, wait=True, session=agent,
                          timeout=timeout, sender=sender)

    from bobi.chat_history import append_chat

    append_chat(project_path, agent, "user", text)
    append_chat(project_path, agent, "agent", result.response)
    return result

