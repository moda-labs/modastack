"""CLI interface for bobi."""

import json
import logging
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import truststore
truststore.inject_into_ssl()

import click

from bobi import logs, paths
from bobi.install import (
    install_pack as _install_pack,
    resolve_agent_pack as _resolve_agent_pack,
    write_install_gitignore as _write_install_gitignore,
)

from .__version__ import __version__
# Module level, not lazy: `workflows resume --verdict` builds its click.Choice
# out of this at import time, so the CLI and the workflow engine cannot drift
# on what a gate may be answered with.
from .workflow.schema import GATE_VERDICTS

_PACKAGE_DIR = Path(__file__).parent

# Prompt hints for framework-level env vars an agent.yaml may reference.
# These are not credentials in the secret sense, so tell the user what a
# blank answer means instead of implying a value is required.
_ENV_VAR_HINTS = {
    "BOBI_EVENT_SERVER":
        "event server URL - leave blank to auto-start the local server",
}


def _interactive_terminal() -> bool:
    """True when both ends of the session are a real terminal.

    Split out so tests can stub interactivity (the test runner replaces
    stdin/stdout with pipes).
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _print_startup_info(project_path: Path, pid: int, log_file: Path):
    """Print a startup summary with environment info."""
    from bobi.service import build_startup_info

    info = build_startup_info(project_path, pid, log_file)

    W = 16  # column width for labels
    lines = []
    lines.append(f"bobi v{info.version}")
    lines.append(f"  {'slot':<{W}}{info.agent_name} ({info.project_path})")
    lines.append(f"  {'pid':<{W}}{info.pid}")
    if info.package:
        lines.append(f"  {'package':<{W}}{info.package}")
    lines.append(
        f"  {'event server':<{W}}{info.event_server_url} ({info.event_server_label})"
    )
    if info.ingress_warning:
        lines.append(f"  {'ingress':<{W}}WARNING: {info.ingress_warning}")
        if info.ingress_hint:
            lines.append(f"  {'':<{W}}{info.ingress_hint}")
    if info.workflows:
        lines.append(f"  {'workflows':<{W}}{', '.join(info.workflows)}")
    if info.monitors:
        lines.append(f"  {'monitors':<{W}}{', '.join(info.monitors)}")
    lines.append(f"  {'logs':<{W}}{info.log_file}")

    click.echo("\n".join(lines))


def _detect_project_root(cwd: Path | None = None) -> Path:
    """Resolve and bind an already-selected runtime root.

    This only honors inherited ``BOBI_ROOT`` or an explicit runtime root. It
    requires an explicit runtime root; interactive runtime commands should be
    invoked through the named-agent CLI so the agent group can bind identity once.
    """
    bound = paths.bound_root()
    if bound is not None:
        return bound
    try:
        root = paths.resolve_root(cwd)
    except RuntimeError as e:
        raise click.UsageError(str(e))
    paths.bind_root(root)
    return root


def _project_state_dir(project_path: Path) -> Path:
    """Runtime state directory for a project's manager."""
    return paths.state_dir(project_path)


def _try_detect_project_root() -> Path | None:
    """Best-effort runtime binding from inherited BOBI_ROOT only."""
    try:
        return _detect_project_root()
    except click.UsageError:
        return None


def _bind_agent_runtime(name: str) -> Path:
    try:
        root = paths.resolve_root_for_agent(name)
    except RuntimeError as e:
        raise click.UsageError(str(e))
    paths.bind_root(root)
    _attach_runtime_log(root)
    _pin_team_brain(root)
    return root


def _pin_team_brain(root: Path) -> None:
    """Select the team's brain for the CLI process itself (#655).

    Sessions this process runs directly (`--as-check` monitor checks, ad-hoc
    wait runs, verdict agents) must use the team's configured brain, not the
    framework default - a gateway team's check would otherwise hit real
    Anthropic with the gateway's token. Detached children don't rely on this:
    their env is rewritten by ``child_agent_env()``. Direct CLI sessions still
    need runtime .env credentials (for example gateway auth) in this process, so
    this CLI runtime-binding path loads them explicitly instead of hiding that side
    effect inside ``Config.load()``.
    """
    from bobi.brain import set_process_brain_from_config
    from bobi.config import Config, load_dotenv
    try:
        load_dotenv(root)
        cfg = Config.load(root)
    except Exception as e:  # noqa: BLE001 — `stop`/`status` must still work
        logging.getLogger(__name__).warning(
            "Could not load team config from %s to select its brain (%s); "
            "sessions run by this command use the framework default.", root, e)
        return
    set_process_brain_from_config(cfg)


def _attach_runtime_log(root: Path) -> None:
    """Also send this process's logs to the runtime's manager.log.

    Stands down when the root logger already reaches that file - either
    because an earlier call attached this same handler, or because Bobi
    spawned this process with its stderr redirected into manager.log, which
    is how the manager and every monitor check are launched. A second writer
    would put each record on disk twice, and a duplicated line inflates the
    counts an operator reads back out of an incident (#851).
    """
    log_path = paths.manager_log_path(root)
    if logs.root_writes_to(log_path):
        return
    logging.getLogger().addHandler(logs.file_handler(log_path))




class _PluginGroup(click.Group):
    """A click Group that also serves plugin commands, lazily.

    Plugins register under the `bobi.commands` entry-point group — the seam
    separately-installed packages deliver commands through (e.g. the private
    deploy plugin adds `bobi deploy` / `deploy-init` / `destroy`). Without such a package
    installed, the CLI is the local product only.

    Lazy on purpose: the entry-point scan reads metadata for every installed
    distribution and ep.load() imports the plugin, so neither may run at
    import time — `import bobi.cli` happens in hot paths that never dispatch
    a plugin command (monitor-scheduler subprocesses, the webapp daemon).
    The scan runs only when a command name misses the built-ins or the
    command list is actually rendered (--help, completion); a plugin loads
    only when ITS command runs. A plugin can never shadow a built-in, and a
    broken plugin must not take the CLI down with it (warn and move on).
    """

    _plugin_eps: dict | None = None

    def _plugins(self) -> dict:
        if self._plugin_eps is None:
            from importlib.metadata import entry_points
            eps = {}
            for ep in entry_points(group="bobi.commands"):
                if ep.name in self.commands:  # built-ins win
                    logging.getLogger(__name__).warning(
                        "CLI plugin '%s' (%s) collides with a built-in "
                        "command — ignored", ep.name, ep.value)
                    continue
                eps[ep.name] = ep
            self._plugin_eps = eps
        return self._plugin_eps

    def list_commands(self, ctx):
        return sorted(set(super().list_commands(ctx)) | set(self._plugins()))

    def get_command(self, ctx, name):
        cmd = super().get_command(ctx, name)
        if cmd is not None:
            return cmd
        ep = self._plugins().get(name)
        if ep is None:
            return None
        try:
            return ep.load()
        except Exception:
            logging.getLogger(__name__).warning(
                "failed to load CLI plugin '%s' (%s)", name, ep.value,
                exc_info=True)
            return None


@click.group(cls=_PluginGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="bobi")
@click.pass_context
def main(ctx):
    """Bobi — build teams of event-driven AI agents."""
    logs.configure_root()
    # httpx logs every request at INFO, which the root level above would put
    # in front of the user's actual output — and `bobi app start` polls
    # /api/ping every 0.2s while the daemon comes up, so a slow start would
    # bury its own "running at ..." line under a stack of transport chatter.
    # Transport logs are debugging detail, not product output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Top-level commands are machine/repo scoped. Runtime identity is bound by
    # `bobi agent <name> ...` or inherited BOBI_ROOT in child processes.
    if ctx.invoked_subcommand is None:
        from bobi.webapp import daemon

        try:
            st = daemon.start(open_browser=True)
        except RuntimeError as e:
            raise click.ClickException(str(e))
        click.echo(f"bobi app is running at {st.url} (pid {st.pid})")
    return


@main.group()
@click.argument("name")
@click.pass_context
def agent(ctx, name):
    """Operate on one installed Bobi Agent runtime."""
    if ctx.invoked_subcommand == "ui":
        ctx.obj = {"agent": name, "root": None}
        return
    root = _bind_agent_runtime(name)
    ctx.obj = {"agent": name, "root": root}


def _has_systemd_service() -> bool:
    """Check if bobi is managed by a systemd user service."""
    svc = Path.home() / ".config" / "systemd" / "user" / "bobi.service"
    if not svc.exists():
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-enabled", "bobi"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _systemctl(action: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", action, "bobi"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        click.echo(f"systemctl {action} failed: {result.stderr.strip()}", err=True)
        return False
    return True




@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("start_args", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def supervise(ctx, start_args):
    """Supervise this agent's manager as the terminal process.

    Spawns the manager, probes it from outside, publishes heartbeat and
    lifecycle telemetry onto the event bus, and listens on the deployment's
    admin topic so an operator can restart a manager that has wedged.

    Everything after `--` is forwarded verbatim to the manager's start
    command; `--foreground` keeps it a supervisable child. This is the
    process a container entrypoint or pod spec runs as PID 1.

    Usage:
        bobi agent eng supervise -- --foreground
        bobi agent eng supervise -- --foreground --subscribe linear:MOD
    """
    from bobi.supervisor.__main__ import run

    # The `agent` group already bound the runtime root from the agent name;
    # hand it straight to the sidecar rather than letting it re-resolve from
    # BOBI_ROOT, which would silently win over the name the operator typed.
    root = (ctx.obj or {}).get("root")
    if root is None:
        raise click.UsageError(
            "`supervise` runs against an installed agent: "
            "bobi agent <name> supervise -- --foreground"
        )
    raise SystemExit(run(root, list(start_args)))


@main.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in the foreground (default: daemonize)")
@click.option("--fresh", is_flag=True, help="Wipe session and start clean")
@click.option("--subscribe", multiple=True, help="Additional subscriptions (e.g. linear:MOD)")
def start(foreground, fresh, subscribe):
    """Start the selected Bobi Agent.

    Reads the installed agent config from run/package/agent.yaml. If no
    agent is installed, run `bobi agents install <path> --name <name>` first.

    Usage:
        bobi agent eng start
        bobi agent eng start --foreground
        bobi agent eng start --subscribe linear:MOD
    """
    from bobi.service import (
        AlreadyRunning,
        LaunchTimeout,
        NestedRuntimeError,
        NoAgentInstalled,
        PreflightFailed,
        run_team_foreground,
        spawn_team,
    )

    project_path = _detect_project_root()

    click.echo("Running preflight checks...")
    try:
        if foreground:
            root = logging.getLogger()
            root.handlers = [
                h for h in root.handlers if not isinstance(h, logging.FileHandler)
            ]
            run_team_foreground(project_path, fresh=fresh, subscribe=list(subscribe))
            return
        result = spawn_team(project_path, fresh=fresh, subscribe=list(subscribe))
    except NoAgentInstalled as exc:
        click.echo("No agent installed. Run `bobi agents install <path> --name <name>` first.", err=True)
        if exc.available:
            click.echo("Available packs to install:", err=True)
            for name, source in exc.available:
                click.echo(f"  {name:20s} [{source}]", err=True)
        raise SystemExit(1)
    except PreflightFailed as exc:
        validation = exc.validation
        click.echo("Preflight:")
        click.echo(validation.format())
        click.echo("\nStartup blocked — fix the issues above.", err=True)
        raise SystemExit(1)
    except AlreadyRunning as exc:
        click.echo(
            f"Already running (pid {exc.pid}). "
            f"Use `bobi agent {paths.agent_name_for_root(project_path)} restart`."
        )
        return
    except NestedRuntimeError as exc:
        click.echo(
            f"A manager is already running at {exc.ancestor} (pid {exc.pid}). "
            f"Sub-agents in {paths.agent_name_for_root(project_path)} will register with that runtime. "
            f"Stop the ancestor first if you need an independent instance here.",
            err=True,
        )
        raise SystemExit(1)
    except LaunchTimeout as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    validation = result.validation
    if getattr(validation, "checks", None):
        click.echo("Preflight:")
        click.echo(validation.format())
        degraded = [c for c in validation.checks if not c.ok and not c.required]
        if degraded:
            names = ", ".join(c.name for c in degraded)
            click.echo(
                f"\nStarting in degraded mode — optional services unavailable "
                f"until configured: {names}.",
                err=True,
            )
    if fresh:
        click.echo("Cleared manager session — starting fresh.")
    elif result.image_rotated:
        click.echo("Installed image changed — rotating session.")
    _print_startup_info(project_path, result.startup.pid, result.startup.log_file)


def _materialize_local_deps(pack_dir: Path, project_path: Path, *,
                            non_interactive: bool) -> None:
    """Drive the local brain to install the team's declared deps (#428 Stage 5).

    The `--with-deps` post-compose pass: resolve the team's full dependency set,
    verify what's already satisfied (idempotent skip), preview a plan, confirm,
    then materialize the rest on THIS host under the team's brain. `host:`
    capabilities are surfaced as a guided fix, never attempted, and sudo is only
    used behind an explicit confirm. Partial failure is non-fatal — doctor and
    the dispatch preflight still gate — so this never raises into install.
    """
    from bobi import local_deps
    from bobi.brain import DEFAULT_BRAIN
    from bobi.build_render import _workspace_root
    from bobi.config import Config
    from bobi.env import child_agent_env
    from bobi.host_caps import host_caps_for_deps
    from bobi.tool_library import resolve_team_dependencies

    click.echo()
    try:
        deps = resolve_team_dependencies(pack_dir, _workspace_root(pack_dir))
    except Exception as e:  # noqa: BLE001 — a dep-resolution failure is non-fatal
        click.echo(f"Could not resolve dependencies (--with-deps skipped): {e}",
                   err=True)
        return
    if not deps:
        click.echo("--with-deps: this team declares no dependencies.")
        return

    # The team's declared brain drives the install, else the local default.
    try:
        brain = Config.load(project_path).brain_kind or DEFAULT_BRAIN
    except Exception:
        brain = DEFAULT_BRAIN

    # Bind the installed runtime so the brain session + `success` checks resolve
    # this team's paths and credentials (its run/.env).
    paths.bind_root(project_path)
    base_env = child_agent_env(project_path)

    plan = local_deps.plan_dependencies(deps, brain=brain, base_env=base_env)
    unmet_caps = [c for c in host_caps_for_deps(deps) if c.satisfied() is False]

    click.echo(f"Dependency check (brain: {brain}):")
    for dp in plan.satisfied:
        click.echo(f"  [ok]   {dp.dep.name} — already satisfied, skipping")
    for dp in plan.todo:
        sudo = " (may need sudo)" if dp.needs_sudo else ""
        click.echo(f"  [todo] {dp.dep.name} — will materialize{sudo}")
    for dp in plan.unmaterializable:
        click.echo(f"  [warn] {dp.dep.name} — unsatisfied but has no install/"
                   f"guide to materialize from; fix manually")
    for cap in unmet_caps:
        click.echo(f"  [host] {cap.spec} — host capability, provision manually: "
                   f"`{cap.fix_command()}`")

    if not plan.todo:
        click.echo("Nothing to install.")
        return

    if not non_interactive and not click.confirm(
            f"\nInstall {len(plan.todo)} dependency(ies) on this machine?",
            default=True):
        click.echo("Skipped dependency materialization.")
        return

    allow_sudo = False
    if plan.needs_sudo:
        if non_interactive:
            click.echo("Some steps may need sudo; skipping sudo "
                       "(non-interactive). Re-run interactively to allow it.")
        else:
            allow_sudo = click.confirm(
                "Some steps may require sudo (system packages). Allow sudo?",
                default=False)

    results = local_deps.install_dependencies(
        plan.todo, brain=brain, allow_sudo=allow_sudo, base_env=base_env)

    click.echo("\nDependency materialization:")
    for r in results:
        glyph = "ok" if r.ok else "FAIL"
        click.echo(f"  [{glyph}] {r.dep}"
                   + (f" — {r.detail}" if r.detail and not r.ok else ""))
        for cmd in r.transcript:
            click.echo(f"         ran: {cmd}")
    failed = [r.dep for r in results if not r.ok]
    if failed:
        slot = paths.agent_name_for_root(project_path)
        click.echo(f"\n{len(failed)} dependency(ies) not satisfied: "
                   f"{', '.join(failed)}. The team still installed; fix these "
                   f"and re-run `bobi agents install ... --with-deps`, or "
                   f"`bobi agent {slot} doctor`.", err=True)


@main.command("login-bootstrap")
@click.option("--timeout", default=600, type=int,
              help="Seconds to wait for the pasted auth code (default: 600).")
def login_bootstrap(timeout):
    """Bootstrap subscription auth over a chat channel + the event bus.

    For BOBI_AUTH=subscription first boot with no credentials on the
    volume: drive `claude auth login --claudeai` under a pty, post the OAuth
    URL to $BOBI_LOGIN_CHANNEL, and wait for the pasted code to arrive as
    a chat event over the event bus. Idempotent — a no-op if credentials
    already exist. Fallback: `fly ssh console` then `claude auth login`.

    The destination is $BOBI_LOGIN_CHANNEL only. This command is on the
    `agent` group any worker can reach and the URL it posts grants
    credentials, so it takes no caller-chosen destination; an operator
    retargeting a one-off sets the env var on the invocation.
    """
    from bobi import auth_bootstrap
    project_path = _detect_project_root()

    if auth_bootstrap.credentials_exist():
        click.echo("Subscription credentials already present — nothing to do.")
        return
    try:
        ok = auth_bootstrap.run_bootstrap(project_path, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — surface a clean CLI error
        click.echo(f"Login bootstrap failed: {exc}", err=True)
        raise SystemExit(1)
    if not ok:
        click.echo("Login bootstrap did not produce credentials.", err=True)
        raise SystemExit(1)
    click.echo("Subscription login complete.")


@main.command()
@click.argument("pack")
@click.option("--name", "slot_name", default=None,
              help="Installed Bobi Agent slot name (defaults to package name).")
@click.option("--non-interactive", is_flag=True,
              help="Skip prompts; read secrets from the environment. "
                   "Suitable for container entrypoints and CI.")
@click.option("--pinned", is_flag=True,
              help="Resolve any `from:` base teams registry-only at locked "
                   "versions (ignore local sibling checkouts). For "
                   "reproducible CI/deploy installs.")
@click.option("--with-deps", "with_deps", is_flag=True,
              help="After composing, drive the local brain to install the "
                   "team's declared dependencies on THIS machine (#428): each "
                   "dependency's `success` is verified, already-satisfied ones "
                   "are skipped, and nothing runs sudo without an explicit "
                   "confirm. Mutates the host — previews a plan and confirms "
                   "first.")
def install(pack, slot_name, non_interactive, pinned, with_deps):
    """Install a Bobi Agent into the machine-wide Bobi home.

    PACK is a local directory path, a local `.tar.gz` archive, a public
    `.tar.gz` URL, or a name to fetch from a remote registry.

    Resolution order:
      1. URL (http/https) → fetch a team archive directly
      2. Local `.tar.gz`/`.tgz` file → extract + install
      3. Local directory path (absolute or relative)
      4. Remote registry lookup by name

    Usage:
        bobi agents install agents/eng-team --name eng
        bobi agents install /path/to/my-agent --name eng
        bobi agents install ./eng-team.tar.gz --name eng
        bobi agents install https://example.com/eng-team.tar.gz --name eng
        bobi agents install eng-team --name eng
        bobi agents install eng-team --name eng --non-interactive
    """
    project_path = paths.home_dir()

    pack_str = str(pack)
    if pack_str.startswith(("http://", "https://")):
        # Public URL → fetch a team .tar.gz directly (the container first-boot /
        # CI injection seam). The installed copy is the source of truth.
        from bobi.registry import fetch_from_url
        try:
            click.echo(f"'{pack}' is a URL, fetching team archive...")
            pack_dir, _ = fetch_from_url(pack_str)
        except Exception as e:
            click.echo(f"Failed to fetch '{pack}': {e}", err=True)
            raise SystemExit(1)
    elif pack_str.endswith((".tar.gz", ".tgz")) and Path(pack).is_file():
        # Local team archive → extract + install (the ssh-push delivery seam:
        # `bobi deploy` pushes a tarball onto the instance, which installs
        # it from the volume). The installed copy is the source of truth.
        from bobi.registry import fetch_from_archive
        try:
            click.echo(f"'{pack}' is a local archive, extracting team...")
            pack_dir, _ = fetch_from_archive(Path(pack).resolve())
        except Exception as e:
            click.echo(f"Failed to install '{pack}': {e}", err=True)
            raise SystemExit(1)
    elif (pack_path := Path(pack).resolve()).is_dir() and (pack_path / "agent.yaml").exists():
        pack_dir = pack_path
    else:
        # Try remote registry. A trailing `@version` pins an immutable per-team
        # asset (D-6: split on the last `@`); a bare name takes latest. The `@`
        # is meaningful ONLY here — the URL / local-archive / local-dir branches
        # above never split on it.
        from bobi.registry import fetch, split_team_ref
        name, version = split_team_ref(pack_str)
        try:
            label = f"{name}@{version}" if version else name
            click.echo(f"'{pack}' is not a local team directory, fetching "
                       f"{label} from remote...")
            fetch(name, version=version)
            resolved = _resolve_agent_pack(name, project_path)
            if not resolved:
                click.echo(f"Failed to fetch '{pack}' from remote registries.", err=True)
                raise SystemExit(1)
            pack_dir = resolved
        except SystemExit:
            raise
        except Exception as e:
            click.echo(f"Failed to fetch '{pack}': {e}", err=True)
            raise SystemExit(1)

    agent_name = slot_name or pack_dir.name
    project_path = paths.agent_run_root(agent_name)
    project_path.mkdir(parents=True, exist_ok=True)
    paths.package_dir(project_path).mkdir(parents=True, exist_ok=True)
    paths.workspace_dir(project_path).mkdir(parents=True, exist_ok=True)

    # Local source of truth: the team source is user-authored and the installed
    # package is a generated build artifact.
    local_source = not pack_dir.is_relative_to(paths.agent_cache_dir())

    try:
        _install_pack(pack_dir, project_path, local_source, pinned=pinned)
    except Exception as e:
        from bobi.compose import ComposeError
        if isinstance(e, ComposeError):
            click.echo(f"\n{e}", err=True)
            raise SystemExit(1)
        raise
    _write_install_gitignore(project_path, local_source)

    click.echo(f"Installed Bobi Agent '{agent_name}' into {project_path}")

    installed = paths.package_dir(project_path)
    parts = []
    for subdir in ["roles", "tools", "workflows", "monitors", "context"]:
        d = installed / subdir
        if d.is_dir():
            items = [p.name for p in d.iterdir()]
            if items:
                parts.append(f"  {subdir}: {', '.join(sorted(items))}")
    if (pack_dir / "workspace").is_dir():
        parts.append("  workspace: seeded to workspace/ (existing files kept)")
    if parts:
        click.echo("\n".join(parts))

    # Collect referenced env vars and write run/.env
    from bobi.config import find_env_var_refs, parse_env_file, write_env_file
    env_refs = find_env_var_refs(project_path)
    if env_refs:
        env_file = paths.env_path(project_path)
        existing = parse_env_file(env_file)

        click.echo()
        missing = [r for r in env_refs
                   if r.name not in existing and r.name not in os.environ]

        if non_interactive:
            # Pull values from the environment — never prompt.
            for ref in env_refs:
                if ref.name not in existing and ref.name in os.environ:
                    existing[ref.name] = os.environ[ref.name]
            # A bare ${VAR} is a required secret; ${VAR:-default} carries its
            # own fallback and is optional. Fail fast on missing required
            # secrets so a container entrypoint (`install --non-interactive
            # && start`) never marches into a broken start with empty
            # credentials.
            #
            # build_only names are excluded: they appear only under `build:`,
            # which bakes an image layer, and nothing reads them to run an
            # agent. Requiring them here blocked an install whose dependency
            # was ALREADY materialized in the image — the deploy side
            # deliberately withholds build secrets from the runtime env-file,
            # so demanding them here made the two halves contradict.
            required_missing = [r.name for r in missing
                                if r.required and not r.build_only]
            optional_missing = [r.name for r in missing if not r.required]
            if required_missing:
                click.echo(
                    "Error: required secrets missing from the environment: "
                    + ", ".join(required_missing)
                    + ". Set them (e.g. `fly secrets set`) and re-run "
                    "`bobi agents install --non-interactive`.",
                    err=True)
                raise SystemExit(1)
            if optional_missing:
                click.echo(
                    "Warning: optional env vars unset: "
                    + ", ".join(optional_missing), err=True)
            write_env_file(env_file, existing)
        elif missing:
            click.echo("This agent needs credentials:")
            for ref in missing:
                hint = _ENV_VAR_HINTS.get(
                    ref.name, "" if ref.required else "optional")
                label = f"  {ref.name} ({hint})" if hint else f"  {ref.name}"
                try:
                    value = click.prompt(label, default="", show_default=False)
                except (EOFError, click.Abort):
                    value = ""
                if value:
                    existing[ref.name] = value

            write_env_file(env_file, existing)
            click.echo(f"Credentials saved to {env_file}")

    if with_deps:
        _materialize_local_deps(pack_dir, project_path,
                                non_interactive=non_interactive)

    if local_source:
        try:
            src_display = pack_dir.relative_to(project_path)
        except ValueError:
            src_display = pack_dir
        click.echo(f"\nSource of truth: {src_display}/ — edit there and reinstall to change the Bobi Agent.")
    else:
        click.echo(f"\nSource of truth: {pack_dir}/")

    click.echo(f"Run `bobi agent {agent_name} start` to launch.")

    # An in-place bobi upgrade replaces the framework underneath whatever is
    # already running, and a team reinstall does not restart it (#928). Say so
    # here, where the operator is, instead of leaving it for doctor to be asked.
    from bobi.launch_stamp import stale_processes

    stale = stale_processes(project_path)
    if stale:
        click.echo("\nStill running the code this install replaced:")
        for process in stale:
            click.echo(f"  {process.name}: {process.detail}")
        click.echo("Restart to pick it up: "
                   + ", ".join(f"`{p.remedy}`" for p in stale))


@main.command()
@click.argument("name", required=False)
@click.option("--model", default=None,
              help="Model for the setup session (alias or full ID).")
def setup(name, model):
    """Interactively design, build, and install an agent team.

    Opens a local web UI (on 127.0.0.1) that goes from an idea to a
    runnable agent team: describe what you want, let bobi suggest what
    it can do on its own, connect services, watch it build the pack, then
    review and install. Interrupt anytime — reopening setup for the same
    team resumes where you left off.
    """
    from urllib.parse import quote, urlencode
    import webbrowser

    from bobi.webapp import daemon

    try:
        st = daemon.start(open_browser=False)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    target = st.url + (f"#/setup/{quote(name)}" if name else "#/setup")
    if model:
        target += "?" + urlencode({"model": model})
    webbrowser.open(target)
    click.echo(f"bobi setup is open at {target}")


@main.group("app")
def app_group():
    """Manage the Bobi web app (dashboard for all your agents)."""


@app_group.command("start")
@click.option("--no-browser", is_flag=True, help="Don't open a browser.")
def app_start(no_browser):
    """Start the web app in the background (idempotent)."""
    from bobi.webapp import daemon

    try:
        st = daemon.start(open_browser=not no_browser)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"bobi app is running at {st.url} (pid {st.pid})")


@app_group.command("stop")
@click.option("--force", is_flag=True,
              help="Signal the recorded pid even if it does not answer as the app")
def app_stop(force):
    """Stop the web app daemon."""
    from bobi.webapp import daemon

    st = daemon.stop(force=force)
    if st.unverified:
        click.echo(
            f"Not running — cleared a stale pid file. Process {st.pid} is "
            "alive but does not answer as the bobi app (the pid was reused "
            "after a crash), so it was left alone. Use --force to signal it "
            "anyway."
        )
        return
    if st.not_permitted:
        raise click.ClickException(
            f"Process {st.pid} is the bobi app but runs as another user, so "
            "the stop signal was refused. It is still running; stop it as "
            "that user (or with sudo)."
        )
    click.echo(f"Stopped (pid {st.pid})." if st.pid else "Not running.")


@app_group.command("restart")
def app_restart():
    """Restart the web app daemon."""
    from bobi.webapp import daemon

    daemon.stop()
    try:
        st = daemon.start(open_browser=False)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"bobi app is running at {st.url} (pid {st.pid})")


@app_group.command("status")
def app_status():
    """Show whether the web app daemon is running."""
    from bobi.webapp import daemon

    st = daemon.status()
    if st.running:
        click.echo(f"Running at {st.url} (pid {st.pid})")
    else:
        click.echo("Not running. Start it with `bobi app start`.")
        raise SystemExit(1)


@app_group.command("run", hidden=True)
def app_run():
    """Run the web app server in the foreground (the daemon child)."""
    from bobi.webapp import daemon

    raise SystemExit(daemon.run_foreground())


@main.command()
@click.argument("deployment", required=False)
@click.option("--app", default=None, hidden=True)
@click.option("--port", "local_port", default=None, type=int, hidden=True)
@click.option("--remote-port", default=None, type=int, hidden=True)
@click.option("--no-browser", is_flag=True, help="Don't open a browser window.")
@click.option("--check", is_flag=True, hidden=True)
@click.pass_context
def ui(ctx, deployment, app, local_port, remote_port, no_browser, check):
    """View and chat with an agent team's agents in a web UI.

    \b
    Local agent:  bobi agent eng ui

    Local mode serves a card per active agent on 127.0.0.1 and talks to the
    running team over the event server (so the team must already be started).
    Deployed instances are administered through the control plane.
    """
    if deployment or app or local_port or remote_port or check:
        raise click.UsageError(
            "`bobi agent <name> ui <deployment>` was removed. "
            "Deployed instances are administered through the control plane."
        )

    # Local mode: bind the registry + event-server root so the cross-process
    # `deliver` behind the chat reaches the same team start command runs.
    selected = ""
    if ctx.parent is not None and isinstance(ctx.parent.obj, dict):
        selected = str(ctx.parent.obj.get("agent") or "")
    if not selected:
        raise click.UsageError("local UI requires `bobi agent <name> ui`")

    from urllib.parse import quote
    import webbrowser

    from bobi.webapp import daemon

    try:
        st = daemon.start(open_browser=False)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    target = st.url + f"#/agents/{quote(selected)}"
    if not no_browser:
        webbrowser.open(target)
    click.echo(f"bobi app is running at {target} (pid {st.pid})")


@main.command()
@click.option("--force", is_flag=True, help="Send SIGKILL if SIGTERM doesn't work")
def stop(force):
    """Stop the selected Bobi Agent.

    Usage:
        bobi agent eng stop
        bobi agent eng stop --force
    """
    if _has_systemd_service() and not force:
        click.echo("Stopping via systemd...")
        _systemctl("stop")
        return

    project_path = _detect_project_root()
    from bobi.service import stop_team

    result = stop_team(project_path, force=force)
    if result.invalid_pid:
        click.echo("Invalid PID file — cleaning up.")
    elif result.stale:
        click.echo(f"Process {result.pid} not found — cleaning up stale PID file.")
    elif result.permission_denied:
        click.echo(f"No permission to signal process {result.pid}.", err=True)
    elif result.stopped:
        click.echo(f"Stopping bobi (pid {result.pid})...")
        click.echo("Stopped.")
    elif result.killed:
        click.echo(f"Stopping bobi (pid {result.pid})...")
        click.echo("Killed.")
    elif result.still_running:
        click.echo(f"Stopping bobi (pid {result.pid})...")
        click.echo("Process didn't exit — try: bobi agent <name> stop --force")
    else:
        click.echo("No PID file found — bobi is not running.")

    if result.event_server_running:
        click.echo(
            f"Event server is still running on port {result.event_server_port}. "
            "Use `bobi agent <name> event-server stop` to stop it."
        )


@main.command()
@click.option("--fresh", is_flag=True, help="Wipe manager session and start clean")
@click.option("--detached-worker", is_flag=True, hidden=True)
def restart(fresh, detached_worker):
    """Stop and restart the selected Bobi Agent.

    Usage:
        bobi agent eng restart
        bobi agent eng restart --fresh   # fresh manager session
    """
    if detached_worker:
        ctx = click.get_current_context()
        click.echo(logs.stamped("INFO", f"Restart worker pid {os.getpid()} starting."))
        ctx.invoke(stop)
        ctx.invoke(start, fresh=fresh)
        click.echo(logs.stamped("INFO", "Restart worker finished."))
        return

    if _has_systemd_service():
        # Resolve before touching systemd so a missing installation fails
        # here, not after the service has already been restarted.
        project_path = _detect_project_root()
        if fresh:
            # Wipes the saved session ID, the bubble credential, and the
            # per-session deployment/cursor state together: a fresh start mints
            # a NEW bubble, and stale deployment_state (whose api_key points at
            # a now-orphaned deployment in the old bubble) would split the
            # restarted sessions across bubbles.
            from bobi.service import clear_manager_session
            clear_manager_session(project_path)
            click.echo("Cleared manager session — starting fresh.")
        click.echo("Restarting via systemd...")
        _systemctl("restart")
        result = subprocess.run(
            ["systemctl", "--user", "show", "bobi", "--property=MainPID", "--value"],
            capture_output=True, text=True, timeout=5,
        )
        pid = result.stdout.strip()
        log_path = paths.manager_log_path(project_path)
        click.echo(f"Bobi restarted (pid {pid}). Logs: {log_path}")
        return

    from bobi.service import RestartFailed, restart_team

    project_path = _detect_project_root()
    try:
        result = restart_team(project_path, fresh=fresh)
    except RestartFailed as exc:
        click.echo(exc.report(), err=True)
        raise SystemExit(1)
    if result.output:
        click.echo(result.output, nl=not result.output.endswith("\n"))


def _resolve_address(to: str | None) -> str | None:
    """Resolve a friendly address to a session name.

    'manager' or None → finds the coordinator session by the installed
    package's entry_point role, then the literal role "manager".
    Anything else → used as-is (exact session name).
    """
    project_path = _detect_project_root()
    from bobi.service import resolve_address
    return resolve_address(project_path, to)


@main.command()
@click.argument("text", required=True)
@click.option("--to", default=None, help="Target session (default: manager)")
@click.option("--wait", is_flag=True, help="Block until the session responds")
@click.option("--timeout", default=300, type=int, help="Timeout in seconds (with --wait)")
def message(text, to, wait, timeout):
    """Send a message to any session via its inbox.

    Usage:
        bobi agent eng message "what are you working on?"
        bobi agent eng message --to eng-42-implement "try a different approach"
        bobi agent eng message --to manager "status?" --wait
    """
    from bobi.service import MessageDeliveryError, send_message

    project_path = _detect_project_root()
    try:
        result = send_message(
            project_path, text, wait=wait, session=to, timeout=timeout, sender="cli"
        )
        if wait and result.response:
            click.echo(result.response)
        else:
            click.echo(f"Sent to {result.address}")
    except MessageDeliveryError as exc:
        msg = str(exc)
        if msg.startswith("No active session"):
            click.echo(msg, err=True)
        else:
            click.echo(f"Failed: {msg}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--to", default=None, help="Target session (default: manager)")
def compact(to):
    """Compact a session's context now — flush its decision log and rotate.

    Triggers the same graceful rotation the token cap does, on demand: the
    session writes its decision log to INDEX.md, then swaps to a fresh
    conversation that reloads only that log. Use it when a long-lived
    session has grown slow. Rotation happens at the session's next idle
    moment (it won't interrupt an in-flight turn).

    Usage:
        bobi agent eng compact                       # compact the manager
        bobi agent eng compact --to eng-42-implement # compact a specific session
    """
    from bobi.inbox import deliver
    from bobi.session import COMPACT_SENTINEL

    address = _resolve_address(to)
    if not address:
        target = to or "manager"
        click.echo(f"No active session found for '{target}'.", err=True)
        raise SystemExit(1)

    ok, response = deliver(address, COMPACT_SENTINEL, sender="cli", wait=False)
    if ok:
        click.echo(f"Compaction requested for {address} — it will flush its "
                   f"decision log and rotate at its next idle moment.")
    else:
        click.echo(f"Failed: {response}", err=True)
        raise SystemExit(1)


@main.command(hidden=True)
@click.argument("question", required=True)
@click.option("--timeout", default=300, type=int, help="Timeout in seconds")
def ask(question, timeout):
    """Ask the manager a question (alias for: message --wait)."""
    from bobi.service import MessageDeliveryError, send_message

    project_path = _detect_project_root()
    try:
        result = send_message(
            project_path, question, wait=True, session="manager",
            timeout=timeout, sender="engineer",
        )
        click.echo(result.response)
    except MessageDeliveryError as exc:
        msg = str(exc)
        if msg.startswith("No active session"):
            click.echo("No active manager session found.", err=True)
        else:
            click.echo(f"Failed: {msg}", err=True)
        raise SystemExit(1)


def _parse_conversation_or_exit(conversation):
    """Parse a conversation reference, exiting with a clear error if invalid."""
    from .conversation import parse_conversation

    conv = parse_conversation(conversation)
    if conv is None:
        click.echo(f"Invalid conversation reference: {conversation}", err=True)
        sys.exit(1)
    return conv


def _gateway_call(fn, *args, **kwargs):
    """Run a gateway client call, exiting with its message on failure."""
    from .events.gateway import GatewayError

    try:
        return fn(*args, **kwargs)
    except GatewayError as e:
        click.echo(str(e), err=True)
        click.echo(
            "(replies go through the event server's channel gateway; make "
            "sure the agent is started so the server is running and its "
            "chat workspace is registered)",
            err=True,
        )
        sys.exit(1)


def _unescape_shell_literals(text):
    """Turn literal ``\\n`` / ``\\t`` escapes into real characters.

    Reply text often arrives through shell arguments where newlines survive
    only as two-character escapes; rendering them literally mangles
    multi-line messages. (Previously done by bobi.slack.format_slack_message
    on the old direct-send path.)
    """
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _channels_reply_send(conversation, text, edit_ref, files=None):
    """Send a reply through the channel gateway (#190 Phase 2).

    Text goes out as raw markdown; the gateway owns formatting, truncation,
    and clearing the typing indicator. Mode ``final`` resolves the response
    context: it edits ``edit_ref`` when given, else posts, then clears the
    thinking indicator.
    """
    from .events.gateway import channels_send

    result = _gateway_call(
        channels_send, _detect_project_root(), conversation,
        _unescape_shell_literals(text),
        mode="final", edit_ref=edit_ref, files=files,
    )
    if edit_ref:
        click.echo(f"Updated {edit_ref} in {conversation}")
    else:
        click.echo(f"Sent to {conversation}")
    return result


def _read_conversation(conversation, limit, as_json):
    """Fetch and render a conversation's messages via the gateway."""
    import json as _json

    from .events.gateway import channels_history

    messages = _gateway_call(
        channels_history, _detect_project_root(), conversation, limit)

    if as_json:
        click.echo(_json.dumps(messages, indent=2))
    else:
        for msg in messages:
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            files = msg.get("files", [])
            click.echo(f"[{ts}] {user}: {text}")
            for f in files:
                name = f.get("name", "file")
                mimetype = f.get("mimetype", "")
                click.echo(f"  >> {name} ({mimetype})")
        click.echo(f"\n{len(messages)} message(s)")


@main.command("reply")
@click.argument("conversation")
@click.argument("text", required=False)
@click.option("--edit", "edit_ref", default="", help="Message ref (ts) to edit instead of posting new")
@click.option("--file", "file_path", type=click.Path(exists=True), default=None,
              help="Upload a file into the conversation (TEXT becomes its comment)")
@click.option("--title", default="", help="File title (with --file)")
def reply(conversation, text, edit_ref, file_path, title):
    """Reply into a conversation, on whatever chat channel it came from.

    CONVERSATION is the ``conversation`` reference carried on the inbound
    event - echo it back verbatim. Reads TEXT from stdin when omitted.
    Write plain markdown; the gateway converts it for the channel.

    Usage:
        bobi reply slack:T0952RZRZ0X:dm:D0B51JP1N4C "Hello"
        bobi reply slack:T0952RZRZ0X:channel:C123:thread:171.42 "Thread reply"
        bobi reply slack:T0952RZRZ0X:channel:C123:thread:171.42 --edit 171.99 "Real response"
        bobi reply slack:T0952RZRZ0X:channel:C123:thread:171.42 --file report.pdf "Here's the report"
        bobi reply slack:T0952RZRZ0X:channel:C123:thread:171.42 --edit 171.99 --file out.png "Done - see attached"
    """
    _parse_conversation_or_exit(conversation)

    if text is None:
        # Fail fast on an interactive terminal instead of blocking on EOF
        # (a piped comment also works alongside --file).
        if not sys.stdin.isatty():
            text = click.get_text_stream("stdin").read().rstrip("\n")
        elif file_path is None:
            click.echo("No text to send (pass TEXT or pipe via stdin)", err=True)
            sys.exit(1)
    text = (text or "").strip()
    if not text and file_path is None:
        click.echo("No text to send (pass TEXT or pipe via stdin)", err=True)
        sys.exit(1)
    if edit_ref and file_path and not text:
        # The gateway edits the target message with TEXT, then attaches the
        # file - without text there is nothing to put in the edited message.
        click.echo("--edit with --file requires TEXT", err=True)
        sys.exit(1)

    files = None
    if file_path is not None:
        from pathlib import Path

        from .events.gateway import file_payload
        files = [file_payload(Path(file_path), title=title)]

    _channels_reply_send(conversation, text, edit_ref, files)


@main.command("read-conversation")
@click.argument("conversation")
@click.option("--limit", "-n", default=100, help="Max messages to fetch (default: 100)")
@click.option("--json-output", "as_json", is_flag=True, help="Output as JSON")
def read_conversation(conversation, limit, as_json):
    """Read the messages of a conversation (e.g. the Slack thread it anchors).

    Usage:
        bobi read-conversation slack:T0952RZRZ0X:channel:C123:thread:171.42
        bobi read-conversation slack:T0952RZRZ0X:dm:D456:thread:171.42 --json-output
    """
    _parse_conversation_or_exit(conversation)
    _read_conversation(conversation, limit, as_json)


@main.command("create-slack-bot")
@click.option("--app-name", default=None,
              help='Display name for the Slack app (default: "bobi agent"; '
                   "prompted when run interactively)")
@click.option("--event-server", default="",
              help="Event server base URL (default: the configured server, "
                   "else prompted interactively, else the bobi cloud)")
@click.option("--socket-mode", is_flag=True,
              help="Generate a Socket Mode manifest for a local event server")
@click.option("--format", "fmt", type=click.Choice(["yaml", "json"]),
              default="yaml", help="Manifest output format")
@click.option("--output", "-o", "output", type=click.Path(), default="",
              help="Write the manifest to a file instead of stdout")
@click.option("--url/--no-url", "show_url", default=True,
              help="Print a one-click 'create from manifest' link")
@click.option("--open/--no-open", "open_browser", default=None,
              help="Open the one-click create link in your browser "
                   "(default: when run interactively; use --no-open for "
                   "headless/CI)")
def create_slack_bot(
    app_name, event_server, socket_mode, fmt, output, show_url, open_browser,
):
    """Create a Slack app (bot) for bobi - generates the manifest and a
    one-click create link, and opens it in your browser.

    Every bobi Slack app needs the same scopes and events.
    This stamps them out for HTTP Events API or Socket Mode so a working app is
    one step away: click the link it opens, feed the file to the Slack CLI
    (`slack create <name> --manifest manifest.json`), or POST it to the App
    Manifest API.

    Usage:
        bobi create-slack-bot
        bobi create-slack-bot --app-name "Eng Bot" --format json -o manifest.json
        bobi create-slack-bot --event-server https://my-worker.workers.dev
        bobi create-slack-bot --socket-mode
        bobi create-slack-bot --no-open                # just print the link
    """
    from .config import DEFAULT_EVENT_SERVER
    from .slack_manifest import (
        create_app_url, manifest_to_json, render_manifest, webhook_url,
    )

    interactive = _interactive_terminal()

    if app_name is None:
        app_name = (click.prompt("Slack app display name", default="bobi agent")
                    if interactive else "bobi agent")

    if not event_server:
        # Resolve from the project config when run inside an install; this
        # command also works before any Bobi Agent is installed, so a missing
        # root is fine.
        try:
            project_path = _detect_project_root()
        except click.UsageError:
            project_path = None
        if project_path:
            from .config import Config
            event_server = Config.load(project_path).event_server_url
    if not event_server and interactive and not socket_mode:
        # No configured server: let the user pick before the manifest is
        # rendered and the create page opens. Slack must be able to reach
        # this URL from the internet, so a laptop running the local event
        # server needs a public tunnel in front of localhost:8080.
        click.echo("Where should Slack send events (the app's request URL)?")
        click.echo("  Press Enter to use the bobi cloud event server, or type "
                   "your own URL.")
        click.echo("  Running the agent on this machine with the local event "
                   "server? Slack can't reach localhost - put a public tunnel "
                   "(e.g. cloudflared or ngrok) in front of localhost:8080 "
                   "and enter the tunnel URL.")
        event_server = click.prompt("Event server URL",
                                    default=DEFAULT_EVENT_SERVER)
        click.echo("")
    if not event_server:
        # Non-interactive with nothing configured: the bobi cloud.
        event_server = DEFAULT_EVENT_SERVER

    try:
        manifest_yaml = render_manifest(
            app_name, event_server, socket_mode=socket_mode,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    rendered = manifest_to_json(manifest_yaml) if fmt == "json" else manifest_yaml

    if output:
        Path(output).write_text(rendered.rstrip("\n") + "\n")
        click.echo(f"Wrote {fmt} manifest to {output}")
    else:
        click.echo(rendered)

    if show_url:
        create_url = create_app_url(manifest_yaml)
        click.echo("")
        if not socket_mode:
            click.echo(f"Request URL:  {webhook_url(event_server)}")
        click.echo("Create the app in one click:")
        click.echo(f"  {create_url}")
        # Open the browser by default when interactive; --open/--no-open
        # forces either way. The default (None) stays quiet under pipes, CI,
        # and the test runner so it never tries to launch a browser there.
        should_open = (
            open_browser if open_browser is not None else sys.stdout.isatty()
        )
        if should_open:
            click.launch(create_url)
            click.echo("")
            click.echo("Opened the create page in your browser.")
    if socket_mode:
        click.echo("", err=True)
        click.echo("Next steps:", err=True)
        click.echo("  1. Create or import the app from this manifest.", err=True)
        click.echo(
            "  2. Generate an xapp- app-level token with connections:write.",
            err=True,
        )
        click.echo(
            "  3. For an existing pack, add credentials.app_token: "
            "${SLACK_APP_TOKEN:-} to its Slack service and reinstall it.",
            err=True,
        )
        click.echo(
            "  4. Store the xapp- token as SLACK_APP_TOKEN alongside "
            "SLACK_BOT_TOKEN.",
            err=True,
        )
        click.echo(
            "  5. Start the self-hosted agent and run doctor to confirm "
            "the connection.",
            err=True,
        )


@main.group()
def transcript():
    """Session transcripts — view, search, and index conversation history."""
    _detect_project_root()


@transcript.command("show")
@click.argument("session", default="manager")
@click.option("-n", "--lines", default=30, help="Number of recent messages to show")
@click.option("-f", "--follow", is_flag=True, help="Follow mode — stream new entries")
def transcript_show(session, lines, follow):
    """Show the transcript for a session.

    Usage:
        bobi agent eng transcript show manager        # manager transcript
        bobi agent eng transcript show eng-70         # engineer transcript
        bobi agent eng transcript show manager -n 50  # last 50 messages
        bobi agent eng transcript show manager -f     # follow mode
    """
    transcript_path = _find_transcript(session)
    if not transcript_path:
        return

    if follow:
        import time
        last_size = 0
        all_lines = transcript_path.read_text().strip().splitlines()
        for line in all_lines[-lines:]:
            _print_transcript_entry(line)
        last_size = transcript_path.stat().st_size
        try:
            while True:
                time.sleep(1)
                cur_size = transcript_path.stat().st_size
                if cur_size > last_size:
                    with open(transcript_path) as f:
                        f.seek(last_size)
                        for line in f:
                            _print_transcript_entry(line.strip())
                    last_size = cur_size
        except KeyboardInterrupt:
            pass
    else:
        all_lines = transcript_path.read_text().strip().splitlines()
        for line in all_lines[-lines:]:
            _print_transcript_entry(line)


def _find_transcript(session: str) -> Path | None:
    """Find the log file for a session."""
    from bobi.sdk import get_registry, session_log_path

    if session == "manager":
        from bobi.service import manager_session_name
        session = manager_session_name(_detect_project_root())

    # Primary: session dir log
    session_log = session_log_path(session)
    if session_log.exists():
        return session_log

    # Fallback: Claude Code transcript via session ID
    from bobi.chat_history import find_claude_transcript
    from bobi.sdk import _sessions_dir
    id_file = _sessions_dir() / f"{session}.id"
    if id_file.exists():
        session_id = id_file.read_text().strip()
        transcript = find_claude_transcript(session_id)
        if transcript is not None:
            return transcript

    click.echo(f"No session '{session}'.")
    registry = get_registry()
    active = [e for e in registry.list_active() if e.role == "engineer"]
    if active:
        names = [e.name for e in active]
        click.echo(f"Active: {', '.join(sorted(names))}")
    sessions = _sessions_dir()
    recent_dirs = sorted(
        [d for d in sessions.iterdir() if d.is_dir() and (d / "state.json").exists()],
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    recent_names = [d.name for d in recent_dirs[:10] if not d.name.startswith("bobi-mgr")]
    if recent_names:
        click.echo(f"Recent: {', '.join(recent_names)}")
    return None


def _print_transcript_entry(line: str) -> None:
    """Render one JSONL line from a Claude Code transcript or activity log."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        # Plain text lines (e.g. orchestrator print output)
        line = line.strip()
        if line:
            click.echo(f"  {line}")
        return

    # Activity log format (from orchestrator/engineer subprocess)
    event = obj.get("event", "")
    if event == "response":
        import datetime
        ts = datetime.datetime.fromtimestamp(obj.get("ts", 0)).strftime("%H:%M:%S")
        text = obj.get("text", "")[:300]
        click.echo(f"{ts}  ← {text}")
        return
    if event == "tool_use":
        import datetime
        ts = datetime.datetime.fromtimestamp(obj.get("ts", 0)).strftime("%H:%M:%S")
        tool = obj.get("tool", "")
        inp = obj.get("input", "")[:150]
        click.echo(f"{ts}  ⚙ {tool}: {inp}")
        return
    if event == "stop":
        click.echo(f"  ◼ turn complete")
        return

    # Claude Code transcript format
    msg_type = obj.get("type", "")
    ts = obj.get("timestamp", "")[:19]

    if msg_type in ("human", "user"):
        content = obj.get("message", {}).get("content", [])
        text = ""
        for part in content:
            if isinstance(part, str):
                text += part
            elif isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
        text = text.strip()
        if text:
            # Truncate long event payloads but show Slack messages in full
            display = text[:300] + "..." if len(text) > 300 else text
            click.echo(f"\n{ts}  → {display}")

    elif msg_type == "assistant":
        content = obj.get("message", {}).get("content", [])
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text", "").strip()
                if text:
                    click.echo(f"{ts}  ← {text}")
            elif part.get("type") == "tool_use":
                name = part.get("name", "")
                inp = part.get("input", {})
                if isinstance(inp, dict):
                    summary = inp.get("command", inp.get("description", str(inp)))
                else:
                    summary = str(inp)
                summary = str(summary)[:150]
                click.echo(f"{ts}  ⚙ {name}: {summary}")



@main.command()
def status():
    """Show active agents — manager + engineer sub-agents."""
    project_path = _detect_project_root()

    from bobi.service import team_status

    result = team_status(project_path)
    if result.manager_running:
        click.echo(f"  Agent: running (pid {result.manager_pid})")
    else:
        click.echo("  Agent: stopped")

    active = result.active_agents
    if not active:
        click.echo("  Sub-agents: none active")
        return

    click.echo(f"  Sub-agents: {len(active)} active")
    for e in active:
        rotation_info = f", rotations={e.rotation_count}" if e.rotation_count else ""
        click.echo(f"    {e.name} ({e.role}) — {e.status}{rotation_info}")


@main.command()
@click.option("--browser", is_flag=True, default=False,
              help="Also run /browse + Chromium sandbox checks")
@click.option("--fix", is_flag=True, help="Offer to apply the Chromium sandbox fix (with --browser)")
def doctor(browser, fix):
    """System health check — verify manager, event server, dashboard, repos, workflows.

    Runs a suite of checks and prints a pass/fail line for each.
    Exit 0 if all pass, 1 if any fail.

    Usage:
        bobi agent eng doctor
        bobi agent eng doctor --browser
        bobi agent eng doctor --browser --fix
    """
    from .doctor import run_doctor
    from bobi.validate import status_glyph, supports_unicode

    # Resolve the glyph set once: ✓/✗/⚠, or [OK]/[ERROR]/[WARN] on
    # unicode-stripped terminals. Shared by the no-install warning below and
    # the per-check rows further down.
    unicode = supports_unicode()
    warn_mark = status_glyph(False, False, unicode=unicode)

    # doctor is advisory and must never silently pass without a selected
    # runtime: a green report outside an installation would be a lie.
    if paths.bound_root() is None:
        click.echo(click.style(f"{warn_mark} No Bobi Agent runtime selected — "
                               "agent-scoped checks will report 'no project "
                               "detected'.", fg="yellow"))

    results = run_doctor()

    if browser:
        from . import browser as browser_mod
        if not browser_mod.is_linux():
            click.echo("Note: Chromium sandbox checks are Linux-specific; "
                       "running browser launch checks only.")
        results.extend(browser_mod.run_doctor())

    all_ok = True
    warnings = 0
    sandbox_failure = False
    # Every result here is a bobi.doctor.CheckResult — browser.run_doctor()
    # constructs the same dataclass, and doctor._check_services() converts
    # bobi.validate's look-alike into it at the boundary. Both fields are
    # declared with defaults, so plain attribute access is total.
    for r in results:
        # ✓ ok / ✗ blocking failure / ⚠ non-blocking warning (optional service),
        # with [OK]/[ERROR]/[WARN] fallback on unicode-stripped terminals.
        mark = status_glyph(r.ok, r.required, unicode=unicode)
        click.echo(f"  {mark} {r.name}: {r.detail}")
        if not r.ok:
            if r.required:
                all_ok = False
            else:
                warnings += 1
            if r.hint:
                click.echo(f"      → {r.hint}")
            if browser and r.sandbox_error:
                sandbox_failure = True

    if all_ok:
        if warnings:
            click.echo(f"\nAll required checks passed ({warnings} warning(s)).")
        else:
            click.echo("\nAll checks passed.")
        return

    if sandbox_failure and fix:
        from . import browser as browser_mod
        click.echo()
        _offer_sandbox_fix(browser_mod)
    elif sandbox_failure:
        click.echo("\nRe-run with `bobi agent <name> doctor --browser --fix` to apply the sandbox fix.")

    raise SystemExit(1)


def _offer_sandbox_fix(browser_mod) -> None:
    """Explain the Chromium sandbox issue and interactively apply the fix.

    Used by `bobi agent <name> doctor --fix`. Asks for confirmation before running
    the sudo sysctl change.
    """
    click.echo("Chromium's sandbox is blocked by the AppArmor restriction on")
    click.echo("unprivileged user namespaces — this prevents /browse from running.")
    click.echo()
    click.echo(f"  The fix:  {browser_mod.FIX_COMMAND}")
    click.echo(f"  Persisted in: {browser_mod.SYSCTL_CONF_PATH}")
    click.echo()
    click.echo("  Security tradeoff: this lets any local process create user")
    click.echo("  namespaces, a historical local-privilege-escalation surface.")
    click.echo("  Acceptable on dedicated dev machines. See scripts/install.sh for")
    click.echo("  a narrower per-binary AppArmor alternative and the --no-sandbox fallback.")
    click.echo()

    try:
        if not click.confirm("  Apply the fix now (requires sudo)?", default=False):
            click.echo("  Skipped. Apply it later with the command above.")
            return
    except (EOFError, click.Abort):
        click.echo("  Skipped.")
        return

    ok, message = browser_mod.apply_sandbox_fix(persist=True)
    if ok:
        click.echo(f"  {message}")
        recheck = browser_mod.check_chromium_launch()
        if recheck.ok:
            click.echo("  Verified — Chromium now launches. /browse is ready.")
        else:
            click.echo(f"  Applied, but Chromium still fails: {recheck.detail}")
    else:
        click.echo(f"  Fix failed: {message}", err=True)


@main.group()
def agents():
    """Installed Bobi Agent management."""
    pass


@agents.command("list")
def agents_list():
    """List installed Bobi Agents."""
    installed = paths.list_agents()
    if not installed:
        click.echo("No Bobi Agents installed.")
        return
    for name in installed:
        root = paths.agent_run_root(name)
        state = "running" if paths.manager_pid_path(root).exists() else "stopped"
        click.echo(f"  {name:24s} {state:8s} {root}")


@agent.group("subagents")
def subagents():
    """Launch, list, inspect, and cancel sub-agents."""
    pass


@subagents.command("list")
def subagents_list():
    """List active sub-agents from the selected Bobi Agent runtime."""
    _detect_project_root()
    from bobi.subagent import list_agents as _list_agents

    active = _list_agents()
    if not active:
        click.echo("No active sub-agents.")
        return

    for a in active:
        state = "running" if a["running"] else "done"
        label = a.get("name") or f"{a['run_key']}/{a['phase']}"
        click.echo(f"  {label} — {state} ({a['elapsed_s']}s)")


@subagents.command("show")
@click.argument("ref")
def subagents_show(ref):
    """Show details for a specific sub-agent."""
    _detect_project_root()
    import time as _time
    from bobi.subagent import find_agent

    entry = find_agent(ref)
    if not entry:
        click.echo(f"No sub-agent found for {ref}")
        return

    click.echo(f"  Session: {entry.name}")
    if entry.run_key:
        click.echo(f"  Run key: {entry.run_key}")
    click.echo(f"  Phase:   {entry.phase}")
    if entry.status in ("starting", "running", "idle"):
        elapsed = int(_time.time() - entry.started_at)
        click.echo(f"  Status:  {entry.status} ({elapsed}s)")
    else:
        click.echo(f"  Status:  {entry.status}")
    if entry.cwd:
        click.echo(f"  CWD:     {entry.cwd}")
    if entry.title:
        click.echo(f"  Task:    {entry.title}")


@subagents.command("cancel")
@click.argument("ref")
def subagents_cancel(ref):
    """Cancel a running sub-agent."""
    _detect_project_root()
    from bobi.subagent import cancel_agent

    if cancel_agent(ref):
        click.echo(f"Cancelled {ref}")
    else:
        click.echo(f"No running sub-agent for {ref}")


# `otel` is registered directly on the `agent` group, the `subagents` pattern
# above: no @main.group, no re-parent list entry, and therefore no window in
# which `bobi otel` leaks as a top-level command.


@contextmanager
def _otel_usage_errors():
    """Surface the library's input bounds as click's own usage errors.

    The validators live in ``bobi.otel.validate`` so the abuse suite can prove
    them without a CLI layer; this is the one place their error type is
    translated.
    """
    from bobi.otel.validate import OtelUsageError

    try:
        yield
    except OtelUsageError as exc:
        raise click.UsageError(str(exc)) from None


def _otel_context(ctx) -> tuple[Path, str]:
    """The bound runtime root and the agent name the labels are stamped with."""
    root = _detect_project_root()
    name = (ctx.obj or {}).get("agent")
    if not name:
        # The group invoked outside its `agent` parent, which a CliRunner can do.
        name = paths.agent_name_for_root(root)
    return root, name


def _otel_resolve(ctx, signal: str):
    """Resolve config + resource attributes, or exit with a typed diagnosis."""
    from bobi.otel import config as otel_config
    from bobi.otel.resource import resource_attributes

    root, name = _otel_context(ctx)
    try:
        cfg = otel_config.resolve_config(root, signal)
    except otel_config.OtelUnconfigured as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)
    except otel_config.OtelMisconfigured as exc:
        click.echo(f"OTLP configuration is unusable: {exc}", err=True)
        raise SystemExit(1)
    if cfg.credential_withheld:
        click.echo(
            f"Withheld configured OTLP headers: the endpoint in use ({cfg.safe_url}) "
            "is not the origin run/.env's credential was configured for.",
            err=True,
        )
    return cfg, resource_attributes(root, name)


def _otel_send(export, cfg, spec, attrs) -> None:
    """Run one export, turning any failure into a loud, bounded diagnosis."""
    from bobi.otel.export import OtelExportError

    try:
        export(cfg, spec, attrs)
    except OtelExportError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)


@agent.group("otel")
def otel():
    """Record agent-authored telemetry to an OTLP endpoint.

    Emits one metric or one log record per invocation, stamped with the fleet
    identity an agent cannot resolve for itself. Configure the destination with
    OTEL_EXPORTER_OTLP_ENDPOINT; see docs/OTEL.md.

    Usage:
        bobi agent eng otel check
        bobi agent eng otel metric tickets.processed 42
        bobi agent eng otel log "reconciled the backlog"
    """
    pass


# `ignore_unknown_options` is what lets a leading-dash argument through the
# parser at all: without it `otel metric queue.delta -3` dies as "No such
# option '-3'" and the negative-value rules below are unreachable. A mistyped
# real option still fails loudly, as "unexpected extra arguments".
@otel.command("metric", context_settings={"ignore_unknown_options": True})
@click.argument("name")
@click.argument("value")
@click.option("--kind", type=click.Choice(["counter", "gauge", "histogram"]),
              default="counter", help="Instrument type (default: counter)")
@click.option("--temporality", type=click.Choice(["delta", "cumulative"]),
              default="delta",
              help="Aggregation temporality for counter/histogram")
@click.option("--attr", "attrs", multiple=True,
              help="Attribute as key=value. Repeatable. Always sent as a string.")
@click.option("--unit", default="", help="UCUM unit, e.g. s, By, 1")
@click.option("--desc", "description", default="", help="Human-readable description")
@click.pass_context
def otel_metric(ctx, name, value, kind, temporality, attrs, unit, description):
    """Record one measurement to the OTLP metrics endpoint.

    `1` is sent as an integer and `1.0` as a double - different wire types, so
    keep one series on one form. Attributes become time-series labels: keep
    them low-cardinality and never put an id or a secret in one.

    Usage:
        bobi agent eng otel metric tickets.processed 42
        bobi agent eng otel metric queue.depth 7 --kind gauge
        bobi agent eng otel metric task.seconds 12.5 --kind histogram --unit s
        bobi agent eng otel metric tickets.total 128 --temporality cumulative
    """
    from bobi.otel.export import MetricSpec, export_metric
    from bobi.otel.validate import validate_attrs, validate_metric_name, validate_value

    # Whether --temporality was TYPED, not what it holds: the option always
    # carries its `delta` default, so a value check would reject every gauge.
    if (kind == "gauge"
            and ctx.get_parameter_source("temporality")
            is click.core.ParameterSource.COMMANDLINE):
        # Gauge's only field is `data_points`; there is nowhere for a
        # temporality to go, so accepting one would silently drop it.
        raise click.UsageError(
            "--temporality does not apply to --kind gauge: a Gauge carries no "
            "aggregation temporality on the wire."
        )

    with _otel_usage_errors():
        validate_metric_name(name)
        parsed = validate_value(value)
        attributes = validate_attrs(attrs)
    if kind == "counter" and parsed < 0:
        raise click.UsageError(
            "--kind counter is monotonic; use --kind gauge for a value that "
            "can fall."
        )

    spec = MetricSpec(
        name=name,
        value=parsed,
        kind=kind,
        temporality=temporality,
        unit=unit,
        description=description,
        attributes=attributes,
    )
    cfg, resource = _otel_resolve(ctx, "metrics")
    _otel_send(export_metric, cfg, spec, resource)
    click.echo(f"Recorded {name}={parsed} ({kind}) to {cfg.safe_url}")


# Same reason as `metric`: an agent-authored body may legitimately start with
# a dash, and that must not read as an option.
@otel.command("log", context_settings={"ignore_unknown_options": True})
@click.argument("body", required=False)
@click.option("--severity", type=click.Choice(["debug", "info", "warn", "error", "fatal"]),
              default="info", help="Severity (default: info)")
@click.option("--attr", "attrs", multiple=True,
              help="Attribute as key=value. Repeatable. Always sent as a string.")
@click.pass_context
def otel_log(ctx, body, severity, attrs):
    """Record one log record to the OTLP logs endpoint.

    The body is sent verbatim and is never parsed as JSON. It leaves this box
    for a third party, so never put a secret or personal data in it. If <body>
    is omitted it is read from stdin, so a multi-line body needs no quoting.

    Usage:
        bobi agent eng otel log "reconciled 42 tickets"
        bobi agent eng otel log "upstream 502" --severity error
        printf 'line one\\nline two\\n' | bobi agent eng otel log
    """
    from bobi.otel.export import LogSpec, export_log
    from bobi.otel.validate import validate_attrs, validate_body

    if body is None:
        stdin = click.get_text_stream("stdin")
        if stdin.isatty():
            raise click.UsageError("Provide the log body as an argument or on stdin.")
        body = stdin.read()

    with _otel_usage_errors():
        body = validate_body(body)
        attributes = validate_attrs(attrs)
    spec = LogSpec(body=body, severity=severity, attributes=attributes)
    cfg, resource = _otel_resolve(ctx, "logs")
    _otel_send(export_log, cfg, spec, resource)
    click.echo(f"Recorded {severity} log to {cfg.safe_url}")


@otel.command("check")
@click.option("--send", is_flag=True,
              help="Also export one throwaway gauge through the real path")
@click.pass_context
def otel_check(ctx, send):
    """Report how OTLP export is configured on this box.

    Without --send this makes NO network call and says so: OTLP has no health
    endpoint, and a GET returns 405 from a Collector, which proves nothing.
    Header values are never printed - only their names - because this output
    lands in the agent's transcript and is rendered in the console.

    Exit 0 when configured, 1 otherwise.

    Usage:
        bobi agent eng otel check
        bobi agent eng otel check --send
    """
    from bobi.otel import config as otel_config
    from bobi.otel.export import MetricSpec, export_metric
    from bobi.otel.resource import resource_attributes

    root, name = _otel_context(ctx)

    try:
        import opentelemetry.proto  # noqa: F401
        click.echo("wire format:  opentelemetry.proto importable")
    except ImportError as exc:
        click.echo(f"wire format:  UNAVAILABLE ({exc})")

    configs: dict[str, otel_config.SignalConfig] = {}
    problems: list[str] = []
    for signal in ("metrics", "logs"):
        try:
            configs[signal] = otel_config.resolve_config(root, signal)
        except otel_config.OtelUnconfigured as exc:
            problems.append(str(exc))
        except otel_config.OtelMisconfigured as exc:
            problems.append(f"{signal}: {exc}")

    for signal in ("metrics", "logs"):
        cfg = configs.get(signal)
        click.echo(f"{signal + ' url:':<14}{cfg.safe_url if cfg else '(unresolved)'}")

    metrics_cfg = configs.get("metrics")
    if metrics_cfg is not None:
        # Names only. A value here would leak on the BENIGN path: an agent
        # debugging a 401 in good faith prints its own write token.
        names = ", ".join(f"{key}=<set>" for key in sorted(metrics_cfg.headers)) or "(none)"
        click.echo(f"headers:      {names}")
        if metrics_cfg.credential_withheld:
            click.echo(
                "headers:      WITHHELD - the endpoint in use is not the "
                "origin run/.env's credential was configured for"
            )
        click.echo(f"timeout:      {metrics_cfg.timeout_s:g}s")

    click.echo("resource attributes:")
    for key, value in sorted(resource_attributes(root, name).items()):
        click.echo(f"  {key}={value}")

    if problems:
        for problem in dict.fromkeys(problems):
            click.echo(problem, err=True)
        raise SystemExit(1)

    if not send:
        click.echo("no request sent (pass --send to export a throwaway gauge)")
        return

    assert metrics_cfg is not None
    spec = MetricSpec(
        name="bobi.otel.check",
        value=1,
        kind="gauge",
        unit="1",
        description="bobi otel check probe",
        attributes={},
    )
    _otel_send(export_metric, metrics_cfg, spec, resource_attributes(root, name))
    click.echo(f"sent bobi.otel.check to {metrics_cfg.safe_url}")


@main.command()
@click.argument("name", default="bobi")
def skill(name):
    """Print a skill guide to stdout.

    Agents can bootstrap themselves with: bobi skill

    Usage:
        bobi skill                # print the bobi usage guide
        bobi skill create-agent   # print the agent creation guide
        bobi skill linear-setup   # print the Linear setup guide
    """
    # In a source checkout, the repo-level skills/ directory is canonical.
    # Wheels bundle that same directory into bobi/skills/ via pyproject
    # force-include, so installed users do not need a repo checkout.
    repo_skills = _PACKAGE_DIR.parent / "skills"
    skills_dir = repo_skills if repo_skills.is_dir() else _PACKAGE_DIR / "skills"
    path = skills_dir / f"{name}.md"
    if not path.exists():
        available = [f.stem for f in skills_dir.glob("*.md")] if skills_dir.exists() else []
        click.echo(f"Skill '{name}' not found.", err=True)
        if available:
            click.echo(f"Available: {', '.join(sorted(available))}", err=True)
        raise SystemExit(1)
    click.echo(path.read_text())


def _show_events(tail: int, decisions_only: bool) -> None:
    """Show recent events and manager decisions as a unified timeline."""
    project_path = _detect_project_root()

    entries = []
    malformed = 0

    if not decisions_only:
        state_dir = paths.state_path(project_path)
        event_files = list(state_dir.glob("events-*.jsonl"))

        seen_events: set[tuple] = set()  # (seq, deployment_id) dedup

        for events_path in event_files:
            for line in events_path.read_text().strip().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue

                # Deduplicate by (seq, deployment_id) when both are present.
                seq = entry.get("seq")
                dep = entry.get("deployment_id")
                if seq is not None and dep is not None:
                    key = (seq, dep)
                    if key in seen_events:
                        continue
                    seen_events.add(key)

                data = entry.get("payload", entry.get("data", {}))
                detail = ""
                if entry.get("source") == "inbox" and isinstance(data, dict):
                    sender = data.get("sender", data.get("from", ""))
                    text = data.get("text", "")
                    if sender and text:
                        detail = f"{sender} → {text}"
                    elif text:
                        detail = text
                if not detail:
                    detail = data.get("text", "") or data.get("title", "") or data.get("run_key", "") if isinstance(data, dict) else ""
                if len(detail) > 80:
                    detail = detail[:80] + "..."
                entries.append((
                    entry["timestamp"],
                    f"  {entry['timestamp']}  {entry['source']:8s}  {entry['type']}"
                    + (f"\n    {detail}" if detail else ""),
                ))

    decisions_path = paths.state_path(project_path) / "decisions.jsonl"
    if decisions_path.exists():
        for line in decisions_path.read_text().strip().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            actions = entry.get("actions", [])
            types = ", ".join(a.get("type", "?") for a in actions)
            reason = ""
            if entry.get("reasoning"):
                reason = f"\n    {entry['reasoning'][:200].replace(chr(10), ' ')}"
            entries.append((
                entry["timestamp"],
                f"  {entry['timestamp']}  decision  {types}{reason}",
            ))

    if not entries:
        click.echo("No events yet.")
        return

    # Sort by instant, not by string: a log can span the aware-UTC timestamp
    # convention change, and pre-upgrade naive-LOCAL strings do not order
    # lexicographically against aware-UTC ones.
    from bobi.timeutil import epoch_seconds
    entries.sort(key=lambda e: epoch_seconds(e[0]))
    for _, text in entries[-tail:]:
        click.echo(text)

    if malformed:
        click.echo(f"\n  ⚠ {malformed} malformed line(s) skipped", err=True)


def _parse_event_publish_payload(json_payload: str | None) -> dict:
    if json_payload is None:
        stdin = click.get_text_stream("stdin")
        if stdin.isatty():
            raise click.UsageError("Provide payload with --json or stdin.")
        raw = stdin.read()
    else:
        raw = json_payload
    raw = raw.strip()
    if not raw:
        raise click.UsageError("Provide payload with --json or stdin.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"Payload must be valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise click.UsageError("Payload must be a JSON object.")
    return payload


def _validate_event_publish_topic(topic: str) -> str:
    source, sep, etype = topic.partition("/")
    if not sep or not source or not etype:
        raise click.UsageError(
            "Topic must use source/type form, e.g. alert/firing."
        )
    reserved_sources = {"github", "linear", "slack"}
    if source in reserved_sources:
        raise click.UsageError(
            "github, linear, and slack sources are reserved for webhooks."
        )
    global_prefixes = ("github:", "linear:", "slack:")
    if (
        topic.startswith(global_prefixes)
        or source.startswith(global_prefixes)
        or etype.startswith(global_prefixes)
    ):
        raise click.UsageError(
            "github:, linear:, and slack: topics are reserved for webhooks."
        )
    return topic


@main.group(invoke_without_command=True)
@click.option("--tail", default=20, help="Number of recent entries to show")
@click.option("--decisions-only", is_flag=True, help="Show only manager decisions")
@click.pass_context
def events(ctx, tail, decisions_only):
    """Show recent events and manager decisions as a unified timeline."""
    if ctx.invoked_subcommand is not None:
        return
    _show_events(tail, decisions_only)


@events.command("publish")
@click.argument("topic")
@click.option("--json", "json_payload", default=None,
              help="JSON object payload. If omitted, payload is read from stdin.")
def events_publish(topic, json_payload):
    """Publish a signed custom-topic event."""
    project_path = _detect_project_root()
    topic = _validate_event_publish_topic(topic)
    payload = _parse_event_publish_payload(json_payload)

    from bobi.events.publish import post_event
    if not post_event(topic, payload, project_path=project_path):
        click.echo(
            "Publish failed. Ensure the agent is started, bubble credentials "
            "are minted, and the event server accepted the signed publish.",
            err=True,
        )
        raise SystemExit(1)

    click.echo(f"Published {topic}")


@events.group("ingest-token")
def ingest_token():
    """Manage scoped ingest tokens for POST /webhooks/ingest/<topic>.

    An ingest token lets an external system that can only send static
    headers (alerting, CI, SaaS webhooks) publish plain JSON to one topic
    in this instance's bubble. The server stores only a hash; the token is
    shown once at creation.
    """


@ingest_token.command("create")
@click.argument("topic")
@click.option("--name", default=None, help="Optional label shown in list output.")
def ingest_token_create(topic, name):
    """Mint a token bound to TOPIC (source/type form, e.g. alert/firing).

    Topic rules are enforced server-side (validateIngestTopic in
    event-server/core/src/core.ts, the single source of truth); a rejection
    surfaces its reason verbatim.
    """
    project_path = _detect_project_root()

    from bobi.events.ingest_tokens import IngestTokenError, create_token
    try:
        minted = create_token(topic, name=name, project_path=project_path)
    except IngestTokenError as e:
        raise click.ClickException(str(e))

    click.echo(f"Ingest token for {minted.get('topic', topic)} "
               f"(id {minted.get('id', '?')}):")
    click.echo(f"\n  {minted.get('token', '')}\n")
    click.echo("Shown once - store it in the external system now. Send events with:")
    click.echo(f'  curl -H "Authorization: Bearer <token>" -d \'{{"title":"..."}}\' '
               f"<event-server>/webhooks/ingest/{topic}")


@ingest_token.command("list")
def ingest_token_list():
    """List this instance's ingest tokens (never shows the tokens themselves)."""
    project_path = _detect_project_root()

    from bobi.events.ingest_tokens import IngestTokenError, list_tokens
    try:
        tokens = list_tokens(project_path=project_path)
    except IngestTokenError as e:
        raise click.ClickException(str(e))

    if not tokens:
        click.echo("No ingest tokens.")
        return
    for t in tokens:
        label = f"  ({t['name']})" if t.get("name") else ""
        click.echo(f"{t.get('id', '?')}  {t.get('topic', '?')}{label}  "
                   f"created {t.get('created_at', '?')}")


@ingest_token.command("revoke")
@click.argument("token_id")
def ingest_token_revoke(token_id):
    """Revoke an ingest token by id. Takes effect immediately."""
    project_path = _detect_project_root()

    from bobi.events.ingest_tokens import IngestTokenError, revoke_token
    try:
        revoke_token(token_id, project_path=project_path)
    except IngestTokenError as e:
        raise click.ClickException(str(e))
    click.echo(f"Revoked {token_id}")



@transcript.command("index")
@click.option("--project", default=None, help="Filter to project (substring match on path)")
def transcript_index(project):
    """Index conversation JSONL files into searchable SQLite.

    Scans the Claude Code transcript roots — $CLAUDE_CONFIG_DIR/projects when
    set, then ~/.claude/projects — for */*.jsonl and indexes messages into a
    local SQLite database for fast searching.

    Usage:
        bobi agent eng transcript index                # index all projects
        bobi agent eng transcript index --project foo  # index only projects matching "foo"
    """
    from .history import index as do_index
    click.echo("Indexing conversations...")
    stats = do_index(project_filter=project)
    click.echo(f"  Scanned {stats['files_scanned']} files, {stats['files_with_new']} had new data")
    click.echo(f"  Indexed {stats['new_messages']} new messages")
    click.echo(f"  Total: {stats['total_conversations']} conversations, {stats['total_messages']} messages")


@transcript.command("search")
@click.argument("query")
@click.option("--limit", default=20, help="Max results")
@click.option("--project", default=None, help="Filter to project")
def transcript_search(query, limit, project):
    """Full-text search across indexed conversation history.

    Searches message content using SQLite FTS. Requires
    `bobi agent <name> transcript index` to have been run first.

    Usage:
        bobi agent eng transcript search "error handling"
        bobi agent eng transcript search "deploy" --project bobi --limit 5
    """
    from .history import search as do_search
    results = do_search(query, limit=limit, project=project)
    if not results:
        click.echo("No results. Run `bobi agent <name> transcript index` first.")
        return
    for r in results:
        branch = r.get("git_branch") or ""
        role = r.get("role") or r.get("type") or ""
        tool = f" [{r['tool_name']}]" if r.get("tool_name") else ""
        snippet = (r.get("snippet") or "")[:200].replace("\n", " ")
        click.echo(f"  {r['timestamp'][:19]}  {role:10s}{tool}  {branch}")
        click.echo(f"    {snippet}")
        click.echo()


@transcript.command("sessions")
@click.option("--limit", default=20)
@click.option("--project", default=None)
def transcript_sessions(limit, project):
    """List indexed conversations with metadata.

    Shows session ID, git branch, message count, and working directory for
    each indexed conversation.

    Usage:
        bobi agent eng transcript sessions
        bobi agent eng transcript sessions --limit 5 --project bobi
    """
    from .history import conversations
    convos = conversations(limit=limit, project=project)
    if not convos:
        click.echo("No conversations indexed. Run `bobi agent <name> transcript index` first.")
        return
    for c in convos:
        branch = c.get("git_branch") or ""
        click.echo(f"  {c['started_at'][:19]}  {c['session_id'][:8]}  {branch:20s}  {c['message_count']} msgs  {c.get('cwd', '')}")


@transcript.command("inspect")
@click.argument("session_id")
@click.option("--limit", default=50)
def transcript_inspect(session_id, limit):
    """Show messages from an indexed session.

    Accepts a full or partial session ID (prefix match). Use
    `bobi agent <name> transcript sessions` to find session IDs.

    Usage:
        bobi agent eng transcript inspect abc12345
        bobi agent eng transcript inspect abc12345 --limit 10
    """
    from .history import session_messages, conversations
    convos = conversations(limit=1000)
    match = [c for c in convos if c["session_id"].startswith(session_id)]
    if not match:
        click.echo(f"No session matching '{session_id}'")
        return
    full_id = match[0]["session_id"]
    msgs = session_messages(full_id)
    for m in msgs[:limit]:
        role = m.get("role") or m.get("type") or ""
        tool = f" [{m['tool_name']}]" if m.get("tool_name") else ""
        text = (m.get("content") or "")[:300].replace("\n", " ")
        click.echo(f"  {role:10s}{tool}  {text}")


@main.group()
def workflows():
    """Workflow engine — manage YAML-based DAG workflows."""
    pass


@workflows.command("list")
def workflow_list():
    """List available workflow definitions from the installed pack.

    Usage:
        bobi agent eng workflows list
    """
    from .workflow.triggers import WorkflowDispatcher

    project_path = _try_detect_project_root()
    dispatcher = WorkflowDispatcher()
    if project_path is not None:
        dispatcher.load_all_workflows(project_path)
    click.echo(dispatcher.format_workflow_menu())


@workflows.command("status")
def workflow_status():
    """Show active and recent workflow runs.

    Displays up to 20 recent runs with their status, trigger issue,
    current step, and start time.

    Usage:
        bobi agent eng workflows status
    """
    _detect_project_root()
    from .workflow.state import WorkflowRun
    runs = WorkflowRun.list_runs()
    if not runs:
        click.echo("No workflow runs found.")
        return
    for run in runs[:20]:
        # The run_key FIELD is authoritative (#1048); the trigger-event copy
        # is display fallback for records written before the field existed.
        event_data = run.trigger_event.get("data", {})
        issue = run.run_key or event_data.get("run_key", "?")
        suffix = ""
        if run.suspended_at_step >= 0:
            suffix = f"  step={run.suspended_at_step}"
        if run.status == "waiting" and run.await_event:
            suffix += f"  awaiting={run.await_event}"
        click.echo(f"  {run.run_id}  {run.workflow_name:20s} {run.status:10s} "
                  f"issue={issue}  {run.started_at[:19]}{suffix}")


@workflows.command("resume")
@click.argument("run_id")
@click.option("--verdict", type=click.Choice(GATE_VERDICTS), default=None,
              help="The gate's answer, carried to the workflow as "
                   "${{event.verdict}}.")
@click.option("--reply", default="",
              help="The human's own words, carried as ${{event.reply}}.")
@click.option("--timeout", default=3600, help="Max execution time in seconds")
def workflow_resume(run_id, verdict, reply, timeout):
    """Resume a suspended workflow run.

    Picks up from the step after the await that suspended it.

    ``--verdict`` and ``--reply`` are the answer the gate was waiting for.
    They land as the ``event`` scope, so a route step placed after the await
    reads them as ``${{event.verdict}}`` / ``${{event.reply}}`` and sends the
    run down the branch the human chose.

    Both are optional and the scope is always set, so a resume with no verdict
    resolves ``${{event.verdict}}`` to the empty string rather than to a
    missing scope. That is not an approval, and a workflow's route is written
    so the non-approving branch is the safe one. An unknown verdict is
    refused here rather than resolved to something a route might advance on.

    Usage:
        bobi agent eng workflows resume abc123 --verdict approve
    """
    # `default=None` on the Choice, because click validates a default too and
    # "" is not one of the verdicts. Absent and empty mean the same thing here.
    verdict = verdict or ""
    _detect_project_root()
    from .workflow.state import WorkflowRun
    from .workflow.triggers import WorkflowDispatcher
    from .workflow.orchestrator import resume_workflow

    try:
        run = WorkflowRun.load(run_id)
    except (FileNotFoundError, KeyError):
        click.echo(f"No run '{run_id}'.", err=True)
        sys.exit(1)

    if run.status != "waiting":
        click.echo(f"Run {run_id} is '{run.status}', not 'waiting'.", err=True)
        sys.exit(1)

    # Everything that can refuse resolves BEFORE the claim. `claim()` renames
    # <id>.json to <id>.resuming.json and nothing renames it back, so exiting
    # after claiming leaves the run findable-forever and resumable-never
    # (state.py's D071). The workflow lookup used to sit on the wrong side of
    # this line.
    dispatcher = WorkflowDispatcher()
    dispatcher.load_all_workflows()
    wf = dispatcher.find_workflow(run.workflow_name)
    if not wf:
        click.echo(f"Workflow '{run.workflow_name}' not found.", err=True)
        sys.exit(1)

    # The same refusal the console makes, for the same reason: without a route
    # on the verdict, a rejection would run the next step - advancing the work
    # the human just refused. Checked here too because this command is driven
    # by hand as well as by the spawn the console detaches.
    from .workflow.schema import GATE_VERDICT_REJECT, reads_gate_verdict
    if verdict == GATE_VERDICT_REJECT and not reads_gate_verdict(
            wf, run.suspended_at_step):
        click.echo(
            f"Workflow '{run.workflow_name}' has no route on the gate's "
            f"verdict at step {run.suspended_at_step}, so a rejection cannot "
            f"be honoured. Resuming would run that step.", err=True)
        sys.exit(1)

    # Claim before resuming. The event-driven path has always done this; this
    # command never did, so two concurrent resumes of the same run both ran it.
    # That is now reachable from the web app, which spawns this command — and
    # the claim belongs in the process that does the work, not in the caller:
    # a claim held by a process that then fails to start strands the run.
    if not run.claim():
        click.echo(f"Run {run_id} was claimed by another process.", err=True)
        sys.exit(1)

    click.echo(f"Resuming {run.workflow_name} for {run.run_key} "
               f"from step {run.suspended_at_step}"
               + (f" with verdict '{verdict}'" if verdict else "") + "...")
    # Always an event, even with no verdict: the scope then exists and holds
    # an empty verdict, which a route reads as "not an approval". Passing no
    # event at all would leave the scope missing, which resolves the same way
    # but only after a log warning about an unknown scope.
    event = {"data": {"verdict": verdict, "reply": reply}}
    if not resume_workflow(run, wf, event=event, timeout=timeout):
        click.echo("Workflow failed.", err=True)
        sys.exit(1)
    # resume_workflow returns True for two different endings: the run finished,
    # or it parked on a LATER await step. The second is dormant, not done - the
    # run's own ledger entry is back to "waiting" (#1048: one run, one record)
    # - so report it as such instead of claiming completion. A rejected gate
    # that reworks and re-gates ends here every cycle.
    if run.status == "waiting":
        click.echo("Workflow suspended again on a later await step. "
                   "Run `workflows status` for the waiting run.")
    else:
        click.echo("Workflow completed.")


@workflows.command("validate")
@click.argument("path", type=click.Path(exists=True))
def workflow_validate(path):
    """Validate a workflow YAML file.

    Parses the YAML, checks the DAG structure, reports variable scopes used,
    and prints the topological execution order if valid.

    Usage:
        bobi agent eng workflows validate workflows/deploy.yaml
        bobi agent eng workflows validate package/workflows/deploy.yaml
    """
    import re
    from .workflow.schema import load_workflow
    try:
        wf = load_workflow(Path(path))
        step_names = [s.name for s in wf.steps]
        click.echo(f"Valid: {wf.name} ({len(wf.steps)} steps)")
        if wf.trigger:
            click.echo(f"Trigger: {wf.trigger.strip()}")
        click.echo(f"Steps: {' -> '.join(step_names)}")

        raw = Path(path).read_text()
        refs = set(re.findall(r'\$\{\{(\w+)\.', raw))
        if refs:
            click.echo(f"Variable scopes: {', '.join(sorted(refs))}")

    except Exception as e:
        click.echo(f"Invalid: {e}", err=True)
        raise SystemExit(1)




main.add_command(workflows)


@main.group()
def roles():
    """Agent roles — list available role prompts."""
    pass


@roles.command("list")
def role_list():
    """List available agent roles.

    Scans the selected Bobi Agent's installed package.

    Usage:
        bobi agent eng roles list
    """
    from .prompts.resolver import discover_roles, format_role_list

    project_path = _detect_project_root()
    roles = discover_roles(project_path)
    click.echo(format_role_list(roles))


main.add_command(roles)


@main.group()
def monitors():
    """Background monitoring tasks — scheduled polling to fill webhook gaps."""
    pass


def _slugify(text: str) -> str:
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "monitor"


@monitors.command("list")
def monitor_list():
    """Show the merged view of monitors across all tiers, with source.

    Usage:
        bobi agent eng monitors list
    """
    from .monitors.registry import MonitorRegistry

    project_path = _detect_project_root()
    registry = MonitorRegistry.load(project_path=project_path)
    monitors = sorted(registry.all_monitors(), key=lambda m: (m.name, m.project))
    if not monitors:
        click.echo("No monitors found.")
        return

    for m in monitors:
        if m.source == "default":
            tier = "default"
        elif m.source == "user":
            tier = "user"
        else:
            tier = f"project:{Path(m.source).name}"
        status = "active" if m.enabled else "paused"
        scope = Path(m.project).name if m.project else "all projects"
        runner = m.check or "manager"
        suffix = _script_cache_summary(m) if m.check == "script_cache" else ""
        click.echo(f"  {m.name:22s} {tier:16s} {m.interval:>5s}  {status:7s} "
                   f"{scope:16s} {m.event:30s} [{runner}]{suffix}")


def _script_cache_summary(monitor) -> str:
    """A compact `mode + cumulative savings` suffix for a script_cache monitor,
    read from its trusted-state sidecar (#327 observability)."""
    try:
        from .monitors.script_cache_checks import _load_trusted_state
        st = _load_trusted_state(monitor.name)
        if not st:
            return "  (no runs yet)"
        cached = st.get("cached_runs", 0)
        fallback = st.get("fallback_runs", 0)
        spent = st.get("total_agent_cost_usd", 0.0)
        avg = (spent / fallback) if fallback else 0.0
        saved = cached * avg  # cached ticks would each have cost ~one agent run
        return (f"  mode={st.get('last_mode', '?')} cached={cached} "
                f"agent={fallback} spent=${spent:.4f} saved~${saved:.4f}")
    except Exception:
        return ""


@monitors.command("add")
@click.argument("name")
@click.option("--interval", default=None, help="How often to run (e.g. 5m, 15m, 1h). Mutually exclusive with --at.")
@click.option("--at", "at_times", multiple=True, help="Wall-clock time(s) HH:MM (repeatable). Schedules instead of --interval.")
@click.option("--tz", default="", help="IANA timezone for --at (e.g. America/Los_Angeles); defaults to host local.")
@click.option("--days", default="", help="Weekday(s) to gate --at to (e.g. 'sun' or 'mon,wed,fri'). Requires --at.")
@click.option("--notify", is_flag=True, help="Fire the event on every scheduled run (a scheduled nudge, not a condition).")
@click.option("--description", default="", help="What the monitor checks (interpreted by the manager)")
@click.option("--event", default=None, help="Synthetic event type to inject (default monitor/<name>)")
@click.option("--check", default="", help="Native check runner (pr_conflicts, stale_prs)")
@click.option("--url", default=None, help="URL the description references (e.g. deploy health)")
def monitor_add(name, interval, at_times, tz, days, notify, description, event, check, url):
    """Add a monitor to the selected Bobi Agent.

    Usage:
        bobi agent eng monitors add "PR conflict check" --interval 15m \\
            --description "Check open PRs for merge conflicts"
        bobi agent eng monitors add deploy-health --interval 5m \\
            --url https://example.com
        bobi agent eng monitors add weekly-prep-doc \\
            --at 21:00 --days sun --tz America/Los_Angeles --notify \\
            --event monitor/prep.weekly_due \\
            --description "Generate my prep doc for the upcoming week"
    """
    import re as _re

    from .monitors.schema import Monitor, parse_at, parse_days, parse_interval
    from .monitors.registry import MonitorRegistry
    from .runtime_guard import with_mutable_runtime_package

    project_path = _detect_project_root()

    at_list = list(at_times)
    day_list = [d for d in _re.split(r"[,\s]+", days.strip()) if d]

    if at_list and interval is not None:
        raise click.ClickException("--interval and --at are mutually exclusive")
    if day_list and not at_list:
        raise click.ClickException("--days only applies to --at scheduling (add --at HH:MM)")

    slug = _slugify(name)
    try:
        if at_list:
            parse_at(at_list)
            parse_days(day_list)
        else:
            parse_interval(interval or "15m")
    except ValueError as e:
        raise click.ClickException(str(e))

    extra = {}
    if url:
        extra["url"] = url

    m = Monitor(
        name=slug,
        description=description,
        interval=interval or "15m",
        at=at_list,
        tz=tz,
        days=day_list,
        notify=notify,
        event=event or f"monitor/{slug}",
        check=check,
        extra=extra,
    )

    with with_mutable_runtime_package(project_path):
        MonitorRegistry.add_project(m, project_path)
    click.echo(f"Added monitor '{slug}' to {paths.package_dir(project_path) / 'monitors.yaml'}")
    if at_list:
        schedule = f"at={','.join(at_list)}"
        if day_list:
            schedule += f" days={','.join(day_list)}"
        if tz:
            schedule += f" tz={tz}"
    else:
        schedule = f"interval={interval or '15m'}"
    click.echo(f"  {schedule} event={m.event} "
               f"{'notify' if notify else (check or 'manager-interpreted')}")


@monitors.command("pause")
@click.argument("name")
def monitor_pause(name):
    """Disable a monitor (writes enabled: false).

    Usage:
        bobi agent eng monitors pause stale-pr-check
    """
    from .monitors.registry import MonitorRegistry
    from .runtime_guard import with_mutable_runtime_package

    project_path = _detect_project_root()
    with with_mutable_runtime_package(project_path):
        paused = MonitorRegistry.pause(name, project_path)
    if paused:
        where = str(paths.package_dir(project_path) / "monitors.yaml")
        click.echo(f"Paused monitor '{name}' (enabled: false in {where})")
    else:
        click.echo(f"No monitor named '{name}' found.", err=True)
        raise SystemExit(1)


@monitors.command("remove")
@click.argument("name")
def monitor_remove(name):
    """Remove a monitor from the selected Bobi Agent.

    Built-in defaults can't be deleted — pause them instead.

    Usage:
        bobi agent eng monitors remove deploy-health
    """
    from .monitors.registry import MonitorRegistry
    from .runtime_guard import with_mutable_runtime_package

    project_path = _detect_project_root()
    with with_mutable_runtime_package(project_path):
        result = MonitorRegistry.remove(name, project_path)
    if result == "removed":
        click.echo(f"Removed monitor '{name}'.")
    elif result == "default-only":
        click.echo(f"'{name}' is a built-in default and can't be removed. "
                   f"Use `bobi agent <agent> monitors pause {name}` to disable it.", err=True)
        raise SystemExit(1)
    else:
        click.echo(f"No monitor named '{name}' found in a writable tier.", err=True)
        raise SystemExit(1)


def _find_monitor(name: str, project_path):
    """Resolve a monitor by name from the effective registry, or None."""
    from .monitors.registry import MonitorRegistry
    registry = MonitorRegistry.load(project_path=project_path)
    for m in registry.effective_monitors():
        if m.name == name:
            return m
    return None


@monitors.command("recache")
@click.argument("name")
def monitor_recache(name):
    """Invalidate a script_cache monitor's cached script (forces regeneration).

    Usage:
        bobi agent eng monitors recache unread-emails
    """
    from .monitors.script_cache_checks import recache

    project_path = _detect_project_root()
    m = _find_monitor(name, project_path)
    if m is None:
        click.echo(f"No monitor named '{name}' found.", err=True)
        raise SystemExit(1)
    if m.check != "script_cache":
        click.echo(f"'{name}' is not a script_cache monitor (check={m.check}).", err=True)
        raise SystemExit(1)
    recache(m)
    click.echo(f"Invalidated cached script for '{name}' — next tick regenerates.")


@monitors.command("approve-script")
@click.argument("name")
def monitor_approve_script(name):
    """Promote a script_cache monitor's pending script to active (review mode).

    Usage:
        bobi agent eng monitors approve-script unread-emails
    """
    from .monitors.script_cache_checks import approve_pending

    project_path = _detect_project_root()
    m = _find_monitor(name, project_path)
    if m is None:
        click.echo(f"No monitor named '{name}' found.", err=True)
        raise SystemExit(1)
    if m.check != "script_cache":
        click.echo(f"'{name}' is not a script_cache monitor (check={m.check}).", err=True)
        raise SystemExit(1)
    if approve_pending(m):
        click.echo(f"Approved + pinned the pending script for '{name}'.")
    else:
        click.echo(f"No valid pending script to approve for '{name}'.", err=True)
        raise SystemExit(1)


@monitors.command("gate", hidden=True)
@click.option("--request", "request_path", required=True,
              help="Path to the gate request JSON written by the scheduler.")
def monitor_gate(request_path):
    """Internal: judge new monitor items against a relevance criterion (#630).

    Scheduler plumbing, launched out-of-band by _default_spawn_gate. Reads
    {"criterion": ..., "name": ..., "items": [{"key": ..., "data": ...}]}
    from the request file, runs the cheap-model gate agent, and prints the
    verdict as a single JSON line: {"success": ..., "relevant": [...]}.
    """
    from .subagent import run_gate_blocking

    project_path = _detect_project_root()
    try:
        request = json.loads(Path(request_path).read_text())
    except (OSError, json.JSONDecodeError, ValueError) as e:
        click.echo(f"Bad gate request file: {e}", err=True)
        raise SystemExit(1)

    criterion = str(request.get("criterion", ""))
    items = request.get("items") or []
    if not criterion or not isinstance(items, list) or not items:
        click.echo("Gate request needs a criterion and a non-empty items list.",
                   err=True)
        raise SystemExit(1)

    result = run_gate_blocking(
        criterion, items, cwd=str(project_path),
        name=request.get("name") or None,
    )
    click.echo(json.dumps({"success": result.success,
                           "relevant": result.relevant}))
    if not result.success:
        click.echo(f"Gate failed: {result.error}", err=True)
        raise SystemExit(1)


@monitors.command("curator", hidden=True)
@click.option("--request", "request_path", required=True,
              help="Path to the rendered sleep-cycle task written by the scheduler.")
def monitor_curator(request_path):
    """Internal: distill the transcript delta into long_term_memory.md (#456).

    Scheduler plumbing, launched out-of-band by _default_spawn_sleep_cycle.
    Reads the full rendered task (prompt + current memory + transcript delta)
    from the request file, runs the sleep-cycle agent, and prints its
    summary as a single JSON line: {"success": ..., "updated": ...}.
    """
    from .subagent import run_curator_blocking

    project_path = _detect_project_root()
    try:
        task = Path(request_path).read_text()
    except OSError as e:
        click.echo(f"Bad sleep-cycle request file: {e}", err=True)
        raise SystemExit(1)
    if not task.strip():
        click.echo("Sleep-cycle request file is empty.", err=True)
        raise SystemExit(1)

    summary, error = run_curator_blocking(task, cwd=str(project_path))
    if summary is None:
        click.echo(json.dumps({"success": False, "summary": error}))
        click.echo(f"Curator failed: {error}", err=True)
        raise SystemExit(1)
    click.echo(json.dumps(summary))


main.add_command(monitors)


# ---------------------------------------------------------------------------
# event-server group
# ---------------------------------------------------------------------------


@main.group("event-server")
def event_server_cmd():
    """Manage the local event server daemon."""
    pass


@event_server_cmd.command("start")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground")
@click.option("--port", default=None, type=int, help="Override webhook port")
def event_server_start(foreground, port):
    """Start the local event server."""
    project_path = _detect_project_root()

    from bobi.events.server import (
        NodeRuntimePrerequisiteError,
        PackagedEventServerArtifactError,
        ensure_running,
        resolve_local_port,
    )
    # An explicit --port wins; everything else resolves through the shared
    # definition, so doctor probes the same port this starts.
    es_port = port if port is not None else resolve_local_port(project_path)
    try:
        result = ensure_running(es_port, project_path=project_path)
    except (
        NodeRuntimePrerequisiteError,
        PackagedEventServerArtifactError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    if result == "skipped":
        click.echo("Remote event_server_url configured — local server not needed.", err=True)
        return

    if foreground:
        click.echo(f"Event server running on port {es_port} (foreground)")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        return

    click.echo(f"Event server running on port {es_port}")
    click.echo(f"  GitHub:  http://localhost:{es_port}/webhooks/github")
    click.echo(f"  Linear:  http://localhost:{es_port}/webhooks/linear")
    click.echo(f"  Slack:   http://localhost:{es_port}/webhooks/slack")


@event_server_cmd.command("stop")
def event_server_stop():
    """Stop the local event server."""
    import signal

    from bobi import launch_stamp
    from bobi.events.server import local_port_file

    project_path = _detect_project_root()
    pid_file = paths.event_server_pid_path(project_path)
    port_file = local_port_file(project_path)
    if not pid_file.exists():
        click.echo("Event server is not running")
        port_file.unlink(missing_ok=True)
        return
    # The pid file is written by another process and a crash can truncate it
    # mid-write, so every read here is defensive: an unparseable pid used to
    # raise out of the command, print a traceback, and leave the stale files
    # behind — which made every subsequent `stop` fail exactly the same way,
    # with no way out but deleting the files by hand. Mirrors the manager stop
    # path ("Invalid PID file — cleaning up.").
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        click.echo("Invalid event-server PID file — cleaning up.")
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"Event server stopped (pid {pid})")
        except ProcessLookupError:
            click.echo("Event server was not running (stale PID file)")
        except PermissionError:
            # The pid was reused by a process we do not own; signalling it
            # would be wrong even if we could. Drop our stale files and say so.
            click.echo(f"Not permitted to signal pid {pid} — it is not ours. "
                       "Clearing the stale PID file.", err=True)
        except OSError as e:
            click.echo(f"Could not signal pid {pid}: {e}", err=True)
    pid_file.unlink(missing_ok=True)
    port_file.unlink(missing_ok=True)
    launch_stamp.clear_launch(project_path, launch_stamp.EVENT_SERVER)


@event_server_cmd.command("restart")
@click.option("--port", default=None, type=int, help="Override webhook port")
@click.pass_context
def event_server_restart(ctx, port):
    """Restart the local event server."""
    ctx.invoke(event_server_stop)
    import time as _time
    _time.sleep(1)
    ctx.invoke(event_server_start, foreground=False, port=port)


@event_server_cmd.command("status")
def event_server_status():
    """Show event server status."""
    from bobi.events.server import health, local_port_from_url, resolve_local_port
    project_path = _detect_project_root()
    try:
        from .config import Config
        configured = Config.load(project_path).event_server_url
    except Exception:
        configured = ""
    if configured and local_port_from_url(configured) is None:
        click.echo(f"Event server: remote ({configured})")
        return

    es_port = resolve_local_port(project_path)
    data = health(f"http://localhost:{es_port}")
    if data:
        click.echo(f"Event server: running on port {es_port}")
        click.echo(f"  Mode: {data.get('mode', 'unknown')}")
        click.echo(f"  Deployments: {data.get('deployments', 0)}")
    else:
        click.echo(f"Event server: not running (port {es_port})")


main.add_command(event_server_cmd)


@subagents.command("launch")
@click.option("--workflow", "-w", required=True, help="Workflow to run (e.g. issue-lifecycle, adhoc)")
@click.option("--role", required=True, help="Agent role (see 'bobi agent <name> roles list')")
@click.option("--id", "run_key", default=None,
              help="Explicit run key for correlation (e.g. an issue number). "
                   "Relaunching the same key resumes that run. Default: a key "
                   "derived from the launch itself - workflow, role, model, "
                   "effort, task text - so an identical launch collides with "
                   "the run already in flight instead of starting a second "
                   "one.")
@click.option("--id-random", "random_key", is_flag=True,
              help="Mint a random run key instead of deriving one from the "
                   "launch. Without it, an un-keyed launch derives its key "
                   "from the workflow, role, model, effort and task text, so "
                   "relaunching the same one while the first run is still "
                   "going is refused as a duplicate. Use this to fan out N "
                   "copies of an IDENTICAL launch on purpose. Cannot be "
                   "combined with --id.")
@click.option("--task", default=None, help="Task description / context for the agent")
@click.option("--timeout", default=3600, type=int, help="Timeout in seconds")
@click.option("--wait", is_flag=True,
              help="Block until the launched agent completes. Requires "
                   "'-w adhoc' — a multi-step workflow launch returns as soon "
                   "as it is dispatched, so there is nothing to join.")
@click.option("--as-check", "as_check", is_flag=True,
              help="Run the task as a short-lived monitoring check")
@click.option("--post-event", "post_event", default=None,
              help="Post this event type on completion (for --as-check)")
@click.option("--requested-by", "requested_by", default=None,
              help='JSON identity of requester, e.g. \'{"from":"Alice","channel":"C1"}\'')
@click.option("--non-interactive", "non_interactive", is_flag=True,
              help="Run without manager — agent makes all decisions autonomously")
@click.option("--persistent", is_flag=True,
              help="Keep the agent alive after initial task, accepting inbox messages")
@click.option("--subscribe", multiple=True,
              help="Subscribe to event topics (e.g. moda-labs/bobi-agent, slack:T123)")
@click.option("--model", default="",
              help="Model override for this launch (provider-native, e.g. haiku, "
                   "opus, or a full model ID). Wins over step and role config.")
@click.option("--effort", default="",
              help="Reasoning-effort override for this launch (provider-native, "
                   "e.g. low, medium, high, xhigh). Wins over step and role "
                   "config.")
@click.option("--fresh", is_flag=True,
              help="Start a new transcript instead of resuming this run key's "
                   "saved session. The run keeps its name (and so its worktree "
                   "branch and registry entry) but does not inherit the "
                   "previous session's context or spent turn budget. Use it on "
                   "every RE-dispatch of a worker that re-orients from durable "
                   "state — a committed checklist, the branch's commits — "
                   "since re-running the same --task otherwise resumes the "
                   "dead session. Implied when the run key is derived (no "
                   "--id), where there is no run the caller meant to continue.")
def subagents_launch(workflow, role, run_key, random_key, task, timeout, wait,
                     as_check, post_event, requested_by, non_interactive,
                     persistent, subscribe, model, effort, fresh):
    """Launch a sub-agent with a workflow and role.

    Every sub-agent runs a workflow with a role. Use 'adhoc' for open-ended tasks.
    Use 'bobi agent <name> roles list' to see available roles.

    Examples:
        bobi agent eng subagents launch -w issue-lifecycle --role engineer --id 42 --task "Fix moda-labs/bobi-agent#42"
        bobi agent eng subagents launch -w adhoc --role engineer --task "Why is CI failing?"
    """
    if subscribe:
        persistent = True
    _dispatch_agent(task=task, workflow=workflow, role=role, run_key=run_key,
                    random_key=random_key,
                    timeout=timeout, wait=wait, as_check=as_check,
                    post_event=post_event, requested_by=requested_by,
                    interactive=not non_interactive,
                    persistent=persistent,
                    subscribe=list(subscribe),
                    model=model, effort=effort, fresh=fresh)


def _parse_requested_by(requested_by: str | None) -> dict:
    """Decode `--requested-by`, exiting with its own message on bad input.

    Both dispatch paths need this: `_dispatch_agent` hands the RAW string to
    `_run_agent_wait` rather than a decoded dict, so the wait path parses it
    for itself.
    """
    if not requested_by:
        return {}
    try:
        parsed = json.loads(requested_by)
    except json.JSONDecodeError:
        click.echo("--requested-by must be valid JSON", err=True)
        raise SystemExit(1)
    if not isinstance(parsed, dict):
        click.echo("--requested-by must be a JSON object", err=True)
        raise SystemExit(1)
    return parsed


def _dispatch_agent(*, task, workflow, role, run_key=None, random_key=False,
                    timeout, wait,
                    as_check=False, post_event=None, requested_by=None,
                    interactive=True, persistent=False, subscribe=None,
                    model="", effort="", fresh=False):
    """Dispatch logic for the agent command."""
    if not workflow:
        click.echo("--workflow is required. Use 'adhoc' for open-ended tasks.", err=True)
        raise SystemExit(1)

    if not task:
        task = f"Run workflow {workflow}"

    # Raises a clean UsageError when run outside an installation.
    project_path = _detect_project_root()
    cwd = str(project_path)

    if as_check and wait:
        click.echo("--as-check cannot be combined with --wait", err=True)
        raise SystemExit(1)
    if as_check and (model or effort):
        # The check harness resolves roles.monitor.* itself; silently
        # ignoring an explicit override would misreport what the check ran at.
        click.echo("--model/--effort are not supported with --as-check "
                   "(configure roles.monitor instead)", err=True)
        raise SystemExit(1)
    if post_event and not as_check:
        click.echo("--post-event requires --as-check", err=True)
        raise SystemExit(1)
    if run_key and random_key:
        click.echo("--id-random cannot be combined with --id (an explicit run "
                   "key already opts out of task-derived dedup)", err=True)
        raise SystemExit(1)

    if as_check:
        _run_check(cwd=cwd, task=task, timeout=timeout, post_event=post_event)
        return

    # --- Validate role ---
    from .prompts.resolver import validate_role, discover_roles
    if not validate_role(role, Path(cwd)):
        available = discover_roles(Path(cwd))
        names = ", ".join(r["name"] for r in available) if available else "(none)"
        click.echo(f"Unknown role '{role}'. Available: {names}", err=True)
        raise SystemExit(1)

    if wait:
        with _launch_refusal_is_readable(project_path):
            _run_agent_wait(cwd=cwd, task=task, workflow=workflow, role=role,
                            run_key=run_key, random_key=random_key,
                            timeout=timeout,
                            requested_by=requested_by, interactive=interactive,
                            persistent=persistent, subscribe=subscribe or [],
                            model=model, effort=effort, fresh=fresh)
        return

    requester = _parse_requested_by(requested_by)

    from .subagent import launch_agent
    with _launch_refusal_is_readable(project_path):
        session_name = launch_agent(
            task=task, cwd=cwd, workflow_name=workflow,
            timeout=timeout, requested_by=requester,
            interactive=interactive,
            role=role,
            persistent=persistent,
            subscribe=subscribe or [],
            run_key=run_key,
            random_key=random_key,
            model=model,
            effort=effort,
            fresh=fresh,
        )
    click.echo(f"Agent started: {session_name}")


@contextmanager
def _launch_refusal_is_readable(project_path: Path):
    """Surface a refused launch as one line on stderr, never a traceback.

    The highest-stakes surface in either launch guard, because the reader is an
    LLM. A raw traceback reads as a transient crash, and the natural recovery
    is to retry with a fresh run key or ``-w adhoc`` - turning one refusal into
    the launch storm the guards exist to stop. Both refusals land here so that
    property holds once rather than per call site:

    - ``LaunchBlockedError`` (lineage, #849) carries its own message, built in
      ``bobi/launch_lineage.py``, which says the block is deterministic and
      names each retry vector.
    - ``DuplicateRunError`` (derived run key, #850) needs the remediation
      spelled out here, where the agent name is resolvable.
    """
    from .launch_lineage import LaunchBlockedError
    from .sdk import ACTIVE_STATUSES
    from .subagent import DuplicateRunError
    try:
        yield
    except LaunchBlockedError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1) from None
    except DuplicateRunError as exc:
        # Render the real agent name: an LLM pastes a `<name>` placeholder
        # verbatim and the remediation fails.
        agent_name = paths.agent_name_for_root(project_path)
        click.echo(f"Launch refused: {exc}", err=True)
        click.echo(f"  Watch it:  bobi agent {agent_name} subagents show "
                   f"{exc.session_name}", err=True)
        # Only offer the cancel when it would do something. `cancel_agent`
        # ignores anything outside ACTIVE_STATUSES, so printing it for a
        # suspended run sends the reader to a command that reports "no running
        # sub-agent" and teaches them the refusal is bogus. The exception's own
        # message carries the remedy that fits that case.
        if exc.status in ACTIVE_STATUSES:
            click.echo(f"  Cancel it: bobi agent {agent_name} subagents cancel "
                       f"{exc.session_name}", err=True)
        raise SystemExit(1) from None



def _run_agent_wait(*, cwd: str, task: str, workflow: str, role: str,
                    run_key: str | None, timeout: int, requested_by,
                    interactive: bool, persistent: bool, subscribe: list[str],
                    model: str = "", effort: str = "",
                    fresh: bool = False, random_key: bool = False) -> None:
    """Run a real agent synchronously and print its final text."""
    if workflow != "adhoc":
        # Deliberate limit, not an oversight: --wait is the fan-out-and-block
        # delegation idiom (#845; see the engineer role's "Parallel Work"
        # section), and an ad-hoc unit is its unit of work. Widening --wait to
        # multi-step workflows is a contract change to make on purpose, not a
        # side effect of the executor consolidation (#1057).
        click.echo(
            f"--wait requires '-w adhoc' (got '{workflow}'). To fan out and "
            f"join, launch adhoc units in the background and 'wait' for them "
            f"in a single shell command.",
            err=True,
        )
        raise SystemExit(1)
    if persistent:
        click.echo("--wait cannot be used with --persistent", err=True)
        raise SystemExit(1)

    requester = _parse_requested_by(requested_by)

    # One launch path (#1057): launch_agent owns the derivation, preflights,
    # admission and lineage for EVERY run, and wait=True just runs the
    # workflow in this process instead of detaching it. The ad-hoc task is
    # literally a one-step workflow execution, with the same ledger entry,
    # checkpoints and period dedupe every other run gets.
    from .subagent import launch_agent
    result = launch_agent(
        task=task, cwd=cwd, workflow_name=workflow, timeout=timeout,
        requested_by=requester, interactive=interactive, role=role,
        subscribe=subscribe, run_key=run_key, random_key=random_key,
        model=model, effort=effort, fresh=fresh, wait=True,
    )
    if result.final_text:
        click.echo(result.final_text)
    if result.error_kind == "suspended":
        # A pack may override `adhoc` with a gated workflow. Parked is
        # dormant, not done - exiting silently as a success would report
        # work finished that never ran past the gate.
        click.echo(
            "Run suspended at an approval gate; it resumes when its event "
            "arrives (see the console runs view).", err=True,
        )
    if not result.success:
        if result.error:
            click.echo(f"Agent failed: {result.error}", err=True)
        raise SystemExit(1)


def _run_check(cwd: str, task: str, timeout: int, post_event: str | None) -> None:
    """Run a non-interactive check, print its verdict, optionally post an event.

    Used by `bobi spawn --non-interactive` and by the monitor scheduler,
    which launches this as a short-lived out-of-band process so the manager's
    context stays clean — the manager only ever sees the resulting event.
    """
    from .subagent import run_check_blocking

    # Cap the check's runtime well below an engineer phase — checks are quick.
    from .subagent import CHECK_TIMEOUT
    check_timeout = min(timeout, CHECK_TIMEOUT) if timeout else CHECK_TIMEOUT

    result = run_check_blocking(description=task, cwd=cwd, timeout=check_timeout)

    verdict = {
        "success": result.success,
        "finding": result.finding,
        "summary": result.summary,
        "details": result.details,
        # The session this check ran under — the monitor scheduler records it
        # on the run so the run's row can open a transcript.
        "session": result.session,
    }
    click.echo(json.dumps(verdict))

    if not result.success:
        click.echo(f"Check failed: {result.error}", err=True)
        raise SystemExit(1)

    if post_event and result.finding:
        from bobi.events.publish import post_event as publish_event

        data = {"summary": result.summary, "text": result.summary, **result.details}
        if publish_event(post_event, data, project_path=_detect_project_root()):
            click.echo(f"Posted event: {post_event}")
        else:
            click.echo(f"Could not post event: {post_event}", err=True)
            raise SystemExit(1)


@agents.command("update")
@click.argument("name", default=None, required=False)
def agents_update(name):
    """Update agent teams from the remote registry.

    Usage:
        bobi agents update eng-team         # update one pack to latest
        bobi agents update eng-team@1.1.0   # pin to an immutable version
        bobi agents update                  # update all cached packs
    """
    from bobi.registry import (fetch, list_cached, check_update,
                                    split_team_ref, _read_local_version)

    if name:
        pkg_name, version = split_team_ref(name)  # D-6: split on the last `@`
        try:
            if version:
                # A pin targets an immutable asset — fetch directly (idempotent),
                # no latest-vs-local short-circuit.
                fetch(pkg_name, version=version)
                new_v = _read_local_version(pkg_name) or version
                click.echo(f"Pinned {pkg_name} to v{new_v}")
                return
            local_v, remote_v = check_update(pkg_name)
            if local_v and remote_v and remote_v == local_v:
                click.echo(f"{pkg_name} v{local_v} is already up to date.")
                return
            path = fetch(pkg_name)
            new_v = _read_local_version(pkg_name) or "unknown"
            if local_v:
                click.echo(f"Updated {pkg_name}: v{local_v} → v{new_v}")
            else:
                click.echo(f"Installed {pkg_name} v{new_v} → {path}")
        except Exception as e:
            click.echo(f"Failed: {e}", err=True)
            raise SystemExit(1)
    else:
        cached = list_cached()
        if not cached:
            click.echo("No cached agent teams to update.")
            return
        failed = 0
        for pack in cached:
            try:
                local_v, remote_v = check_update(pack["name"])
                if local_v and remote_v and remote_v == local_v:
                    click.echo(f"  {pack['name']} v{local_v} — up to date")
                elif remote_v:
                    fetch(pack["name"])
                    click.echo(f"  {pack['name']} v{local_v} → v{remote_v}")
                else:
                    click.echo(f"  {pack['name']} v{local_v} — could not check remote")
            except Exception as e:
                click.echo(f"  {pack['name']} — failed: {e}", err=True)
                failed += 1
        # Keep going through every pack (one bad registry must not hide the
        # rest), but report the failure. Without this the update-all form
        # exited 0 while the named-pack form above exited 1 for the identical
        # failure, so nothing scripted could tell an update from a no-op.
        if failed:
            raise SystemExit(1)


@agents.command("add-registry")
@click.argument("repo")
def agents_add_registry(repo):
    """Add a registry to fetch agent teams from.

    A registry is a GitHub repo containing an agents/ directory
    with agent teams and a registry.yaml index.

    Usage:
        bobi agents add-registry myorg/my-agents
    """
    import yaml as _yaml

    config_path = paths.ensure_global_config()
    raw = _yaml.safe_load(config_path.read_text()) or {}
    registries = raw.get("registries", [])

    if repo in registries:
        click.echo(f"Registry '{repo}' is already configured.")
        return

    registries.append(repo)
    raw["registries"] = registries
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_yaml.dump(raw, default_flow_style=False))
    click.echo(f"Added registry: {repo}")


@agents.command("remove-registry")
@click.argument("repo")
def agents_remove_registry(repo):
    """Remove a registry.

    Usage:
        bobi agents remove-registry myorg/my-agents
    """
    import yaml as _yaml

    config_path = paths.ensure_global_config()
    raw = _yaml.safe_load(config_path.read_text()) or {}
    registries = raw.get("registries", [])

    if repo not in registries:
        click.echo(f"Registry '{repo}' is not configured.", err=True)
        raise SystemExit(1)

    registries.remove(repo)
    raw["registries"] = registries
    config_path.write_text(_yaml.dump(raw, default_flow_style=False))
    click.echo(f"Removed registry: {repo}")


@agents.command("browse")
def agents_browse():
    """Browse available agent teams from the remote registry.

    Shows all packs available for install, along with their versions
    and whether they're already cached locally.

    Usage:
        bobi agents browse
    """
    from bobi.registry import list_remote, list_cached, DEFAULT_REPO

    remote = list_remote()
    if not remote:
        click.echo("Could not fetch remote registry.", err=True)
        raise SystemExit(1)

    cached_packs = list_cached()
    cached = {p["name"]: str(p["version"]) for p in cached_packs}

    click.echo("Available agent teams:\n")
    for pack in remote:
        name = pack["name"]
        # A registry.yaml is third-party YAML: `version: 1.0` parses as a float,
        # which `:8s` below rejects with "Unknown format code 's'" — taking down
        # the whole listing over one row. It also made the local-vs-remote
        # comparison a silent str-vs-float mismatch, so an installed pack read
        # as an available upgrade to itself.
        version = str(pack.get("version", "?"))
        desc = pack.get("description", "")
        registry = pack.get("registry", DEFAULT_REPO)
        local_v = cached.get(name)
        if local_v:
            if local_v == version:
                status = "installed"
            else:
                status = f"v{local_v} → v{version} available"
        else:
            status = "not installed"
        click.echo(f"  {name:20s} v{version:8s} [{status}]")
        if desc:
            click.echo(f"  {'':20s} {desc}")
        if registry != DEFAULT_REPO:
            click.echo(f"  {'':20s} registry: {registry}")
        click.echo()

    click.echo("Install with: bobi agents update <name>")


# ---------------------------------------------------------------------------
# kb group
# ---------------------------------------------------------------------------

@main.group()
def kb():
    """Knowledge base — create, populate, and search named KBs."""
    pass


@kb.command("create")
@click.argument("name")
def kb_create(name):
    """Create a new knowledge base.

    Usage:
        bobi agent <name> kb create docs
    """
    from bobi.kb.store import KBStore
    _detect_project_root()
    try:
        store = KBStore.create(name)
        click.echo(f"Created KB '{name}'")
    except FileExistsError:
        click.echo(f"KB '{name}' already exists.", err=True)
        raise SystemExit(1)


@kb.command("add")
@click.argument("name")
@click.option("--file", "-f", "file_path", type=click.Path(exists=True),
              help="Path to file to add")
@click.option("--text", "-t", "text", help="Inline text to add")
def kb_add(name, file_path, text):
    """Add content to a knowledge base.

    Usage:
        bobi agent <name> kb add docs --file README.md
        bobi agent <name> kb add docs --text "Important fact"
    """
    from bobi.kb.store import KBStore
    from bobi.kb.embedder import embed
    _detect_project_root()

    try:
        store = KBStore(name)
    except FileNotFoundError:
        click.echo(f"KB '{name}' does not exist. Create it first with the named kb create command.", err=True)
        raise SystemExit(1)

    if file_path:
        ids = store.add_file(Path(file_path), embed_fn=embed)
        if not ids:
            click.echo(f"File already indexed (unchanged)")
        else:
            click.echo(f"Added {len(ids)} chunks from {file_path}")
    elif text:
        ids = store.add_text(text, embed_fn=embed)
        click.echo(f"Added {len(ids)} chunks")
    else:
        click.echo("Provide --file or --text", err=True)
        raise SystemExit(1)


@kb.command("search")
@click.argument("name")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--mode", type=click.Choice(["hybrid", "fts", "vector"]),
              default="hybrid", help="Search mode")
def kb_search(name, query, limit, mode):
    """Search a knowledge base.

    Usage:
        bobi agent <name> kb search docs "authentication flow"
        bobi agent <name> kb search docs "login bug" --limit 5
        bobi agent <name> kb search docs "exact phrase" --mode fts
    """
    from bobi.kb.store import KBStore
    from bobi.kb.embedder import embed
    _detect_project_root()

    try:
        store = KBStore(name)
    except FileNotFoundError:
        click.echo(f"KB '{name}' does not exist.", err=True)
        raise SystemExit(1)

    embed_fn = embed if mode in ("hybrid", "vector") else None
    results = store.search(query, limit=limit, embed_fn=embed_fn)

    if not results:
        click.echo("No results.")
        return

    for i, r in enumerate(results, 1):
        source = r.get("source", "")
        score = r.get("score", 0)
        content = r["content"][:200].replace("\n", " ")
        click.echo(f"  {i}. [{score:.3f}] {source}")
        click.echo(f"     {content}")
        click.echo()


@kb.command("list")
def kb_list():
    """List all knowledge bases.

    Usage:
        bobi agent <name> kb list
    """
    from bobi.kb.store import KBStore
    _detect_project_root()

    kbs = KBStore.list_kbs()
    if not kbs:
        click.echo("No knowledge bases. Create one with the named kb create command.")
        return
    for k in kbs:
        click.echo(f"  {k['name']:20s} {k['entry_count']} entries  {k['created_at'][:19]}")


@kb.command("info")
@click.argument("name")
def kb_info(name):
    """Show knowledge base statistics.

    Usage:
        bobi agent <name> kb info docs
    """
    from bobi.kb.store import KBStore
    _detect_project_root()

    try:
        store = KBStore(name)
    except FileNotFoundError:
        click.echo(f"KB '{name}' does not exist.", err=True)
        raise SystemExit(1)

    info = store.info()
    click.echo(f"  Name:       {info['name']}")
    click.echo(f"  Entries:    {info['entry_count']}")
    click.echo(f"  Sources:    {info['source_count']}")
    click.echo(f"  Model:      {info['embedding_model']}")
    click.echo(f"  Created:    {info['created_at']}")
    if info.get("sources"):
        click.echo(f"  Files:")
        for s in info["sources"]:
            click.echo(f"    {s['source']}: {s['count']} chunks")


@kb.command("remove")
@click.argument("name")
@click.confirmation_option(prompt="Delete this knowledge base?")
def kb_remove(name):
    """Delete a knowledge base.

    Usage:
        bobi agent <name> kb remove docs
    """
    from bobi.kb.store import KBStore
    _detect_project_root()

    try:
        KBStore.remove(name)
        click.echo(f"Removed KB '{name}'")
    except FileNotFoundError:
        click.echo(f"KB '{name}' does not exist.", err=True)
        raise SystemExit(1)


main.add_command(kb)


@main.command("recall-memory")
@click.argument("query")
@click.option("--limit", "-n", default=5, type=click.IntRange(1, 50),
              help="Max results")
def recall_memory(query, limit):
    """Search the cold long-term memory reference KB."""
    from bobi.kb.store import KBStore
    from bobi.kb.embedder import embed
    from bobi.memory import COLD_MEMORY_KB_NAME

    _detect_project_root()
    try:
        store = KBStore(COLD_MEMORY_KB_NAME)
    except FileNotFoundError:
        click.echo("No cold memory index yet.")
        return

    results = store.search(query, limit=limit, embed_fn=embed)
    if not results:
        click.echo("No memory matches.")
        return

    for i, r in enumerate(results, 1):
        metadata = r.get("metadata") or {}
        category = metadata.get("category", "memory")
        source = r.get("source", "")
        score = r.get("score", 0)
        content = r["content"][:500].replace("\n", " ")
        click.echo(f"{i}. [{score:.3f}] {category} {source}")
        click.echo(f"   {content}")
        click.echo()


# ---------------------------------------------------------------------------
# costs command
# ---------------------------------------------------------------------------


@main.group(invoke_without_command=True)
@click.option("--by", "group_by", default="provider",
              type=click.Choice(["provider", "model", "session", "role"]),
              help="Group costs by dimension")
@click.pass_context
def costs(ctx, group_by):
    """Show cost attribution across sessions, grouped by provider/model/role.

    Aggregates total_cost_usd and model_usage from all session state files.

    Usage:
        bobi agent <name> costs
        bobi agent <name> costs --by model
        bobi agent <name> costs --by role
        bobi agent <name> costs --by session
        bobi agent <name> costs backfill --write
    """
    if ctx.invoked_subcommand is not None:
        return

    from .costs import rollup_costs, format_costs

    project_path = _detect_project_root()
    sessions_dir = paths.sessions_dir(project_path)
    summary = rollup_costs(sessions_dir)

    if summary.sessions_counted == 0:
        click.echo("No cost data found. Costs are recorded as sessions run.")
        return

    click.echo(format_costs(summary, group_by=group_by))


@costs.command("backfill")
@click.option("--claude-config-dir", type=click.Path(file_okay=False,
                                                     path_type=Path),
              default=None,
              help="Claude config dir holding projects/ (default: "
                   "$CLAUDE_CONFIG_DIR, else ~/.claude)")
@click.option("--write", is_flag=True,
              help="Apply the repairs. Without it nothing is written.")
@click.option("--dry-run", is_flag=True,
              help="Report what would be repaired (the default).")
def costs_backfill(claude_config_dir, write, dry_run):
    """Recover token telemetry for sessions that recorded zero (#935).

    Sessions run before the Claude SDK camel-case fix persisted provider
    dollars with every token counter at zero. This reads each session's
    retained Claude transcript and fills ONLY the missing counters - recorded
    tokens and provider dollars are never overwritten, and a session whose
    transcript is gone stays unknown rather than estimated into place.

    Dry run by default; re-running after a write is a no-op.

    Usage:
        bobi agent <name> costs backfill
        bobi agent <name> costs backfill --write
    """
    if write and dry_run:
        raise click.UsageError("--write and --dry-run are mutually exclusive.")

    from .usage_backfill import backfill_usage

    root = _detect_project_root()
    report = backfill_usage(claude_config_dir=claude_config_dir,
                            write=write, root=root)
    click.echo(report.render(write=write))
    if report.repaired and not write:
        click.echo("\nDry run - nothing written. Re-run with --write to apply.")


for _cmd_name in [
    "start", "stop", "restart", "status", "ui", "message", "ask", "compact",
    "events", "costs", "doctor", "login-bootstrap", "recall-memory",
    "supervise",
]:
    if _cmd_name in main.commands:
        agent.add_command(main.commands[_cmd_name])

for _group_name in ["transcript", "workflows", "roles", "monitors", "kb", "event-server"]:
    if _group_name in main.commands:
        agent.add_command(main.commands[_group_name])

for _cmd_name in ["install"]:
    if _cmd_name in main.commands:
        agents.add_command(main.commands[_cmd_name])

for _old_top_level in [
    "start", "stop", "restart", "status", "ui", "message", "ask", "compact",
    "events", "costs", "doctor", "transcript", "workflows", "roles", "monitors", "kb",
    "event-server", "login-bootstrap", "recall-memory", "install", "supervise",
]:
    main.commands.pop(_old_top_level, None)


if __name__ == "__main__":
    main()
