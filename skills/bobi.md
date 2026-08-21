# Using Bobi

Guide the user through running, operating, and extending Bobi Agents.
Bobi is an event-driven framework for persistent AI agent teams. Domain
behavior comes from Bobi Agent packages: roles, workflows, monitors,
tools, context files, and workspace templates.

## Directory Model

`BOBI_HOME` is the only user-configurable home location. It is set by
environment variable and defaults to `~/.bobi`.

```text
$BOBI_HOME/
├── config.yaml
└── agents/
    └── <name>/
        ├── src/              # editable Bobi Agent source
        └── run/              # selected runtime root
            ├── package/      # installed frozen package
            ├── state/        # sessions, logs, pid files, policy
            ├── workspace/    # user-owned domain files and outputs
            └── .env          # runtime credentials
```

### Runtime logs

`state/` holds append-only logs retained across days: `manager.log` (the
manager and everything it spawns) and `embedding-sidecar.log` (the knowledge
base's embedding server). Every line carries a full ISO-8601 local timestamp
with its UTC offset, and each record appears exactly once:

```text
2026-07-25T15:00:19.417-07:00 [INFO] Monitor sales-call-manager due - spawning non-interactive check
```

Read the date, not just the clock. Adjacent lines in these files are routinely
days apart - a monitor that fires once daily puts four `15:00` lines next to
each other - so a bare wall clock reads as a burst that never happened. For how
often a monitor actually ran, `state/monitor_state.json` is authoritative; the
log shows only what was retained.

Two logs nearby are NOT in this format. `state/event-server.log` is written by
the Node event server through bare `console` calls and carries no timestamp at
all, so it cannot be read for timing the way the two above can. The web app's
`app.log` is not under `state/` at all: it lives at `$BOBI_HOME/webapp/app.log`,
and its uvicorn lines carry their own timestamp-free format.

Runtime commands are scoped to one installed Bobi Agent:

```bash
bobi agents list
bobi agents install ./agents/eng-team --name eng
bobi agent eng start
bobi agent eng status
bobi agent eng ask "what's the status?"
```

## Machine Commands

```bash
bobi app start                        # unified web app (dashboard + onboarding
bobi app stop|restart|status          #   + chat), runs in the background
bobi setup <name>                     # design/build/install a Bobi Agent
bobi agents install <source> --name <name>
bobi agents install <source> --name <name> --with-deps  # + install declared deps locally
bobi agents list
bobi agents browse
bobi agents update <name>
bobi agents add-registry <repo>
bobi build <team> --tag <ref> [--push]  # render a team into a ready-to-run
                                        #   image (needs the deploy plugin)
```

`<source>` can be a local source directory, local `.tar.gz`, public
`.tar.gz` URL, or registry name.

## Runtime Commands

```bash
bobi agent <name> start
bobi agent <name> stop
bobi agent <name> restart      # safe to run from inside the runtime
bobi agent <name> start --fresh
bobi agent <name> status
bobi agent <name> doctor

# Supervise the manager as the terminal process (containers, pod specs).
# Spawns + probes the manager, publishes heartbeat/lifecycle telemetry, and
# listens on the admin topic so a wedged manager can still be restarted.
# Everything after `--` forwards to the manager's start command.
# This is what a container entrypoint runs as PID 1 - not for interactive use.
bobi agent <name> supervise -- --foreground

bobi agent <name> ask "question"
bobi agent <name> message "text"
bobi agent <name> compact
bobi agent <name> events
bobi agent <name> events publish alert/firing --json '{"title":"x"}'

# Scoped ingest tokens: let an external system (alerting, CI, SaaS webhooks)
# POST plain JSON to one topic via /webhooks/ingest/<topic>. The token is
# shown once at creation; the server stores only a hash.
bobi agent <name> events ingest-token create alert/firing --name oncall
bobi agent <name> events ingest-token list
bobi agent <name> events ingest-token revoke <id>

bobi agent <name> transcript show manager
bobi agent <name> transcript search "query"
bobi agent <name> costs

# Recover token telemetry for sessions that recorded zero (#935). Reads each
# session's retained Claude transcript and fills ONLY missing counters -
# recorded tokens and provider dollars are never overwritten. Dry run by
# default; re-running after a write is a no-op.
bobi agent <name> costs backfill
bobi agent <name> costs backfill --write

# Reply into a chat conversation (channel-agnostic; ref comes from the event)
bobi reply <conversation> "markdown text"
bobi reply <conversation> --edit <ts> "text"     # resolve a placeholder
bobi reply <conversation> --file <path> "comment"
bobi read-conversation <conversation> [-n 50] [--json-output]
```

Use `bobi reply` and `bobi read-conversation` for Slack and any other
chat channel delivered through the channel gateway.

`restart` hands its stop and start phases to a detached worker. The restart
therefore completes even when stopping the manager also kills the process that
requested it. The latest worker record is kept in
`~/.bobi/agents/<name>/run/state/restart.log` for diagnosis.

## Upgrading Bobi In Place

A local upgrade replaces bobi's files underneath whatever is already
running, and neither the team reinstall nor `bobi agent <name> restart`
restarts the local event server. Restart both:

```bash
uv tool install --upgrade bobi
bobi agents install ./agents/<team> --name <name>
bobi agent <name> restart
bobi agent <name> event-server restart
```

Each long-lived process records the bobi it launched from, so anything
still on the replaced code is named - by the install itself, and by the
`Running code` check in `bobi agent <name> doctor`, with the restart
command to clear it. Containers cannot drift this way: the image is the
unit of update, so a new version is a new process.

## Sub-Agents

Sub-agents are child executions launched by a Bobi Agent runtime. Use
them for delegated work and workflow steps.

```bash
bobi agent <name> subagents launch -w adhoc --role engineer --task "Fix CI"
bobi agent <name> subagents launch -w adhoc --role engineer --wait --task "Fix CI"
bobi agent <name> subagents launch -w adhoc --role monitor --as-check --task "Check prod"
bobi agent <name> subagents list
bobi agent <name> subagents show <id>
bobi agent <name> subagents cancel <id>
```

### Run keys and duplicate suppression

Every launch has a run key. It names the session
(`wf-<workflow>-<project>-<key>`), so two launches that agree on it are the same
run: the second is refused while the first is active, and resumes it once it is
not.

- `--id <key>` sets it explicitly. Use it for work with a natural identity - an
  issue number, a checklist unit. Relaunching that key resumes that run.
- With no `--id` the key is **derived** from the launch itself - workflow,
  project, role, model, effort and the task text - so relaunching an identical
  one while the first is still running is refused. That is the guardrail
  against a dispatch chain that keeps launching itself; rewording the task to
  get past it defeats it. Fanning one task across two roles is fine: they
  derive different keys.
- `--id-random` mints a random key, for deliberately running N copies of an
  identical task at once. It cannot be combined with `--id`, and its keys are
  prefixed `rand-` so `subagents list` shows which runs opted out.
- A workflow that declares `period:` overrides all of the above: the key is
  always `<workflow>-<period bucket>` (e.g. `daily-standup-2026-08-10`), one
  run per period per repo across every dispatch path. `--id` and
  `--id-random` are overridden, and a completed period refuses relaunch -
  `--fresh` is the deliberate escape hatch to run a period again.

Relaunching a key whose previous run **failed** resumes from its step
checkpoint rather than replaying completed steps, with the new `--task` and
`--input` values taking effect from the resumed step onward. `--fresh`
replays from step 0.

A derived key also implies `--fresh`: it is an inference about the launch, not a
caller pointing at a run to continue. It additionally refuses to land on a
**suspended** (`waiting`) run - only an explicit `--id` may re-dispatch onto one.

`--wait` blocks until the launched adhoc agent completes and prints its final
text. It still requires `-w adhoc` (the fan-out unit shape), but it is the
same launch as a detached one (#1057): an ad-hoc task runs as a one-step
workflow, so a `--wait` run gets a run-ledger entry, checkpoint retry, and
every rule above - the derived key, the active-run guard, and period
override included. Two identical `--wait` launches therefore refuse each
other; fan identical copies out with `--id-random`, exactly as detached.
`--timeout` is the run's declared deadline for the dead-man reconciler, the
same as a detached run - the in-process bound on a runaway agent is the
role's turn cap, not a wall clock.
`--as-check` is the explicit short-lived monitoring-check harness; it prints
verdict JSON and is the only `subagents launch` mode that accepts
`--post-event`. It never reaches the launch admission path, so run keys do not
apply to it.

To fan out and join without burning a turn per check, start the units in the
background and block on all of them in a **single** shell command:

```bash
bobi agent <name> subagents launch -w adhoc --role engineer --wait \
  --task "Review bobi/workflow/" > /tmp/r1.log 2>&1 &
bobi agent <name> subagents launch -w adhoc --role engineer --wait \
  --task "Review bobi/brain/" > /tmp/r2.log 2>&1 &
wait; tail -20 /tmp/r1.log /tmp/r2.log
```

Polling a log in a loop instead is the pattern this replaces: one real engineer
session spent 79 of its 201 turns doing exactly that (#845).

`--model` and `--effort` override the launched agent's model and reasoning
effort for the whole run (provider-native values; they win over workflow step
and role config), so an agent can pick both dials per delegation:

```bash
bobi agent <name> subagents launch -w adhoc --role engineer \
  --model gpt-5.6 --effort xhigh --task "Design the migration"
```

## Telemetry

Record agent-authored metrics and logs to an OTLP endpoint. The agent chooses
what is worth recording; bobi stamps the fleet identity labels.

```bash
bobi agent <name> otel check
bobi agent <name> otel check --send
bobi agent <name> otel metric tickets.processed 42
bobi agent <name> otel metric queue.depth 7 --kind gauge --attr queue=inbox
bobi agent <name> otel log "reconciled the backlog" --severity info
```

The destination comes from `OTEL_EXPORTER_OTLP_ENDPOINT` (see `docs/OTEL.md`);
`check` reports an unconfigured box as such and makes no network call without
`--send`. `--attr` values are always sent as strings and must stay
low-cardinality - they become time-series labels. Opt in per team with
`tool_library: [otel]`; it is deliberately not in every agent's default prompt.

## Package Surfaces

Installed package files live under `run/package/`:

```text
package/
├── agent.yaml
├── agent.md
├── roles/<role>/ROLE.md
├── tools/*.md
├── workflows/*.yaml
├── monitors/defaults.yaml
└── context/*.md
```

Edit the source under `$BOBI_HOME/agents/<name>/src/` or the
user-chosen source directory, then reinstall. Runtime state and
credentials live under `run/` and should not be edited into package
source.

## Common Tasks

```bash
# Create a new Bobi Agent interactively
bobi setup support

# Install a checked-out team source
bobi agents install ~/agent-teams/support --name support

# Run and talk to it
bobi agent support start
bobi agent support ask "summarize the current queue"

# Inspect operation
bobi agent support status
bobi agent support events
echo '{"title":"x"}' | bobi agent support events publish alert/firing
bobi agent support transcript show manager
```

## Rules of Thumb

- Use the `agents` command group for machine-wide Bobi Agent management.
- Use the named `agent` command group for runtime operations.
- Use `subagents` for child agent executions.
- Put source-controlled team definitions in `src/` or another explicit
  source directory.
- Treat `run/package/` as generated install output and `run/state/` as
  mutable runtime state.
