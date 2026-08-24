# Quickstart

Go from nothing to a running Bobi agent - installed, set up, and deployed
either on your own machine or as a container in the cloud. Every command is
copy-pasteable.
Expect about 15 minutes for the local path; add 15-30 more for a cloud deploy.

If you get stuck at any point, jump to [If you get stuck](#if-you-get-stuck) -
worst case, you can point a Claude Code or Codex session at Bobi's own docs and
have it troubleshoot with you.

## What you'll end up with

- The `bobi` CLI installed.
- An agent runtime (Claude Code) installed and logged in.
- Your first agent team, designed with the `bobi setup` client and installed
  on your machine.
- The agent running locally, or deployed as an always-on container instance.
- Optionally, Slack wired up so you can talk to your team from chat.

## Prerequisites

- **macOS or Linux** with a terminal.
- **An Anthropic account** - a Claude Pro/Max subscription or an API key.
  The `bobi setup` client runs on Claude Code, so you need this even if your
  agents will run on OpenAI Codex. (Agents themselves can run on either
  runtime; see [Choose the runtime](../README.md#choose-the-runtime-optional)
  for switching a team to Codex.)
- **Homebrew or uv** to install the CLI. If you have neither, uv is the
  quickest to get:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **Node.js 20 or newer** for Bobi's embedded local event server.
  For current releases, install Node separately and verify `node --version` reports 20 or newer, including when installing Bobi with Homebrew.
  A companion Homebrew formula update will automate this prerequisite once it ships.

- **Optional, for cloud deployment**: somewhere to run a container - a VM with
  Docker, a Kubernetes cluster, or any container host (Step 5, Option B).

You do not need to clone the Bobi repo. Bobi is a published package - install
the CLI and go.

## Step 1: Install and log in to Claude Code

Bobi's setup client and (by default) each agent run on Claude Code. Skip this
step if you already have it installed and logged in.

```bash
brew install --cask claude-code
```

Or with npm:

```bash
npm install -g @anthropic-ai/claude-code
```

Then launch it once to log in:

```bash
claude
```

Follow the login prompt (choose your Claude subscription or API key), then exit
the session with `/exit`. Verify it works:

```bash
claude --version
```

> **Tip:** if `claude` is not found after installing, open a new terminal so
> your `PATH` refreshes. See the
> [Claude Code docs](https://docs.anthropic.com/en/docs/claude-code) for
> platform-specific help.

## Step 2: Install Bobi

With Homebrew:

```bash
brew install moda-labs/bobi-agent/bobi
```

Until the companion Homebrew formula update ships, Homebrew users must install Node.js 20 or newer separately.
For every install method, verify the supported runtime first:

```bash
node --version
```

Then install with uv:

```bash
uv tool install bobi
```

Verify the install:

```bash
bobi --version
```

> **Tip:** with uv, if `bobi` is not found, run
> `export PATH="$HOME/.local/bin:$PATH"` (and add it to your shell profile).
> You can also install with `pipx install bobi` if you prefer pipx.

Everything Bobi creates lives under one home directory, `~/.bobi` by default.
You can inspect or delete it at any time without affecting anything else.

## Step 3: Create your agent team with the setup client

```bash
bobi setup my-agent
```

This opens a local web UI (on `127.0.0.1` - nothing leaves your machine) that
takes you from an idea to an installed, runnable agent team. Name it whatever
you like - the examples below use `my-agent`; if you pick a different name,
substitute it in every later command. You can interrupt setup anytime; rerunning the same
command picks up where you left off:

```bash
bobi setup my-agent
```

The client walks you through four phases:

### 3a. Set up the agent team

<!-- TODO(screenshot): the intro screen - location field, Browse button, and the create/modify/registry tabs -->

First, pick where the agent team lives on your local drive (it defaults to the
`$BOBI_HOME/agents` library — `~/.bobi/agents/<name>/src` unless you set
`BOBI_HOME` — with a Browse button to choose elsewhere), and choose how to
start:

- **Start from scratch** - describe what you want and Bobi authors the team.
- **Use a template** - open an existing team on disk, or pull one from a
  registry, and modify it from there.

### 3b. Configure the team by chatting with Bobi

<!-- TODO(screenshot): the main setup screen - chat on the left, the team panel cards filling in on the right -->

Next, you configure the team in a conversation. The Bobi setup agent
interviews you for the details of your agent team and tracks your progress in
the panel along the right - cards fill in and check off live as you talk.
Plain-language answers are fine; "watch my GitHub repo and summarize new
issues every morning" is a perfectly good start. You'll cover:

- **Goal and roles.** Define what each agent on the team does and what
  success looks like. Bobi reflects back what it heard so you can correct it.
- **Automations and schedules.** Anything the team should do on its own -
  recurring checks, scheduled reports, triggered follow-ups. Each automation
  carries its own leash: notify you, ask first, or act and report.
- **Connections.** Hook the team up to the outside systems it needs, usually
  via MCP:
  - By default, Bobi looks for [Venn AI](https://venn.ai) - an MCP gateway
    that lets you remotely set up and manage many MCP connections from a
    single point, with one shared key.
  - For services Venn doesn't offer, your best bet is the service's
    **official hosted MCP**: drop in its official URL and connect with an API
    key.
  - For a service with no official remote MCP, you can build an MCP server
    locally and connect it via a local command.
  - If you ever get stuck on the mechanics - creating a GitHub API token, or
    any other credential - just ask the Bobi setup agent in the chat. It will
    walk you through it.
- **Chat integration.** Set up Slack (or other chat) configuration here. You
  can always talk to your team from the command line, but the real magic is
  talking to it from Slack - you finish wiring that up after deployment
  (Step 6).

### 3c. Preview and confirm

<!-- TODO(screenshot): the post-build file browser - file tree, file contents, Open folder -->

Before anything is final, Bobi builds the team and shows you every file it
created - browse the tree, read the contents, open the folder on disk - and
you confirm the agent team creation.

### 3d. The completion screen: your next three moves

<!-- TODO(screenshot): the completion screen - success banner and the three next actions to cycle through -->

Once your team is created, the final screen confirms it now lives on your
local drive and walks you through your next three actions - one full screen
each, cycle forward and back through them (you can close the window or return
to the home screen at any point):

1. **Test your Bobi team locally** by running it from your terminal - Step 4
   below.
2. **Deploy the agent team** - on your own hardware or as a cloud container -
   Step 5.
3. **Finalize chat (Slack) configuration** so you can send and receive
   messages from Slack - Step 6.

The rest of this guide follows those same three moves.

## Step 4: Start the agent and talk to it

Start it:

```bash
bobi agent my-agent start
```

This runs preflight checks, launches a local event server (loopback only -
nothing leaves your machine), and starts the agent as a background daemon.

Check that it's healthy:

```bash
bobi agent my-agent status
bobi agent my-agent doctor
```

`doctor` runs a full health check - runtime layout, credentials, workflows,
monitors - and prints a hint for anything it finds wrong. It's the first thing
to run whenever something misbehaves.

Now talk to your agent. `ask` blocks until it responds:

```bash
bobi agent my-agent ask "What can I help with right now?"
```

`message` is fire-and-forget:

```bash
bobi agent my-agent message "FYI: the staging deploy is paused this week"
```

Hand it a one-off task as a sub-agent:

```bash
bobi agent my-agent subagents launch -w adhoc --role engineer --task "Fix the login bug"
```

(`-w adhoc` runs an open-ended task outside any structured workflow.)

(With no `--id`, the run key is derived from the launch itself - workflow,
project, role, model, effort and the task text - so relaunching the same task
while the first run is still going is refused as a duplicate. Pass `--id <key>`
for work with a natural identity, or `--id-random` to run several copies of an
identical task at once.)

(Roles are defined by the team - run `bobi agent my-agent roles list` to see
what yours has.)

Watch what it's doing:

```bash
bobi agent my-agent events              # recent events and decisions
bobi agent my-agent subagents list      # running sub-agents
bobi agent my-agent transcript show manager   # full session transcript
bobi agent my-agent costs               # spend accounting
```

Stop and restart whenever you like - state persists:

```bash
bobi agent my-agent stop
bobi agent my-agent start
```

At this point you have a working, event-driven agent. Step 5 decides where it
lives long-term.

## Step 5: Deploy it

### Option A: run it on your own machine (always-on if the machine is)

`bobi agent my-agent start` is already a local deployment: the agent runs as a
daemon with a loopback event server, reacts to its scheduled monitors, and
answers `ask`/`message` from your terminal. If you're on hardware that stays
on - a Mac mini, a home server, a remote box - this can be your permanent
deployment. Have Claude Code or Codex help you configure the event server and
deployment for your machine: open a session and ask it to read
[EVENT_SERVER.md](EVENT_SERVER.md) and set things up with you.

Two limits to know about:

- The agent only works while the machine is on.
- Inbound webhooks from the public internet, including GitHub and Slack's default HTTP Events API, can't reach a loopback server.
  For those, deploy to Fly (Option B) or point the agent at a deployed event server.
  A self-hosted Slack agent can instead use outbound [Socket Mode](../skills/slack-setup.md#verify-socket-mode-and-migrate-safely) with the local Node event server.

If local covers your needs, skip ahead to
[Step 6](#step-6-finalize-chat-slack-configuration) or
[Where to go next](#where-to-go-next).

### Option B: deploy as a container (always-on)

Bobi publishes a reference image, `ghcr.io/moda-labs/bobi`, that runs an agent
as an always-on container on anything that already runs containers - a VM with
Docker, a Kubernetes cluster, your own orchestrator. The instance holds an
outbound WebSocket to an internet-reachable event server, so Slack and GitHub
webhooks reach it without a tunnel on your machine, and it needs no public
ingress of its own.

The easiest way through this is to let Claude Code or Codex drive it. Open a
session and paste:

```plaintext
Deploy my bobi agent "my-agent" as an always-on container from ghcr.io/moda-labs/bobi. Read https://github.com/moda-labs/bobi-agent/blob/main/docs/REFERENCE_IMAGE.md for the runtime env contract and walk me through the env vars, the durable volume, and the secrets it needs.
```

Or do it by hand:

**1. Decide how the container gets your team.** On first boot it installs the
team named by `BOBI_TEAM` (a bundled or registry name) or fetched from
`BOBI_TEAM_URL` (a public `.tar.gz` of one team package) - set exactly one. If
you designed your own team in Step 3, publish it as a tarball and point
`BOBI_TEAM_URL` at it. With neither set the container enters a *wait-for-team*
state and polls rather than crashing, which is the hook for pushing a team onto
the volume out of band.

**2. Write a secrets file.** Deployed instances default to API-key auth, so
you need an Anthropic API key (from
[console.anthropic.com](https://console.anthropic.com)) plus every credential
your team uses (the same ones you captured in setup - they live in
`~/.bobi/agents/my-agent/run/.env`). For an agent with no external services:

```bash
printf 'ANTHROPIC_API_KEY=sk-ant-your-key-here\n' > ./my-agent.env
```

Keep this file out of git.

**3. Run it.** The container expects a real PID 1 (`--init`) and a durable
volume at `/data`:

```bash
docker run -d --init --name my-agent \
  -v my-agent-data:/data \
  --env-file ./my-agent.env \
  -e BOBI_INSTANCE=my-agent \
  -e BOBI_TEAM=my-team \
  -e BOBI_AUTH=api_key \
  -e BOBI_EVENT_SERVER=https://your-event-server.example.com \
  ghcr.io/moda-labs/bobi:latest
```

First boot takes a few minutes (team install plus runtime warm-up). The volume
at `/data` is the agent's identity and its only durable state: reuse it and the
agent resumes where it left off, discard it and you get a fresh agent.

**4. Verify and operate.**

```bash
docker logs -f my-agent
```

Tear it down when you're done - dropping the volume deletes the only copy of
the instance's state:

```bash
docker rm -f my-agent && docker volume rm my-agent-data
```

`docs/REFERENCE_IMAGE.md` is the full runtime contract: every environment
variable, the PID-1 requirement per orchestrator, first-boot behavior, and the
`TEAM_DEPS` hook for baking your team's own dependencies into a derived image.
`docs/SELF_HOSTED_EVENT_SERVER.md` covers standing up the event server this
instance connects out to.

For CI-driven fleets, GitOps reconciliation across many instances, and a hosted
admin control plane, those are part of Moda's managed/enterprise offering - see
the README's [Cloud Deployment](../README.md#cloud-deployment) section.

## Step 6: Finalize chat (Slack) configuration

The command line works, but the real magic is talking to your Bobi team from Slack.
If you set up Slack during `bobi setup`, choose the default HTTP Events API path or self-hosted Socket Mode:

1. **Create the Slack app.** The manifest generator prefills the scopes and event subscriptions.
   Use the default command for a reachable HTTP webhook, or add `--socket-mode` when the local Node event server should dial out without a public Request URL:

   ```bash
   # HTTP Events API
   bobi create-slack-bot --app-name "Bobi"

   # Socket Mode for a self-hosted local Node event server
   bobi create-slack-bot --socket-mode --app-name "Bobi"
   ```

   HTTP mode points Slack's Event Subscriptions Request URL at `<event-server>/webhooks/slack`.
   Socket Mode instead requires the generated `xapp-` app-level token to be saved as `SLACK_APP_TOKEN`.

2. **Add the app to your workspace** and, ideally, invite it to a dedicated
   channel for the team (`/invite @your-bot`).
3. **Pass that channel's ID to Bobi** (the `SLACK_CHANNELS` variable in
   `~/.bobi/agents/my-agent/run/.env`) so the agent knows where to listen and
   post, then restart the agent. Send a test message in the channel to
   confirm the round trip.

The full walkthrough, including manifest contents, transport verification, migration, and troubleshooting, is in [Slack setup](../skills/slack-setup.md).

## If you get stuck

Work through these in order:

1. **Run the doctor.** It checks the runtime, credentials, services,
   workflows, and monitors, and prints a fix hint per failure:

   ```bash
   bobi agent my-agent doctor
   ```

2. **Check status and events.** `bobi agent my-agent status` shows whether
   the agent is actually running; `bobi agent my-agent events` shows what it
   last saw and decided.

3. **Fix credentials directly.** Team secrets live in
   `~/.bobi/agents/<name>/run/.env`. Edit that file and restart the agent:

   ```bash
   bobi agent my-agent restart
   ```

   Restart uses a detached worker, so it can complete when requested from
   inside the runtime. Its latest record is stored at
   `~/.bobi/agents/<name>/run/state/restart.log`.

4. **Start fresh.** If a session is wedged, wipe it and start clean (your
   workspace files and credentials are kept):

   ```bash
   bobi agent my-agent start --fresh
   ```

5. **Container deploys:** the container's own logs are the first stop
   (`docker logs <name>`, or your orchestrator's equivalent). The entrypoint
   fails loudly rather than degrading quietly - an unresolvable team name and a
   missing provider key under API-key auth each abort the boot with the reason.
   `docs/REFERENCE_IMAGE.md` documents the runtime contract those checks
   enforce.

6. **Enlist an AI pair.** Bobi's docs are written to be agent-readable. Open a
   Claude Code or Codex session next to your terminal and paste:

   ```plaintext
   Read https://raw.githubusercontent.com/moda-labs/bobi-agent/main/skills/bobi.md and help me troubleshoot my bobi agent. Here's what's happening: <describe your problem and paste any error output>
   ```

   It can run the diagnostic commands above with you and interpret the output.

7. **File an issue.** If you've found a real bug, open one at
   [github.com/moda-labs/bobi-agent/issues](https://github.com/moda-labs/bobi-agent/issues)
   with the `doctor` output and relevant logs.

## Where to go next

- **Understand how it all works** - teams, tools, workflows, events, memory:
  [OVERVIEW.md](OVERVIEW.md).
- **Talk to your agent from Slack** instead of the terminal:
  [Slack setup](../skills/slack-setup.md).
- **Connect Linear** so ticket updates drive the agent:
  [Linear setup](../skills/linear-setup.md).
- **Extend or build teams** - roles, workflows, monitors, tools:
  [create-agent skill](../skills/create-agent.md) and
  [BUILDING_AGENT_TEAMS.md](BUILDING_AGENT_TEAMS.md).
- **Full CLI reference:** [skills/bobi.md](../skills/bobi.md).
- **How the event bus works:** [EVENT_SERVER.md](EVENT_SERVER.md).
- **Security model** (read before installing third-party teams):
  [SECURITY.md](SECURITY.md).
