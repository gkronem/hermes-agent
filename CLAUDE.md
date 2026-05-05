# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

```bash
uv venv venv --python 3.11 && source venv/bin/activate
uv pip install -e ".[all,dev]"

python -m pytest tests/ -q                      # Run full test suite
python -m pytest tests/test_hermes_state.py -q  # Run a single test file
python -m pytest tests/ -q -k "keyword"         # Run matching tests

python cli.py                  # Start interactive TUI
python cli.py -q "your query"  # Single-query mode
```

Config lives at `~/.hermes/config.yaml`; API keys at `~/.hermes/.env`. `HERMES_HOME` env var overrides the default `~/.hermes` path.

## Architecture

See `AGENTS.md` for the full project structure map with file descriptions.

**Entry points:**
- `cli.py` — `HermesCLI`: full `prompt_toolkit` TUI with slash-command autocomplete, interrupt-and-redirect, streaming tool output
- `run_agent.py` — `AIAgent`: programmatic conversation loop used by CLI, gateway, batch runner, and RL environments
- `hermes_cli/main.py` — all `hermes` subcommands (`hermes model`, `hermes tools`, `hermes gateway`, `hermes setup`, etc.)

**Core import dependency chain** (order is load-order sensitive):
```
tools/registry.py → tools/*.py → model_tools.py → run_agent.py / cli.py
```

**Key packages:**
- `agent/` — internals: `prompt_builder.py` (system prompt assembly), `context_compressor.py` (auto-compression when context fills), `memory_manager.py`, `model_metadata.py` (context window sizes), `display.py` (spinner, tool preview)
- `tools/` — one file per tool; each calls `registry.register()` at import time. `terminal_tool.py` orchestrates 6 terminal backends (local, Docker, SSH, Modal, Daytona, Singularity). `approval.py` handles dangerous command detection.
- `gateway/` — messaging adapters (Telegram, Discord, Slack, WhatsApp, Signal, etc.); `run.py` is the main loop, `session.py` handles conversation persistence
- `hermes_state.py` — `SessionDB`: SQLite with WAL mode + FTS5 full-text search across all session messages
- `hermes_cli/` — CLI subcommands, config migration, skin engine, skills hub, model switching
- `cron/` — in-process cron scheduler (`scheduler.py`, `jobs.py`) for scheduled automations delivered to any platform
- `acp_adapter/` — ACP server for IDE integrations (VS Code, Zed, JetBrains)

**Skills vs Tools:** New capabilities should almost always be skills (a `SKILL.md` file invoking existing tools), not new Python tool implementations. See `CONTRIBUTING.md` for the full decision criteria and the bundled vs optional-skills distinction.

## Testing

Tests use `pytest` + `pytest-asyncio`. The `tests/conftest.py` provides shared fixtures. Tests that require external API access are gated by env vars. The RL submodule (`tinker-atropos`) must be initialized separately if needed: `git submodule update --init tinker-atropos`.
