# Adapter Spec (v1)

Each harness adapter is a Python module at `obench/adapters/<name>.py`.

## Required module-level API

```python
NAME: str                # canonical harness name, matches filename
MODELS: dict[str, str]   # canonical model name -> harness-specific model string
                         # canonical key required for M3: "gpt-5.5-medium"

def run(instruction: str, workdir: str, model: str, timeout_s: int) -> dict:
    """Run the harness headlessly on `instruction` with cwd=`workdir`.

    - `model` is the CANONICAL model name; the adapter maps it via MODELS.
    - Must never prompt interactively.
    - Must enforce timeout_s (Python subprocess timeout; macOS has no `timeout` cmd).
    - The agent's file edits must land in `workdir` (the runner passes a
      disposable copy of the task workspace).

    Returns:
    {
      "completed": bool,        # harness process exited 0 within timeout
                                # (NOT task success - the checker decides that)
      "error": str | None,      # timeout / crash / unsupported-model reason
      "output_tail": str,       # last ~2000 chars of combined stdout+stderr
      "tokens": int | None,     # if the harness reports usage, else None
      "turns": int | None,      # if the harness reports it, else None
      "cmd": list | str,        # what was executed (for the results log)
      # OPTIONAL keys (extra keys are spec-compatible; the runner ignores
      # unknown ones). Provide where cheaply available:
      "full_output": str,       # full UNTRUNCATED stdout+stderr. The runner
                                # persists this (else output_tail) as the cell's
                                # local transcript. LOCAL-ONLY: transcripts are
                                # never published without a manual scrub review.
    }
    """
```

## Rules

- stdlib only (`subprocess`, `os`, `shutil`, `tempfile`, `json`, ...).
- Auth quirks live INSIDE the adapter:
  - `pi`: isolated `HOME` (temp dir) with only `~/.pi/agent/auth.json` copied in,
    so the user's personal extensions never load.
  - `opencode`: strip `OPENAI_API_KEY` from the child env to force subscription
    OAuth (stored credential at `~/.local/share/opencode/auth.json`).
  - `codex` / `cursor`: use the user's existing login as-is.
  - `codex_v1` / `codex_v2`: compose a runtime temp `CODEX_HOME` from the
    checked-in ablation config/instructions plus only the runtime Codex
    `auth.json`; auth is never copied into the repo.
- Never modify the user's real config files (`~/.codex/config.toml`,
  `~/.pi/*`, `~/.cursor/*`, opencode config). Read-only use.
- Task success is decided by the runner's checker, never by the adapter.

## Runner contract (context)

The runner copies `tasks/<task>/workspace/` to a fresh temp dir, calls
`run()`, then executes `tasks/<task>/checker.sh` with cwd=that temp dir.
Checker exit 0 = task success. A built-in `null` adapter (does nothing,
returns completed=True) is used as a negative control.

## Verified model pins (canonical `gpt-5.5-medium`)

| Harness  | Invocation hint (verify against --help before relying on it)        |
|----------|---------------------------------------------------------------------|
| codex    | `codex exec -m gpt-5.5 -c model_reasoning_effort="medium" ...`       |
| codex_v1 | `codex` with runtime `CODEX_HOME` from `ablation/codex-home-v1`       |
| codex_v2 | `codex` with runtime `CODEX_HOME` from `ablation/codex-home-v2`       |
| pi       | `pi -p --model openai/gpt-5.5 ...` (thinking-level syntax `:medium`) |
| opencode | `opencode run -m openai/gpt-5.5 --variant medium ...`                |
| cursor   | `cursor-agent -p --force --model gpt-5.5-medium ...`                 |
| devin    | CLI at ~/.local/bin/devin - headless support unknown, investigate    |

## Open-model registry (optional)

Adapters that support open / BYO models carry an in-code `OPEN_MODELS` dict.
That dict is the default. To add or retune a route without editing adapter
code, write a TOML registry and the adapters merge it over their defaults at
import time:

1. `$OPENBENCH_OPEN_MODELS` (explicit path; set it empty to disable lookup)
2. `.openbench/open_models.toml`, walking up from the working directory
3. `~/.openbench/open_models.toml`

Shared fields go on the model table, adapter-specific ones in a subtable named
for the adapter:

```toml
[models.qwen3-coder]
provider = "openrouter"
model_id = "qwen/qwen3-coder"
base_url = "https://openrouter.ai/api/v1"
env_key  = "OPENROUTER_API_KEY"
display  = "OpenRouter Qwen3 Coder"

[models.qwen3-coder.opencode]
variant = "high"
```

One `[models.<name>]` block is enough for every adapter: each fills its own
field (`effort`, `variant`, `thinking`, `proxy_route`) from its defaults unless
a subtable overrides it. `grokbuild` derives `proxy_route` from `provider`.

A name containing a dot must be quoted, or TOML reads it as a dotted key:
`[models."glm-5.2"]`, not `[models.glm-5.2]`. Most of the built-in names have a
dot in them.

Overriding one field of a built-in leaves the rest of that entry intact, nested
tables (`compat`, `thinkingLevelMap`) included, so
`[models."glm-5.2"] model_id = "glm-5.2-airx"` keeps the existing `env_key`,
`base_url` and compat flags.

`model_id`, `base_url`, `env_key` and `display` are required on any model the
registry introduces, plus whatever the adapter needs. A malformed registry
raises at import rather than silently falling back, so a bad file fails the run
instead of quietly changing which model was measured.

Every adapter also exposes `OPEN_MODELS_SOURCE`, the registry path or `None`.
`obench doctor` marks a registry-sourced model as `(open via registry)` and
prints the path, so a run whose model set was overridden is visible in preflight
rather than only in the config file.
