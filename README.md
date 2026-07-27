# OpenBench

[![CI](https://github.com/minghinmatthewlam/openbench/actions/workflows/ci.yml/badge.svg)](https://github.com/minghinmatthewlam/openbench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A benchmark for comparing coding-agent **harnesses** — the CLI tools that wrap a
model in a run loop, tool set, and permission policy (`codex`, `pi`, `opencode`,
`cursor`, `devin`, and open-model `claude`). The question it answers is: *given
the same underlying model and task, how much does the harness around it matter?*

**New here?** [`WRITEUP.md`](WRITEUP.md) tells the story arc; [`RESULTS.md`](RESULTS.md)
has the milestone analyses; [`SETUP.md`](SETUP.md) is the practical runbook for a
first local cell, Docker, imported tasks, and open-model keys. Evaluating
harnesses on a **private codebase**? Start with
[`docs/private-evals.md`](docs/private-evals.md) (`obench init`, author a task
from a small code slice, validate polarity, run, report — transcripts stay
local). To run OpenBench tasks on Harbor cloud sandboxes, see
[`docs/harbor-export.md`](docs/harbor-export.md) (`obench export harbor`).
To pull Harbor-format tasks into OpenBench, see
[`docs/harbor-import.md`](docs/harbor-import.md) (`obench import harbor`).
Versioned packs (`org/name@version`, tasks or harness manifests) are covered in
[`docs/task-packs.md`](docs/task-packs.md) (`obench pack install …`).
To compare a fixed coding harness and model over direct and gateway serving
routes, see [`docs/gateway-bench.md`](docs/gateway-bench.md) (`obench gateway
validate|doctor|run|report|publish|verify`). For direct request latency,
accounting, and route-integrity probes without a coding harness, see
[`docs/gateway-probe.md`](docs/gateway-probe.md) (`obench gateway probe
validate|doctor|run|report`).

**Live results:** https://openbench.run/ — the landing
page is the leaderboard itself, with a tab each for Harness Bench and Gateway
Bench (built by `obench site build`; see [`docs/site.md`](docs/site.md)).

## What this measures

Each harness runs headlessly against a set of self-contained coding tasks. A task
is graded by a checker script (exit 0 = solved, optional `SCORE:` for partial
credit), never by the harness's own claim of success.

OpenBench has two separate benchmark families. **Harness Bench**, described
throughout this README, varies the coding-agent harness while holding the model
and task fixed. **Gateway Bench** holds the Pi harness, model, inference
provider, sampling, and task fixed while varying the API gateway. It uses the
`fixed_model_provider` track and compares each gateway with the provider's
direct API under matched blocks. See
[`docs/gateway-bench.md`](docs/gateway-bench.md) for its methodology, metrics,
examples, and separate `obench gateway` command group. **Gateway Probe** is the
request-level companion: it compares cold and verified warm same-socket
requests without Pi, tasks, checkers, or correctness scores.

Automatic model or provider selection is a different product surface. A future
**Router Bench** will own that work under `obench router`; Gateway Bench does
not make routing decisions.

Current Harness Bench tiers:

- **Core synthetic tasks (`tasks/`).** Small-to-medium tasks built in this repo,
  including partial-credit harder tasks such as `make-ci-green`, `add-feature`,
  and `misleading-error`.
- **Exercism imported tier (`tasks-imported/exercism/`).** MIT-licensed problem-
  specification tasks with per-task provenance.
- **Terminal-Bench imported frontier tier (`tasks-imported/terminal-bench/`).**
  Five Apache-2.0 Terminal-Bench tasks adapted for OpenBench's Docker lane and
  scored separately from the core tier.

Two result framings are used:

- **Track A — same model, harness varies.** Every compatible harness is pinned to
  the same canonical model, `gpt-5.5-medium`, so differences come from the
  harness (scaffolding, tools, prompting, permissions), not the model.
- **Open-model panels.** Open models (`glm-5.2`, `deepseek-v4-flash`,
  `kimi-k2.7-code`, `glm-4.7-flash`) run through adapters that can reach the
  providers. `pi`, `opencode`, and `claude` call providers directly; `codex` uses
  the local Responses↔Chat bridge in `bench/openmodel_bridge.sh`.

**Honest caveats:** `devin` is flaky in the latest published analysis and is
excluded from M4.5 rankings; `cursor`/`devin` have closed model menus for open
models; `claude` is open-model-only here so a run cannot accidentally bill an
Anthropic subscription.

## Headline findings so far

- **Correctness saturates for frontier harnesses on repo-authored tasks.** M3 and
  M4.5 both hit the ceiling: clean frontier harnesses solved every core task, so
  correctness does not support a leaderboard on those tiers.
- **Efficiency separates even when correctness does not.** Wall-clock spread is up
  to ~4× and token tax up to ~8× on the early matrices; `pi` is repeatedly the
  fastest/leanest harness in the measured panels.
- **Open models are surprisingly close.** In M4, three of four open models reached
  the GPT-5.5-medium frontier baseline on the same hard tasks; the 72-run open-
  model matrix cost about **$1.02** in first-party API spend.
- **Terminal-Bench is the new frontier tier.** The current local TB frontier run
  covers 45 cells (3 harnesses × 5 TB tasks × 3 trials) and lands at 12/15 per
  harness (mean score 0.80), finally below the synthetic-task ceiling. The raw TB
  run log currently lives under local-only `results/` and should be promoted to a
  committed dataset before citing it externally.

## Hardware and accounts

No GPU is required. Core and Exercism tasks run on a normal laptop; Terminal-Bench
cells can take minutes and should use Docker isolation. Real harness runs require
that harness's CLI and auth. Frontier Track A uses subscription/OAuth logins;
open-model panels use first-party provider API keys kept outside the repo. See
[`SETUP.md`](SETUP.md) for install/auth caveats and one-cell commands.

## Reproduce a cheap open-model panel (~$1)

The committed **M4 open-model matrix** — two harnesses (`pi`, `opencode`) × four
open models × three harder tasks × three trials — reproduces for about **$1 of
API credit**. The dataset is in [`data/m4-2026-07-03/`](data/m4-2026-07-03/).

Create `~/.openbench/keys.env` (or export the same names) with names only in the
repo docs; values stay local:

```dotenv
ZAI_API_KEY=
DEEPSEEK_API_KEY=
MOONSHOT_API_KEY=
```

Preflight (no token spend):

```bash
obench doctor --harness pi,opencode --model glm-4.7-flash
```

Run the resumable matrix:

```bash
for m in glm-4.7-flash glm-5.2 deepseek-v4-flash kimi-k2.7-code; do
  obench run --harness pi,opencode \
    --task make-ci-green,add-feature,misleading-error \
    --model "$m" --trials 3
done
```

Report:

```bash
obench report --efficiency --results-path results/results.jsonl
```

For a single first run, Docker, imported tasks, `claude`, or `codex` open-model
runs through the bridge, use [`SETUP.md`](SETUP.md).

## Layout

```
tasks/                 core benchmark tasks (see "Task format")
tasks-imported/        separately scored Exercism and Terminal-Bench tiers
obench/                installable package (CLI: obench)
obench/run.py          the runner (one row per task x harness x trial)
obench/report.py       aggregates results into a table with Wilson CIs
obench/adapters/*.py   one adapter per harness (+ built-in "null" control)
obench/ADAPTER_SPEC.md the adapter contract
obench/openmodel_bridge.sh  foreground Codex Responses↔Chat bridge for open models
obench/scrub.py        PII scrubber for transcripts (local-only; see below)
obench/entry.py        in-container entrypoint for --exec docker
obench/docker_exec.py  container-per-cell execution backend
obench/docker/         isolation image for --exec docker
bench/*.py             thin deprecation shims → obench
validate_tasks.py      forwarding shim → obench validate
results/results.jsonl  append-only local results log (gitignored)
transcripts/           per-cell agent transcripts (gitignored, local-only)
```

## Install

The framework installs as **`obench`** (the PyPI name `openbench` is taken):

```bash
# from this checkout
pip install -e .

# or, without cloning first
pip install "git+https://github.com/minghinmatthewlam/openbench.git"

# future: pip install obench
```

Then use the umbrella CLI: `obench init`, `obench run`, `obench report`,
`obench doctor`, `obench validate`, `obench gate`, `obench compare`,
`obench publish`, `obench verify`, `obench pack`, `obench export`,
`obench import`, and `obench gateway validate|doctor|run|report|publish|verify`. Legacy
`python3 bench/run.py` (and friends) still forward with a deprecation note.
Versioned packs (`org/name@version`) are documented in
[`docs/task-packs.md`](docs/task-packs.md).

## Quickstart

Everything in the core harness runner is Python 3 standard library only — no
third-party Python dependencies. Real harnesses, Docker, and the Codex
open-model bridge have external CLI/tool requirements; see [`SETUP.md`](SETUP.md).

**1. Validate the tasks.** Confirms each checker fails on the untouched
workspace and passes on the golden solution (see "Task format"). This covers both
`tasks/` and imported tiers under `tasks-imported/`:

```
obench validate
# legacy: python3 validate_tasks.py
```

**2. Preflight.**

```
obench doctor
# legacy: python3 bench/doctor.py
```

For each harness it checks — spending no tokens — that the CLI is installed, its
auth/login or required key name is present, and the canonical model pin resolves
to the harness's own model string. A failing preflight exits nonzero.

**3. Run.** Pick harnesses and tasks. Start with the zero-cost `null` control to
confirm the plumbing, then add real harnesses:

```
# negative control — does nothing, so every task should fail (no tokens used)
obench run --harness null --task fix-failing-test,build-a-cli,make-it-run
# legacy: python3 bench/run.py --harness null --task ...

# a real harness, 3 trials per task
obench run --harness codex --task fix-failing-test,build-a-cli,make-it-run --trials 3
```

Multiple harnesses/tasks are comma-separated. The run loop is **resumable**: a
cell whose `run_id` already appears in `results/results.jsonl` is skipped, so you
can stop and re-run freely. Use `--force` to re-run existing cells. Full options:
`obench run --help`.

**4. Report.**

```
obench report
# legacy: python3 bench/report.py
```

Example output (from the `null` control on two tasks):

```
harness  fix-failing-test  make-it-run  overall   wilson95        mean_s  tokens
-------  ----------------  -----------  --------  --------------  ------  ------
null     0/1               0/1          0/2 (0%)  [0.000, 0.658]  0.00    -
```

## Task format

A core task is a directory under `tasks/<name>/`; an imported task is addressed
as `tasks-imported/<collection>/<name>/` and run with `--tasks-dir tasks-imported`:

```
instruction.md      what the harness is told (reads as a normal engineering request)
workspace/          starting files; copied fresh into a temp dir for every run
                    OR workspace.toml — git-ref materialization (see below)
checker.sh          grades the result; exit 0 = solved
solution/           golden files, used ONLY by validate_tasks.py (never shown to the harness)
checker_data/       optional: inputs/expected outputs the checker owns (kept out of workspace/)
```

Contract the runner honors for every cell:

- The starting workspace is materialized into a disposable temp dir; the harness
  edits that copy. The source under `tasks/` is never modified.
  - **Snapshot mode:** `workspace/` is copytree'd.
  - **Git mode:** `workspace.toml` exports a git ref via `git archive` (optional
    `subdir` / `setup` script). See [`docs/private-evals.md`](docs/private-evals.md).
  Provide exactly one of `workspace/` or `workspace.toml`.
- `checker.sh` runs with **cwd = the temp workspace copy** and the environment
  variable **`TASK_DIR`** set to the absolute task directory. Checkers reference
  their own data via `$TASK_DIR/checker_data/...` rather than a relative path, so
  they work regardless of cwd.
- Instructions never mention the checker, the solution, or that this is a
  benchmark.

#### Partial credit (the `SCORE:` contract)

A checker exit code is binary, but a checker MAY also emit a **`SCORE:`** line to
grade partial progress. The rules the runner applies:

- A checker may print `SCORE: <float 0.0–1.0>` to stdout. The **last parseable**
  such line wins; a malformed value is ignored (as if that line were absent), and
  values are clamped to `[0.0, 1.0]`.
- **Exit 0 is always a full pass** — `success = true` and `score` is coerced to
  `1.0` regardless of any `SCORE:` line.
- **Nonzero exit** — `success = false`, and `score` is the checker's `SCORE:`
  value if present, else `0.0`. This is how a task awards partial credit.
- A checker **timeout** records `score = 0.0`.

`SCORE:` is optional and backward compatible: a checker that never prints one
behaves exactly as before (pass → 1.0, fail → 0.0).

### Validation discipline

`validate_tasks.py` enforces that each checker is correctly polarized:

1. Run the checker against a freshly materialized workspace → it **must fail**
   (otherwise the task is scored solved before the agent does anything).
2. Run it against that workspace with `solution/` overlaid → it **must pass**
   (otherwise a correct answer would be rejected).

This catches the two ways a checker can silently lie about difficulty, and is
why expected outputs for data-driven tasks are generated *from* the golden
solution rather than written by hand. Current validated tiers are 8 core tasks,
11 Exercism imports, and 5 Terminal-Bench imports; each tier is reported and
scored separately.

## Adapters

Each harness is a module `obench/adapters/<name>.py` exposing `NAME`, a `MODELS`
map, and `run(instruction, workdir, model, timeout_s) -> dict`. The adapter maps
the canonical model name to the harness's own flags, runs the CLI headlessly with
cwd = `workdir`, enforces the timeout via `subprocess` (no `timeout` command —
macOS has none), and returns `completed` / `error` / `tokens` / `turns` / `cmd`.
`completed` means the CLI exited cleanly; it is **not** task success — the checker
decides that. Full contract: [`docs/adapter-contract.md`](docs/adapter-contract.md).
Adapters may optionally export `DOCTOR = {"cli", "auth"}` so `obench doctor`
picks them up without editing the doctor allowlist; third-party CLIs use
`--candidate` manifests instead (see [`docs/byo-harnesses.md`](docs/byo-harnesses.md)).

Auth is handled inside each adapter, read-only — the user's real config files are
never modified:

| Harness  | Frontier `gpt-5.5-medium` | Open models | Auth handling |
|----------|----------------------------|-------------|---------------|
| codex    | `gpt-5.5`, `model_reasoning_effort=medium` | Via foreground `bench/openmodel_bridge.sh` | Uses existing `codex` login for frontier; bridge/vendor keys for open models. |
| pi       | `gpt-5.5`, `--thinking medium` | Direct vendor endpoints | Isolated `HOME` (temp dir) with only `.pi/agent/auth.json` copied in, plus `--no-extensions`, so personal extensions never load. |
| opencode | `openai/gpt-5.5`, `--variant medium` | Direct vendor endpoints | Strips `OPENAI_API_KEY` from frontier child env to force subscription OAuth; open models use provider keys. |
| cursor   | `gpt-5.5-medium` (effort baked into name) | Not supported (closed menu) | Uses the existing `cursor-agent` login as-is. |
| devin    | `gpt-5.5` / configured pin (see caveats) | Not supported (closed menu) | Uses the existing `devin` login; latest data is flaky and excluded where noted. |
| claude   | Not supported by design | Direct Anthropic-compatible vendor endpoints | Open-model-only adapter; isolated config and vendor keys, never Anthropic subscription/OAuth. |

The built-in `null` adapter does nothing and reports `completed=True`. Because it
never edits the workspace, every task's checker fails — it is the benchmark's
**negative control**, and uses no tokens.

## Results

Findings from the milestone runs are in [`RESULTS.md`](RESULTS.md), with committed
datasets under [`data/`](data/). Local scratch runs stay under gitignored
`results/` unless intentionally promoted to `data/`.

`bench/run.py` appends one JSON object per line to `results/results.jsonl`. The
fields:

| Field          | Meaning                                                                 |
|----------------|-------------------------------------------------------------------------|
| `run_id`       | `harness:task:model:trialN` — the resumable identity of the cell        |
| `ts_iso`       | local timestamp when the cell ran                                       |
| `harness`      | adapter name (or `null`)                                                |
| `model`        | canonical model name (default `gpt-5.5-medium`)                         |
| `task`         | task directory name                                                     |
| `trial`        | 1-based trial index                                                     |
| `success`      | **the graded result** — `checker_exit == 0`                            |
| `completed`    | harness CLI exited cleanly (self-reported; not success)                 |
| `error`        | timeout / crash / adapter exception, else `null`                        |
| `wall_time_s`  | adapter wall-clock seconds                                              |
| `tokens`       | fresh tokens reported by the harness (uncached input + output), else `null` |
| `turns`        | turns reported by the harness, else `null`                             |
| `cmd`          | the command line executed (for auditability)                            |
| `checker_exit` | checker's integer exit code, or `"timeout"`                            |
| `exec_mode`    | `local` or `docker` (what actually ran, after any fallback)             |
| `score`        | graded score in `[0.0, 1.0]` (see the `SCORE:` contract); `1.0`/`0.0` for a plain pass/fail |
| `harness_version` | version string from the adapter's optional `version()`, `"builtin"` for `null`, else `null` |
| `timeout_s`    | per-cell adapter timeout cap used for this row (default `2400`)            |

Rows written before `score`, `harness_version`, or `timeout_s` existed simply
omit them; the report derives a score from `success` (`1.0`/`0.0`) for those.

`bench/report.py` reads that log and prints one row per harness: per-task
success (`x/n`), overall success with a Wilson 95% interval, **mean score**
(averaged over all trials, the discriminating number for partial-credit tasks),
mean wall-clock time, tokens-per-solve, and mean turns. `--efficiency` prints a
per-harness efficiency summary; `--results-path` points it at an alternate log.

Tokens-per-solve is basis-aware: it uses self-reported `tokens` when present,
otherwise the counting-proxy fresh total (`tokens_proxy_input_uncached +
tokens_proxy_output`) when `token_basis_proxy` is `proxy_measured`. Cache-read
is not mixed into that number. Proxy-derived figures are marked `*`; mixed-basis
tables print a warning. HTML publish/report cards use the same rule and badge
arms as `unmetered` / `self-reported` / `proxy-measured`.

### Transcripts are LOCAL-ONLY

Alongside each results row, the runner writes that cell's full agent transcript
to `transcripts/<results-file-stem>/<run_id>.txt` (a `transcripts/` sibling of
the results log; override with `--transcripts-dir`). These are the **raw,
unscrubbed** harness output and can contain your absolute home paths, username,
hostname, email, or secrets the agent echoed (API keys, tokens).

**Hard rule: transcripts are never published as-is.** `transcripts/` is
gitignored. `obench publish` builds a shareable comparison bundle (HTML card +
filtered results + provenance) and **refuses** to include transcripts; see
[`docs/publish.md`](docs/publish.md). Before sharing any transcript you must do
a manual review pass:

```bash
python3 -m obench.scrub transcripts/ --check          # REPORT potential PII (exit 1 if any)
python3 -m obench.scrub transcripts/ --out scrubbed/   # write scrubbed copies (originals untouched)
python3 -m obench.scrub scrubbed/ --check             # confirm the copies are clean (idempotent)
```

`scrub.py` replaces emails, home paths, the local username, hostnames, and
key/token-shaped strings with placeholders (`<EMAIL>`, `<HOME>`, `<USER>`,
`<HOST>`, `<REDACTED_KEY>`, …). It never modifies originals and over-redacts on
purpose — a false positive is cheap, a leaked secret is not. `--check` is a
report only; read it with your own eyes before trusting the scrubbed output.

### Reading the Wilson interval

Every success rate here is estimated from a handful of trials, so the point
estimate (say "2/3 = 67%") is noisy. The **Wilson 95% confidence interval** is
the range of true success rates consistent with what was observed — we are about
95% confident the harness's real success rate lies inside it. Two things to keep
in mind: with few trials the interval is **wide** (3 trials can easily span most
of 0–100%), so overlapping intervals mean the harnesses are *not* distinguishable
yet; and the interval shrinks as trials grow. Wilson (rather than the textbook
`p ± z·√(p(1−p)/n)`) is used because it stays inside `[0, 1]` and behaves sensibly
at 0/n and n/n, which the naive formula does not.

## Methodology & limitations

- **Same-model pinning (Track A).** All harnesses target `gpt-5.5-medium` so the
  harness is the variable — with the devin exception noted above.
- **Fresh workspace per run.** Every cell gets an untouched copy of the task
  workspace; runs cannot contaminate each other or the source tree.
- **Checker is the sole judge.** Success is `checker.sh` exit 0, never the
  harness's self-report. `validate_tasks.py` guarantees each checker actually
  discriminates a solved workspace from an unsolved one.
- **Negative control.** The `null` adapter should score 0% everywhere; a nonzero
  `null` success would indicate a broken (too-lenient) checker.
- **Isolation modes.** `--exec local` (default) runs on the host; `--exec docker`
  runs each cell in a fresh disposable container built from `obench/docker/`
  (`docker build -t openbench-harness:latest obench/docker`). The **same adapter
  module runs unchanged** in both modes — the container only adds isolation, with
  auth bind-mounted read-only at runtime (never baked into the image). Docker is
  fail-closed by default; pass `--docker-fallback` to opt into whole-run local
  homogenization when the daemon/image is unavailable (mixed lanes still abort).
- **Docker image is partial.** `codex`, `pi`, and open-model `claude` are installed
  and version-checked in the default image; `opencode`, `cursor`, and `devin` are
  behind `--build-arg INSTALL_UNVERIFIED=true` and their Linux installs are not
  yet confirmed. For those harnesses, use `--exec local` for now.
- **Sample size.** These are small tasks in small numbers; the current results
  are a plumbing/shakedown sample, not a verdict. Read the Wilson intervals, not
  the point estimates, and treat cross-harness gaps as real only when the
  intervals separate.

## License

OpenBench is available under the MIT License. See [LICENSE](LICENSE).
