# OpenBench — Agent Context

Read this first. It captures what this project is trying to become, so any agent
or contributor picks up the strategic context, not just the mechanics in
`README.md` / `WRITEUP.md`.

## Local execution context

Before changing benchmark execution or starting a run, read `agents.env` when
it exists. It is a gitignored, machine-local source of truth for where code is
developed and where benchmarks are executed. Never commit it or put credentials
in it.

For this installation, code changes belong in the laptop checkout. Benchmark
runs normally execute on the Mac Mini from an exact pushed commit. Do not edit
source on the Mini, do not launch from a dirty or stale checkout, and check for
active benchmark processes before starting another run.

## What OpenBench is

A benchmark framework for comparing coding-agent **harnesses** (codex, pi,
opencode, cursor, devin, claude, ...) — the CLI products that wrap a model in a
run loop, tool set, and permission policy. Tasks are self-contained
(`task.toml` + `instruction.md` + Docker environment + verifier); the verifier
is the sole judge.

## Execution ownership

The canonical path is `obench run [suite.toml]`. OpenBench compiles immutable
suite intent, stock/custom profiles, one exact Harbor job per task set, and
comparison-plan sidecars. Pinned Harbor owns task/trial execution, Docker
sandboxes, concurrency, retries, resume, locks, verifier execution, and ATIF
artifacts. OpenBench then validates/imports every intended job as one atomic
suite result, reconciles optional proxy evidence, and owns comparison,
statistics, publication policy, and site acceptance.

`obench legacy run` is compatibility only. Manual Harbor export, job-run, and
result-import commands remain diagnostics and migration tools; do not describe
them as the default workflow.

## Product goals (the two things we are building toward)

1. **Community harness flywheel.** Make OpenBench trivially importable and
   usable so third parties can add their own harnesses or harness variations
   and evaluate them against the stock adapters. If someone builds a better
   harness or feature, they should *want* to use OpenBench to prove it and post
   the results publicly — that showing-off loop is how the framework grows.
2. **Company/private-codebase evals.** More teams evaluate agents on their own
   codebases and use cases rather than general benchmarks. OpenBench should be
   easily installable inside a private repo so companies can benchmark
   harnesses and models on *their* tasks with the same rigor (checker polarity,
   token metering, Wilson CIs) as the public tiers.

## Our niche vs. the landscape (assessed Jul 2026)

- **Harbor** (Apache-2.0, Terminal-Bench 2.0's official harness) owns
  task/trial execution, Docker isolation, retries/resume, locks, verifiers,
  ATIF artifacts, cloud-scale evals, its dataset registry, and the TB
  leaderboard. OpenBench is the suite/comparison/evidence/publication layer.
- **Prime Intellect verifiers / Environments Hub** (MIT) owns RL+eval
  environments with a package hub (versioned wheels per environment, 1k+ envs).
  Their distribution mechanics (versioned installable task packs, hub
  publishing, seeded supply via bounties) are the playbook to copy; their v1 is
  drifting toward running real CLIs, so our neutrality matters.
- **OpenBench's defensible edge:** harness-vs-harness comparison under
  realistic conditions — same-model pinning, subscription/OAuth auth handling,
  counting-proxy token metering, polarity-validated checkers
  (`validate_tasks.py`), the null negative control, and the candidate
  admission gate (`obench/candidate_gate.py`, `docs/byo-harnesses.md`). Plus an
  ultra-light stdlib-only, files-plus-shell-checker contract that non-Python
  users and private repos can adopt without learning a framework API.

## Code map (where to look)

| Area | Path |
|------|------|
| CLI entry (`obench …`) | `obench/cli.py`, `obench/__main__.py` |
| Canonical suite run / sealing | `obench/suite_run.py`, `obench/suite.py` |
| Legacy cell runner | `obench/run.py` |
| Task workspace (snapshot + git archive) | `obench/workspace.py` |
| Checker polarity / validate | `obench/validate_tasks.py` |
| Task admission (structure, ownership, determinism) | `obench/admission_gate.py` |
| Candidate / BYO harness gate | `obench/candidate_gate.py`, `obench/candidates.py` |
| Report / stats / compare | `obench/report.py`, `obench/stats.py`, `obench/compare.py` |
| Publish / verify digests | `obench/publish.py` |
| Leaderboard site (harness + gateway) | `obench/site.py`, `obench/leaderboard.py`, `docs/site.md` |
| Counting proxy | `obench/proxy.py` |
| Harbor jobs/results/evidence | `obench/harbor_job.py`, `obench/harbor_results.py`, `obench/harbor_run.py` |
| Versioned packs (tasks + harness) | `obench/packs.py`, `docs/task-packs.md`, `docs/packs.json` |
| Stock adapters | `obench/adapters/` |
| Unit tests | `obench/tests/` |
| Tasks | `harbor-tasks/` (canonical), `tasks/` (historical compatibility), `.openbench/tasks/` (private-init) |

## Always-run CI (offline)

Match [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```bash
pip install -e .
python3 -m unittest discover -s obench/tests -v
obench validate --no-imported
obench validate --tasks-dir tasks-imported/terminal-bench
obench validate --tasks-dir tasks-imported/exercism
```

No live harness or model-API calls; stdlib-only. Docker-backed imported tiers,
including Terminal-Bench 2, require a separately provisioned validation lane.

## Dangerous zones

- **`obench/run.py` `ROW_FIELDS` / append / resume** — corrupt JSONL or dropped
  fields silently skew resume and published claims; keep append fsync + fail-closed
  corrupt-line handling.
- **Suite config/plan/result seals** — final Harbor config bytes, plan digest,
  suite manifest, task-set identity, and atomic result/run-manifest pair must
  remain mutually bound. Never write partial suite results.
- **`exec_mode` / docker fallback** — never mix docker and local cells in one
  comparable results file (`docker_fallback` defaults off).
- **Publish digests** — `task_content_digest` must cover oracle inputs
  (`checker_data/`); verify must FAIL on missing digests.
- **`obench/report.py` aggregates** — key by `(harness, model)`, not harness alone.
- **Auth / proxy / transcripts** — auth is read-only staging; transcripts are
  LOCAL-ONLY and never published unscrubbed (`obench/scrub.py`).
- **Legacy `bench/` tree** — shims may remain; new code and docs target `obench/`.

## Roadmap (priority order)

- **P0 — Package it. [DONE Jul 2026]** `pyproject.toml`, console entry points,
  PyPI name **`obench`** (`pip install obench`, `obench run ...`). Umbrella CLI
  (`run / report / doctor / validate / gate / compare / init / publish / verify /
  pack / …`).
  CWD discovery (`tasks/`, then `.openbench/tasks/`) when run outside the repo.
- **P0 — Arbitrary task roots. [DONE Jul 2026]** `validate_tasks.py` accepts
  custom task directories; `--preflight-smoke` picks a smoke task from the given
  root (prefers `make-it-run` when present).
- **P0 — `obench init` for private repos. [DONE Jul 2026]** `.openbench/`
  scaffold with `openbench.toml` config defaults; git-mode workspaces
  (`workspace.toml`: repo/ref/subdir/setup, `git archive` staging, resolved
  SHA recorded as `workspace_source` provenance); `docs/private-evals.md`.
- **P1 — Show-off loop. [PARTIAL Jul 2026]** `obench publish` / `obench verify`
  ship a shareable HTML card + provenance digests (`docs/publish.md`). Still
  open: community submission path onto the public site with CI re-verifying
  digests, and seeding by porting 2–3 popular harnesses ourselves.
- **P1 — Soften allowlists. [DONE Jul 2026]** `doctor.py` discovers optional
  adapter `DOCTOR` exports (pi migrated) and accepts `--candidate` preflight;
  proxy metering for manifests is declaration-driven (`base_url_env` +
  `proxy_route`); candidate auth persist-back defaults off with
  `persist_auth = true` opt-in. Docker image's fixed CLI set remains a follow-up.
- **P1 — Harbor-first suites. [DONE Aug 2026]** `obench run [suite.toml]`
  compiles one deterministic pinned Harbor job per task set, executes all jobs,
  fail-closed imports Harbor locks/results/verifier/ATIF/workspace/usage
  evidence, and atomically seals one suite JSONL plus public and local run
  records. Exact resume is idempotent; divergent output fails.
- **P2 — Versioned packs. [DONE Jul 2026]** Task and harness packs as
  versioned, installable-by-name artifacts (`org/pack@version`) via
  `obench pack` (`init` / `install` / `list` / `verify` / `publish-index`):
  local dir, git (`git archive`), or HTTPS zip/tarball — no custom package
  server (`docs/task-packs.md`). `pack.toml` `kind = "tasks"|"harness"`;
  layout `.openbench/packs/<org>/<name>/<version>/` with `pack_source.json`
  provenance (scheme-2 task digests or per-manifest `spec_sha256`). Harness
  packs resolve as `--candidate org/name[@version][:manifest]`. Static index
  `docs/packs.json` + site Packs section; seeds under
  `data/packs/openbench-core-smoke/` and `data/packs/openbench-aider/`.
  Still open: a community hub beyond the static JSON index.

## Non-goals

- No OpenBench-owned task scheduler, sandbox runtime, cloud backend, or RL
  training story. Harbor owns execution; OpenBench owns intent and evidence.
- Do not abandon the stdlib-only, files-plus-shell-checker task contract; it is
  the accessibility edge.

## Working conventions for agents

- The checker is the sole judge of success; never trust harness self-reports.
- Every new task must pass `validate_tasks.py` polarity (fails untouched,
  passes with `solution/` overlaid).
- Transcripts are local-only and never published unscrubbed (`obench/scrub.py`).
- Committed datasets live under `data/`; local scratch stays in gitignored
  `results/`.

## Cursor Cloud specific instructions

Setup is `pip install -e .` (stdlib-only, no third-party deps). The full
offline dev/CI loop is the "Always-run CI (offline)" block above — don't
duplicate those commands; use them as-is.

Non-obvious gotchas for this environment:

- **`obench` console script is not on PATH by default.** `pip install -e .`
  installs it to `~/.local/bin`, which is not on the default `PATH`. Either run
  `python3 -m obench ...` or prepend `export PATH="$HOME/.local/bin:$PATH"`.
- **The VM hostname is `cursor`, which breaks the publish/site/leaderboard
  tests.** `obench/scrub.py` treats the machine hostname as PII to redact, and
  `cursor` is also a stock harness name that legitimately appears in generated
  leaderboard/publish HTML. So the PII scanner trips (`[hostname] cursor` hit),
  causing ~79 errors + 1 failure in `test_publish` / `test_site` /
  `test_leaderboard` (and thus in a full `python3 -m unittest discover`). This
  is an environmental coincidence, not a code defect. To get a fully green
  suite, run under a non-`cursor` hostname first: `sudo hostname openbench-dev`
  (does not persist across VM restarts, so re-apply each session).
- **No linter is configured** (no ruff/flake8/black/mypy). The lint-equivalent
  correctness gate is `obench validate` (task checker polarity: workspace FAILs,
  solution PASSes).
- **Offline "hello world" / smoke of the run pipeline:** use the built-in
  `null` negative-control harness, which makes no external calls —
  `obench legacy run --task make-it-run --harness null --results-path results/results.jsonl`
  then `obench report --results-path results/results.jsonl`. It correctly
  records `success=False` (the checker is the sole judge). A real `obench run`
  suite additionally needs Harbor + Docker + installed harness CLIs + provider
  auth, none of which exist in the offline dev environment.
