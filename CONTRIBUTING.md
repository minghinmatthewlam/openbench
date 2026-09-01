# Contributing

Thanks for your interest in contributing to OpenBench. There are two main ways
to contribute, each with its own contract.

OpenBench is **Python 3 standard library only** — there is nothing to install
beyond the local package. Before opening a PR, run the same two checks CI runs
(both are offline and use no credentials):

```
pip install -e .
python3 -m unittest discover -s obench/tests -v
obench validate
```

## 1. Contribute a task

New benchmark tasks are the highest-value contribution. A task is a small,
**original** coding problem with a checker that fails on the unsolved workspace
and passes on a golden solution. The full contract — directory layout, the
`SCORE:` partial-credit line, the fail-on-workspace/pass-on-solution discipline,
the original-code-only rule, and how CI validates it — is in
**[CONTRIBUTING-TASKS.md](CONTRIBUTING-TASKS.md)**.

You do **not** need API keys or to run any harness to contribute a task; a clean
`obench validate` is the bar. Maintainers pilot difficulty post-merge.

## 2. Contribute a harness adapter

To add a coding-agent harness to the comparison, write an adapter module
`obench/adapters/<name>.py` that maps the canonical model name to that harness's
CLI flags, runs it headlessly, and returns a normalized result dict. The exact
interface (`NAME`, `MODELS`, `run(...)`, the optional `version()`, timeout and
auth rules) is specified in **[obench/ADAPTER_SPEC.md](obench/ADAPTER_SPEC.md)**;
the existing adapters in `obench/adapters/` are working references. Auth must be
handled inside the adapter, read-only — never modify the user's real config
files.

To add an open / BYO model route to the existing adapters you do not need to
touch adapter code at all: write a `.openbench/open_models.toml` registry as
described in the same spec.

## Pull requests

- Keep changes focused and reviewable.
- Include tests or verification steps for behavior changes (the runner, report,
  and adapters have unit tests under `obench/tests/`).
- Update documentation when changing user-facing behavior or project workflows.
- Both CI checks above must pass.

## Issues

When filing an issue, include:

- The expected behavior.
- The actual behavior.
- Reproduction steps or relevant context.

See also the [Code of Conduct](CODE_OF_CONDUCT.md).
