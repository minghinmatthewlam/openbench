#!/usr/bin/env python3
"""Queue-based benchmark runner with retry budgets, arm pausing, and coverage.

Usage::

    obench matrix --spec experiments/specs/laguna-inkling.toml

The spec is a TOML file declaring arms (harness|harness-pack x model), task
groups, retry budgets, and run configuration.  The queue manager:

1.  Enumerates all planned cells (arm x task x trial).
2.  For each cell, checks a results JSONL for a SATISFIED row (failure_class
    NOT in the excluded-from-solve-rate set).  Cells whose only rows are
    excluded-class are RE-QUEUED against their retry budget.
3.  Applies exponential backoff for rate-limited retries.
4.  Pauses an ARM after N consecutive excluded results, revisiting at the end.
5.  Reports per-arm COVERAGE (satisfied / planned) and lists exhausted cells.

Persistent queue-state.json enables exact resume after kill or host restart.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Any

from . import run as bench_run
from . import failure_class as fc_mod
from .paths import PACKAGE_DIR, SOURCE_ROOT, default_results_path, default_tasks_dir

HERE = PACKAGE_DIR
REPO = SOURCE_ROOT

# Invoke the runner as a MODULE, not a file path: obench/run.py uses relative
# imports, so `python3 obench/run.py` dies with "attempted relative import
# with no known parent package" (observed on first live queue use).
DEFAULT_RUNNER_MODULE = "obench.run"
DEFAULT_MAX_CONSECUTIVE_EXCLUDED = 5
POLL_INTERVAL_S = 2.0

# Retry budgets: how many times a cell with a given failure class is re-queued.
DEFAULT_RETRY: dict[str, int] = {
    "infra": 2,
    fc_mod.STALLED: 1,
    "rate_limited": 3,
}
LEGACY_RETRY_ALIASES = {"stall": fc_mod.STALLED}
DEFAULT_RATE_LIMITED_BACKOFF_START_S = 60
DEFAULT_STALL_TIMEOUT = 600


class SpecError(ValueError):
    """Raised when the TOML spec is invalid or incomplete."""


def resolve_retry_budgets(retry_cfg: Any) -> dict[str, int]:
    """Return retry budgets keyed by canonical failure-class names.

    ``stall`` was the original public TOML key even though the runner has
    always emitted ``stalled``. Keep it as an input alias so existing matrix
    specs continue to work, but never retain it in the runtime lookup map.
    """
    budgets = dict(DEFAULT_RETRY)
    if not isinstance(retry_cfg, dict):
        return budgets

    for legacy, canonical in LEGACY_RETRY_ALIASES.items():
        if legacy in retry_cfg and canonical not in retry_cfg:
            budgets[canonical] = int(retry_cfg[legacy])
    for failure_class in DEFAULT_RETRY:
        if failure_class in retry_cfg:
            budgets[failure_class] = int(retry_cfg[failure_class])
    return budgets


def load_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a TOML spec file into a dict.

    Supports stdlib tomllib (Python 3.11+) and the ``toml`` backport.
    """
    import tomllib
    with open(path, "rb") as fh:
        spec = tomllib.load(fh)
    _validate_spec(spec, str(path))
    return spec


def _validate_spec(spec: dict[str, Any], source: str) -> None:
    """Raise SpecError for missing or invalid fields."""
    arms = spec.get("arm") or spec.get("arms")
    if not arms:
        raise SpecError(f"{source}: at least one [[arm]] is required")
    for i, arm in enumerate(arms):
        if not arm.get("harness"):
            raise SpecError(f"{source}: arm[{i}] missing 'harness'")
        if not arm.get("model"):
            raise SpecError(f"{source}: arm[{i}] missing 'model'")

    task_groups = spec.get("task_group") or spec.get("task_groups") or []
    if not task_groups:
        raise SpecError(f"{source}: at least one [[task_group]] is required")
    for i, tg in enumerate(task_groups):
        if not tg.get("tasks"):
            raise SpecError(f"{source}: task_group[{i}] missing 'tasks'")

    if not spec.get("results_path"):
        raise SpecError(f"{source}: 'results_path' is required")


def enumerate_arms(spec: dict[str, Any]) -> list[dict[str, str]]:
    """Return list of arm dicts from the spec."""
    return spec.get("arm") or spec.get("arms", [])


def enumerate_task_groups(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return list of task group dicts from the spec."""
    return spec.get("task_group") or spec.get("task_groups", [])


def resolve_tasks(spec: dict[str, Any], base_dir: str = "") -> list[str]:
    """Resolve task names from all task groups into a flat, deduplicated list."""
    seen: set[str] = set()
    tasks: list[str] = []
    for tg in enumerate_task_groups(spec):
        for name in tg.get("tasks", []):
            if name not in seen:
                seen.add(name)
                tasks.append(name)
    return tasks


def resolve_group_tasks_dir(
    group: dict[str, Any], spec: dict[str, Any], spec_dir: str
) -> str | None:
    """Resolve one task group's tasks directory, or None to let the runner decide.

    Precedence: the group's own ``tasks_dir`` beats the spec-level ``tasks_dir``.
    Both are interpreted relative to the spec file's directory. When neither is
    set we return None: the runner then resolves the tasks dir from config or
    discovery, and ``--tasks-dir`` is omitted entirely (see build_runner_command).
    This is what lets one spec mix a local core-tasks group with a docker
    terminal-bench group that each point at a different tree.
    """
    group_dir = group.get("tasks_dir") or spec.get("tasks_dir")
    if group_dir:
        return os.path.abspath(os.path.join(spec_dir, str(group_dir)))
    return None


def resolve_groups(
    spec: dict[str, Any], spec_dir: str, default_exec_mode: str
) -> list[dict[str, Any]]:
    """Resolve each task group's tasks, tasks_dir, and exec_mode.

    A group may set its own ``exec_mode`` (e.g. a docker terminal-bench group in
    an otherwise-local spec); it falls back to the spec-level default otherwise.
    """
    groups: list[dict[str, Any]] = []
    for tg in enumerate_task_groups(spec):
        groups.append({
            "tasks": list(tg.get("tasks", [])),
            "tasks_dir": resolve_group_tasks_dir(tg, spec, spec_dir),
            "exec_mode": tg.get("exec_mode") or default_exec_mode,
            # Per-group timeout override (None -> use the spec-global timeout).
            # A small algorithmic task and a large agentic task in the same spec
            # need very different budgets; one global timeout either starves the
            # big task or wastes wall-time on a hung small one.
            "timeout": tg.get("timeout"),
        })
    return groups


def expand_cells(
    arms: list[dict[str, str]],
    tasks: list[str],
    trials: int,
) -> list[dict[str, Any]]:
    """Enumerate every (arm, task, trial) cell with a stable run_id."""
    cells: list[dict[str, Any]] = []
    for arm in arms:
        harness = arm["harness"]
        model = arm["model"]
        for task in tasks:
            for trial_num in range(1, trials + 1):
                run_id = bench_run.make_run_id(harness, task, model, trial_num)
                cells.append({
                    "arm": f"{harness} x {model}",
                    "arm_idx": arms.index(arm),
                    "harness": harness,
                    "model": model,
                    "task": task,
                    "trial": trial_num,
                    "run_id": run_id,
                })
    return cells


def expand_cells_grouped(
    arms: list[dict[str, str]],
    groups: list[dict[str, Any]],
    trials: int,
) -> list[dict[str, Any]]:
    """Enumerate cells across per-task-group tasks_dir/exec_mode.

    Each cell carries the ``tasks_dir`` and ``exec_mode`` of its task group so a
    single spec can run a local core-tasks group and a docker terminal-bench
    group in the same queue. Cells are deduplicated by ``run_id`` (the same task
    name in two groups would collide; first group wins).
    """
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for arm in arms:
        harness = arm["harness"]
        model = arm["model"]
        arm_idx = arms.index(arm)
        for group in groups:
            for task in group.get("tasks", []):
                for trial_num in range(1, trials + 1):
                    run_id = bench_run.make_run_id(harness, task, model, trial_num)
                    if run_id in seen:
                        continue
                    seen.add(run_id)
                    cells.append({
                        "arm": f"{harness} x {model}",
                        "arm_idx": arm_idx,
                        "harness": harness,
                        "model": model,
                        "task": task,
                        "trial": trial_num,
                        "run_id": run_id,
                        "tasks_dir": group.get("tasks_dir"),
                        "exec_mode": group.get("exec_mode", "local"),
                        "timeout": group.get("timeout"),
                    })
    return cells


# ── Queue state ─────────────────────────────────────────────────────────

class QueueState:
    """Persistent queue state for resume-safe run management.

    Written to a JSON file after each cell completes so a killed or restarted
    ``obench matrix`` invocation resumes exactly where it left off.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                raw = self.path.read_text(encoding="utf-8")
                return json.loads(raw) if raw.strip() else {}
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.rename(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def update(self, mapping: dict[str, Any]) -> None:
        self._data.update(mapping)

    @property
    def data(self) -> dict[str, Any]:
        return self._data


# ── Retry and arm-state tracking ────────────────────────────────────────

class ArmState:
    """Per-arm retry and pause tracking."""

    def __init__(self, name: str, retry_budgets: dict[str, int] | None = None) -> None:
        self.name = name
        self.budgets: dict[str, int] = dict(retry_budgets or DEFAULT_RETRY)
        self.consecutive_excluded = 0
        self.paused = False
        # A SET, not a counter: an incrementing int carried across resumes
        # double-counted cells re-verified in a later pass and printed
        # coverage above 100% (observed: 'Total: 4/2 satisfied'). Coverage
        # is a property of WHICH cells have verdicts, so store which.
        self.satisfied_cells: set[str] = set()
        self.planned = 0
        self.exhausted_cells: list[str] = []
        # A harness/config error: the runner exited nonzero and wrote no row at
        # all. Distinct from an exhausted retry budget -- it means the arm is
        # mis-wired, so we stop it and surface the reason instead of retrying.
        self.config_error: str | None = None

    @property
    def satisfied(self) -> int:
        """Distinct cells with a verdict. Never exceeds ``planned``."""
        return len(self.satisfied_cells)

    def retry_budget(self, failure_class: str | None) -> int:
        """Re-queues allowed for a cell that failed with ``failure_class``.

        This governs RETRIES only. A cell with no prior row has no failure class
        and is not retrying, so callers must let the first attempt through
        without consulting this budget (see ``should_exhaust``).
        """
        if failure_class is None:
            return 0
        return self.budgets.get(failure_class, 0)

    def should_exhaust(self, failure_class: str | None, failed_attempts: int) -> bool:
        """True when a cell has used up its retries for this failure class.

        ``failed_attempts == 0`` is the first run of the cell, never an exhaustion:
        charging it against a retry budget marked every cell EXHAUSTED before
        it ever executed (observed on first live use: 0/9 satisfied).

        The first failed attempt is the original run, not a retry. Therefore a
        retry budget of one is exhausted only after two failed attempts.
        """
        if failed_attempts == 0:
            return False
        return failed_attempts > self.retry_budget(failure_class)

    def retries_remaining(self, failure_class: str | None, failed_attempts: int) -> int:
        """Retries left after ``failed_attempts`` including the original run."""
        retries_used = max(0, failed_attempts - 1)
        return max(0, self.retry_budget(failure_class) - retries_used)

    def record_excluded(self) -> None:
        self.consecutive_excluded += 1

    def record_included(self) -> None:
        self.consecutive_excluded = 0

    def record_satisfied(self, run_id: str) -> None:
        """Record a verdict and clear any stale exhaustion for the same cell."""
        self.satisfied_cells.add(run_id)
        self.exhausted_cells = [
            exhausted_id for exhausted_id in self.exhausted_cells
            if exhausted_id != run_id
        ]
        self.record_included()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "budgets": dict(self.budgets),
            "consecutive_excluded": self.consecutive_excluded,
            "paused": self.paused,
            "satisfied_cells": sorted(self.satisfied_cells),
            "planned": self.planned,
            "exhausted_cells": list(self.exhausted_cells),
            "config_error": self.config_error,
        }

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], retry_budgets: dict[str, int] | None = None,
    ) -> "ArmState":
        # The active spec is authoritative on resume. Without an explicit
        # override, normalize old persisted ``stall`` keys for standalone
        # callers loading legacy queue state.
        budgets = (
            dict(retry_budgets)
            if retry_budgets is not None
            else resolve_retry_budgets(d.get("budgets"))
        )
        self = cls(d["name"], budgets)
        self.consecutive_excluded = d.get("consecutive_excluded", 0)
        self.paused = d.get("paused", False)
        self.satisfied_cells = set(d.get("satisfied_cells") or [])
        self.planned = d.get("planned", 0)
        self.exhausted_cells = list(d.get("exhausted_cells", []))
        self.config_error = d.get("config_error")
        return self


# ── Cell satisfaction check ─────────────────────────────────────────────

def load_results_snapshot(
    results_path: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Load latest rows and trailing excluded-attempt counts per run ID."""
    rows: dict[str, dict[str, Any]] = {}
    excluded_attempts: dict[str, int] = {}
    if not os.path.isfile(results_path):
        return rows, excluded_attempts
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rid = row.get("run_id")
            if rid:
                rows[rid] = row
                if cell_is_satisfied(row):
                    excluded_attempts[rid] = 0
                else:
                    excluded_attempts[rid] = excluded_attempts.get(rid, 0) + 1
    return rows, excluded_attempts


def load_results_ids(results_path: str) -> dict[str, dict[str, Any]]:
    """Load results JSONL into {run_id: row} mapping (last row wins per id)."""
    rows, _excluded_attempts = load_results_snapshot(results_path)
    return rows


def cell_is_satisfied(row: dict[str, Any] | None) -> bool:
    """A cell is SATISFIED when it has a row whose failure_class is NOT excluded."""
    if row is None:
        return False
    # class_for_report unconditionally: `row.get("failure_class") or ...`
    # short-circuited on any stored value, so a cell stored as rate_limited was
    # re-queued even when the row's own fields prove the checker reached a
    # verdict (a 429 that recovered mid-run). That burns provider quota re-running
    # cells that are already judged -- 2 of the 7 laguna tb-mid "gaps" were this.
    fc = fc_mod.class_for_report(row)
    return fc not in fc_mod.EXCLUDED_FROM_SOLVE_RATE


def effective_failed_attempts(
    row: dict[str, Any] | None,
    persisted_attempts: int,
    result_attempts: int = 0,
) -> int:
    """Count a saved excluded result as its original failed attempt.

    A seeded results file can exist without queue state. Treating its row as
    attempt zero grants one more retry than the configured budget.
    """
    recorded = max(persisted_attempts, result_attempts)
    if recorded > 0:
        return recorded
    if row is not None and not cell_is_satisfied(row):
        return 1
    return 0


def row_failure_class(row: dict[str, Any] | None) -> str | None:
    """Return the row's effective canonical failure class, if available."""
    if row is None:
        return None
    fc = fc_mod.class_for_report(row)
    if fc in fc_mod.FAILURE_CLASSES:
        return fc
    return None


def load_cumulative_wall(results_path: str) -> dict[str, float]:
    """Total wall-clock seconds spent across ALL attempts of each run_id.

    Unlike ``load_results_snapshot`` (latest row per id), this sums ``wall_time_s``
    over every row so a cell's cumulative retry cost can be capped. A cell that
    keeps timing out under throttle otherwise re-burns a full timeout on every
    re-queue -- observed once as ~947 requests / ~9h on a single cell.
    """
    totals: dict[str, float] = {}
    if not os.path.isfile(results_path):
        return totals
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            rid = row.get("run_id")
            wall = row.get("wall_time_s")
            if rid and isinstance(wall, (int, float)):
                totals[rid] = totals.get(rid, 0.0) + float(wall)
    return totals


def wall_cap_exceeded(cumulative_wall_s: float, max_cell_wall_s: float | None) -> bool:
    """True when a cell has already spent its allowed cumulative wall time.

    ``max_cell_wall_s`` is ``None`` -> the cap is disabled and this never fires,
    so existing specs keep their current unbounded-retry behavior.
    """
    if max_cell_wall_s is None:
        return False
    return cumulative_wall_s >= max_cell_wall_s


# ── Backoff ──────────────────────────────────────────────────────────────

def backoff_for_failure(fc: str | None, attempt: int,
                        base_s: float = DEFAULT_RATE_LIMITED_BACKOFF_START_S) -> float:
    """Return the backoff delay in seconds before re-queueing a cell.

    Only rate_limited gets exponential backoff.  Other failure classes wait
    a fixed 10s to avoid hammering the harness.
    """
    if fc == "rate_limited":
        return base_s * (2 ** (attempt - 1))
    return 10.0


# ── Queue execution ─────────────────────────────────────────────────────

def build_runner_command(
    cell: dict[str, Any],
    results_path: str,
    tasks_dir: str | None,
    timeout: int,
    stall_timeout: int | None,
    exec_mode: str,
    allow_version_drift: bool = False,
) -> list[str]:
    """Build the ``obench run`` subprocess argv for one cell.

    ``tasks_dir`` and ``exec_mode`` are DEFAULTS: a cell carrying its own
    ``tasks_dir``/``exec_mode`` (set by ``expand_cells_grouped`` for per-task-group
    execution) overrides them. When neither the cell nor the caller supplies a
    tasks dir, ``--tasks-dir`` is OMITTED so the runner resolves it via config
    or discovery -- emitting ``--tasks-dir`` with an empty/None value made the
    runner fail with no row written, which the queue then misread as an
    unclassifiable exhausted cell (observed on the first live coverage-gap spec).
    """
    effective_tasks_dir = cell.get("tasks_dir", tasks_dir)
    effective_exec_mode = cell.get("exec_mode", exec_mode)
    # A cell carrying its own timeout (from a task group's override) wins over
    # the spec-global default; None/0 falls back to the caller's timeout.
    effective_timeout = cell.get("timeout") or timeout
    cmd = [sys.executable, "-m", DEFAULT_RUNNER_MODULE]
    cmd.extend([
        "--force",  # always re-run even if run_id exists (prior excluded rows)
        "--harness", cell["harness"],
        "--model", cell["model"],
        "--task", cell["task"],
        "--trial", str(cell["trial"]),
        "--timeout", str(effective_timeout),
        "--results-path", results_path,
    ])
    if effective_tasks_dir:
        cmd.extend(["--tasks-dir", effective_tasks_dir])
    if effective_exec_mode == "docker":
        cmd.extend(["--exec", "docker"])
    if allow_version_drift:
        # Uniform waiver across every arm: a local run against a host CLI newer
        # than the Dockerfile pin. Each row records version_drift=true, so the
        # off-pin state is annotated rather than silently bumping the pin (which
        # is coupled to tests, docker image-contexts, and docs).
        cmd.append("--allow-version-drift")
    if stall_timeout is not None:
        cmd.extend(["--stall-timeout", str(stall_timeout)])
    # Enable proxy for stall-kill support (required for stall-timeout to work)
    if stall_timeout is not None:
        cmd.append("--proxy")
    return cmd


STDERR_TAIL_CHARS = 4000


def _last_meaningful_line(text: str | None) -> str:
    """Return the last non-blank line of ``text`` (usually the error summary)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line[:500]
    return ""


def run_runner(cmd: list[str], timeout_s: float | None = None) -> tuple[int, str]:
    """Run one obench runner invocation; return ``(exit_code, stderr_tail)``.

    stderr is captured to a temp file (not a PIPE) so a long, chatty runner
    cannot deadlock on a full pipe buffer, while stdout stays inherited so live
    progress still streams. The returned tail is only meaningful for diagnosing
    a failed invocation -- the queue logs it when a nonzero exit wrote no row.
    """
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    try:
        proc = subprocess.Popen(cmd, start_new_session=True, stderr=stderr_file)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 15)  # SIGTERM
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
                proc.wait()
            rc = -1
        try:
            stderr_file.seek(0)
            data = stderr_file.read()
        except OSError:
            data = b""
        tail = data[-STDERR_TAIL_CHARS:].decode("utf-8", "replace") if data else ""
        return rc, tail
    finally:
        stderr_file.close()


# ── Main ────────────────────────────────────────────────────────────────

def run_matrix(spec: dict[str, Any], spec_dir: str, cwd: str) -> int:
    """Execute the full matrix queue and return exit code (0 = full coverage)."""
    results_path = os.path.abspath(os.path.join(spec_dir, spec.get("results_path", "results.jsonl")))
    timeout = spec.get("timeout", 2400)
    exec_mode = spec.get("exec_mode", "local")
    trials = spec.get("trials", 1)
    # Cap the cumulative wall time a single cell may spend across retries. None
    # (default) leaves retries bounded only by the per-class retry budget, which
    # lets a throttle-dominated cell re-burn a full timeout on every re-queue.
    max_cell_wall_s = spec.get("max_cell_wall_s")
    allow_version_drift = bool(spec.get("allow_version_drift", False))
    stall_timeout = spec.get("stall_timeout") or (
        int(os.environ.get("OPENBENCH_STALL_TIMEOUT", "0")) or None
    )
    if stall_timeout is None and spec.get("proxy", False):
        stall_timeout = DEFAULT_STALL_TIMEOUT

    retry_cfg = spec.get("retry", {})
    retry_budgets = resolve_retry_budgets(retry_cfg)
    rate_limited_backoff = float(
        retry_cfg.get("rate_limited_backoff_start", DEFAULT_RATE_LIMITED_BACKOFF_START_S)
        if isinstance(retry_cfg, dict) else DEFAULT_RATE_LIMITED_BACKOFF_START_S
    )
    max_consecutive_excluded = int(
        spec.get("max_consecutive_excluded", DEFAULT_MAX_CONSECUTIVE_EXCLUDED)
    )

    groups = resolve_groups(spec, spec_dir, exec_mode)
    tasks = resolve_tasks(spec, spec_dir)
    arms = enumerate_arms(spec)
    all_cells = expand_cells_grouped(arms, groups, trials)

    # Quick sanity check: only groups with a resolved tasks_dir can be checked
    # here. Groups that omit tasks_dir defer resolution to the runner (config or
    # discovery), so we can't pre-verify them and must not guess a path.
    missing: list[str] = []
    for group in groups:
        td = group.get("tasks_dir")
        if not td:
            continue
        for task in group["tasks"]:
            leaf = task.split("/")[0]
            if (not os.path.isfile(os.path.join(td, task, "instruction.md"))
                    and not os.path.isfile(os.path.join(td, leaf, "instruction.md"))):
                missing.append(f"{task} (under {td})")
    if missing:
        print(
            f"ERROR: {len(missing)} task(s) not found: " + ", ".join(missing[:5]),
            file=sys.stderr,
        )
        return 1

    # ── Initialize queue state ──────────────────────────────────────────
    # Resolve ledger_dir against the spec dir just like results_path. A bare
    # relative ledger_dir was previously taken as-is (relative to CWD), so the
    # persistent queue-state.json landed in a shared location and stale
    # satisfied/planned counts from an unrelated run leaked back in on resume.
    ledger_dir = spec.get("ledger_dir")
    if ledger_dir:
        qdir = os.path.abspath(os.path.join(spec_dir, str(ledger_dir)))
    else:
        qdir = os.path.join(os.path.dirname(results_path), ".matrix-queue")
    queue_state_path = os.path.join(str(qdir), "queue-state.json")
    os.makedirs(str(qdir), exist_ok=True)
    state = QueueState(queue_state_path)

    # Per-arm state (restored or fresh)
    arm_states_raw = state.get("arm_states", {})
    arm_states: dict[str, ArmState] = {}
    for arm in arms:
        name = f"{arm['harness']} x {arm['model']}"
        if name in arm_states_raw:
            arm_states[name] = ArmState.from_dict(
                arm_states_raw[name], retry_budgets=retry_budgets)
        else:
            as_ = ArmState(name, retry_budgets)
            as_.planned = sum(1 for c in all_cells if c["arm"] == name)
            arm_states[name] = as_

    # Pending cell queue: list of (arm_name, cell) tuples.
    # Restore from saved state if available.
    pending_raw = state.get("pending", [])
    by_run_id = {c["run_id"]: c for c in all_cells}
    if pending_raw:
        pending: list[tuple[str, int, dict[str, Any]]] = [
            (p[0], p[1], by_run_id[p[2]]) for p in pending_raw if p[2] in by_run_id
        ]
        # A saved queue can name cells the CURRENT spec no longer contains --
        # narrowing a spec (dropping a task group) while its ledger holds state
        # is the normal way to refocus a campaign. The old code did
        # next(c for c in all_cells if ...) and died on a bare StopIteration
        # with no indication that the spec had changed. Drop the stale entries
        # and say so; cells that are genuinely done are still skipped via the
        # results file, which is the source of truth for completion.
        dropped = len(pending_raw) - len(pending)
        if dropped:
            print(f"  spec changed since the last run: dropping {dropped} queued "
                  f"cell(s) no longer in it, re-planning the rest")
        known = {p[2] for p in pending_raw}
        added = [c for c in all_cells if c["run_id"] not in known]
        if added:
            print(f"  spec changed since the last run: adding {len(added)} new cell(s)")
            pending += [(c["arm"], c["arm_idx"], c) for c in added]
    else:
        pending = [(c["arm"], c["arm_idx"], c) for c in all_cells]

    # Track retry counts per run_id
    retry_counts: dict[str, int] = dict(state.get("retry_counts", {}))

    paused_arms: list[tuple[str, int, dict[str, Any]]] = []
    visit_paused = False

    print(
        f"Matrix queue: {len(all_cells)} planned cells, "
        f"{len(arms)} arm(s), {len(tasks)} task(s), {trials} trial(s)"
    )

    while pending or (paused_arms and not visit_paused):
        # If we exhausted the main queue and have paused arms, revisit them.
        if not pending and paused_arms and not visit_paused:
            print(f"\n-- Revisiting {len(paused_arms)} paused arm(s) --")
            pending = list(paused_arms)
            paused_arms = []
            visit_paused = True

        arm_name, arm_idx, cell = pending.pop(0)
        as_ = arm_states[arm_name]
        run_id = cell["run_id"]

        # Check if cell is already satisfied
        existing, result_attempt_counts = load_results_snapshot(results_path)
        row = existing.get(run_id)
        if cell_is_satisfied(row):
            as_.record_satisfied(run_id)
            print(f"    SATISFIED {run_id} (coverage {as_.satisfied}/{as_.planned})")
            state.set("arm_states", {n: a.to_dict() for n, a in arm_states.items()})
            state.save()
            continue

        # Check retry budget
        attempt = effective_failed_attempts(
            row,
            retry_counts.get(run_id, 0),
            result_attempt_counts.get(run_id, 0),
        )
        fc = row_failure_class(row)
        budget = as_.retry_budget(fc)
        if as_.should_exhaust(fc, attempt):
            as_.exhausted_cells.append(run_id)
            print(f"    EXHAUSTED {run_id} (fc={fc} attempts={attempt} budget={budget})")
            state.set("arm_states", {n: a.to_dict() for n, a in arm_states.items()})
            state.save()
            continue

        # Backoff if re-trying
        if attempt > 0:
            delay = backoff_for_failure(fc, attempt, rate_limited_backoff)
            print(f"    BACKOFF {run_id} attempt={attempt}/{budget} fc={fc} wait={delay:.0f}s")
            time.sleep(delay)

        # Run the cell. A per-group timeout override (carried on the cell) sets
        # both the runner's --timeout and this outer kill budget.
        print(f"    RUN    {run_id}", flush=True)
        cell_timeout = cell.get("timeout") or timeout
        cmd = build_runner_command(
            cell, results_path, None, timeout, stall_timeout, exec_mode,
            allow_version_drift=allow_version_drift)
        rc, stderr_tail = run_runner(cmd, cell_timeout + 60)

        if rc != 0:
            print(f"    WARN runner exit={rc} for {run_id}", file=sys.stderr)

        # Re-check satisfaction
        existing, result_attempt_counts = load_results_snapshot(results_path)
        row = existing.get(run_id)
        if cell_is_satisfied(row):
            as_.record_satisfied(run_id)
            print(f"    SATISFIED {run_id} (coverage {as_.satisfied}/{as_.planned})")
        elif row is None:
            # The runner wrote NO row for this cell. A completed cell -- even a
            # failing one -- always appends a classified row, so an absent row
            # means the runner itself never ran the cell: a bad --tasks-dir, an
            # argparse error, a missing adapter, an import crash. This is a
            # harness/config error, not an unclassifiable capability result.
            # Burning retries against fc=None here is exactly what silently
            # declared cells EXHAUSTED on the first live coverage-gap spec, so we
            # STOP the arm and surface the runner's own stderr instead.
            reason = _last_meaningful_line(stderr_tail) or f"exit={rc}, no row written"
            as_.config_error = f"exit={rc}: {reason}"
            print(
                f"    CONFIG-ERROR {run_id}: runner exit={rc} wrote no row -- "
                f"stopping arm {arm_name!r}. Reason: {reason}",
                file=sys.stderr,
            )
            if stderr_tail.strip():
                print("    --- runner stderr (tail) ---", file=sys.stderr)
                print(stderr_tail.rstrip(), file=sys.stderr)
                print("    --- end runner stderr ---", file=sys.stderr)
            # Drop every remaining cell for this arm from the queue and the
            # paused list; there is no point retrying a mis-wired arm.
            pending = [(n, i, c) for n, i, c in pending if n != arm_name]
            paused_arms = [(n, i, c) for n, i, c in paused_arms if n != arm_name]
        else:
            new_fc = row_failure_class(row)
            retry_counts[run_id] = max(
                attempt + 1, result_attempt_counts.get(run_id, 0))
            if new_fc is not None and new_fc in fc_mod.EXCLUDED_FROM_SOLVE_RATE:
                as_.record_excluded()
                # Re-queue if budget allows
                budget_remaining = as_.retries_remaining(
                    new_fc, retry_counts.get(run_id, 0))
                # A throttle-dominated cell classifies rate_limited yet burns a
                # full timeout every attempt; the per-class retry budget alone
                # lets it re-burn that timeout many times. Stop once its
                # cumulative wall time crosses the cap, even with budget left.
                cell_wall = load_cumulative_wall(results_path).get(run_id, 0.0)
                if wall_cap_exceeded(cell_wall, max_cell_wall_s):
                    if run_id not in as_.exhausted_cells:
                        as_.exhausted_cells.append(run_id)
                    print(f"    EXHAUSTED {run_id} (fc={new_fc} wall cap: "
                          f"{cell_wall:.0f}s >= {max_cell_wall_s}s)")
                elif budget_remaining > 0:
                    pending.insert(0, (arm_name, arm_idx, cell))
                    print(f"    RE-QUEUED {run_id} (fc={new_fc} retry={budget_remaining} left)")
                else:
                    if run_id not in as_.exhausted_cells:
                        as_.exhausted_cells.append(run_id)
                    print(f"    EXHAUSTED {run_id} (fc={new_fc} retry budget exhausted)")

        # Arm pause check (skip if the arm just hit a config error and stopped)
        if as_.config_error is None and as_.consecutive_excluded >= max_consecutive_excluded:
            if not as_.paused:
                as_.paused = True
                # Move all remaining cells for this arm to paused list
                arm_cells = [(n, i, c) for n, i, c in pending if n == arm_name]
                pending = [(n, i, c) for n, i, c in pending if n != arm_name]
                paused_arms.extend(arm_cells)
                print(f"    PAUSED arm={arm_name} after {as_.consecutive_excluded} consecutive excluded")

        # Persist state
        state.set("arm_states", {n: a.to_dict() for n, a in arm_states.items()})
        state.set("retry_counts", retry_counts)
        remaining_pending = []
        for n, i, c in pending:
            remaining_pending.append([n, i, c["run_id"]])
        state.set("pending", remaining_pending)
        state.save()

    # ── Final summary ───────────────────────────────────────────────────
    total_planned = len(all_cells)
    total_satisfied = sum(a.satisfied for a in arm_states.values())
    exhausted = [(n, a.exhausted_cells) for n, a in arm_states.items() if a.exhausted_cells]
    paused_final = [n for n, a in arm_states.items() if a.paused]

    print("\n" + "=" * 60)
    print("MATRIX COVERAGE REPORT")
    print("=" * 60)
    for arm in arms:
        name = f"{arm['harness']} x {arm['model']}"
        as_ = arm_states[name]
        pct = (as_.satisfied / as_.planned * 100) if as_.planned > 0 else 0
        marker = " [PAUSED]" if as_.paused else ""
        if as_.config_error is not None:
            marker = " [CONFIG-ERROR]"
        exhausted_count = len(as_.exhausted_cells)
        exhausted_mark = f" {exhausted_count} exhausted" if exhausted_count else ""
        print(f"  {name}: {as_.satisfied}/{as_.planned} ({pct:.1f}%){marker}{exhausted_mark}")

    config_errored = [(n, a.config_error) for n, a in arm_states.items()
                      if a.config_error is not None]
    if config_errored:
        print("\nArms stopped by a harness/config error (not a retry exhaustion):")
        for arm_name, reason in config_errored:
            print(f"  {arm_name}: {reason}")

    if exhausted:
        print("\nExhausted cells (retry budget depleted):")
        for arm_name, cells in exhausted:
            for c in cells:
                print(f"  {arm_name}: {c}")

    failed_arms = [n for n, a in arm_states.items()
                   if a.exhausted_cells or a.config_error is not None]
    exit_code = 1 if failed_arms else 0
    print(f"\nTotal: {total_satisfied}/{total_planned} satisfied")
    print(f"Exit: {exit_code}")
    return exit_code


def missing_task_images(spec, spec_dir, docker_runner=None):
    """Pinned task images the spec needs that this host lacks at that digest.

    A spec whose tasks pin per-task images is unrunnable on a host that does not
    hold those exact digests: every cell dies immediately with "cannot inspect
    Docker image". That happened on a live re-run -- the host had 23
    ``openbench-tb2`` images but none at the pinned digests, and the queue burned
    the launch to discover it. The existing preflight checks CLI pins, not task
    images, so nothing caught it.

    Returns a list of ``(task, image)``; empty when the host can run the spec.
    """
    import tomllib
    runner = docker_runner or (lambda ref: subprocess.run(
        ["docker", "image", "inspect", ref],
        capture_output=True, text=True, timeout=30).returncode == 0)
    missing = []
    for group in spec.get("task_group") or []:
        tasks_dir = resolve_group_tasks_dir(group, spec, spec_dir)
        # A group that omits tasks_dir defers resolution to the runner (config or
        # discovery) and pins no per-task docker image, so there is nothing to
        # preflight here. Skipping it also avoids os.path.join(None, ...), which
        # otherwise crashes an entirely local spec before any cell runs.
        if tasks_dir is None:
            continue
        for task in group.get("tasks") or []:
            toml_path = os.path.join(tasks_dir, task, "task.toml")
            if not os.path.isfile(toml_path):
                continue
            try:
                with open(toml_path, "rb") as fh:
                    image = tomllib.load(fh).get("docker_image")
            except (OSError, ValueError):
                continue
            if image and not runner(image):
                missing.append((task, image))
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Matrix queue: retry-aware benchmark runner for OpenBench.")
    parser.add_argument("--spec", required=True,
                        help="TOML spec file defining arms, task groups, retry budgets")
    args = parser.parse_args(argv)

    spec_path = os.path.abspath(args.spec)
    if not os.path.isfile(spec_path):
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return 1

    spec_dir = os.path.dirname(spec_path)
    cwd = os.getcwd()

    try:
        spec = load_spec(spec_path)
    except SpecError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    missing = missing_task_images(spec, spec_dir)
    if missing:
        print("ERROR: pinned task images are not present at their exact digest "
              "on this host:", file=sys.stderr)
        for task, image in missing:
            print(f"  {task}: {image}", file=sys.stderr)
        print("Every cell would die on 'cannot inspect Docker image'. Build or "
              "pull them, or run on a host that has them.", file=sys.stderr)
        return 1

    return run_matrix(spec, spec_dir, cwd)


if __name__ == "__main__":
    raise SystemExit(main())
