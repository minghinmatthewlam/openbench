#!/usr/bin/env python3
"""Tests for the matrix queue (retry budgets, arm pausing, coverage)."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

from obench import matrix_queue as mq
from obench import failure_class as fc_mod
from obench import run as bench_run


class SpecLoadingTests(unittest.TestCase):
    """Verify TOML spec parsing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mq_spec_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_spec(self, content):
        path = os.path.join(self.tmp, "spec.toml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_valid_spec(self):
        path = self.make_spec("""
results_path = "results.jsonl"
timeout = 2400
trials = 3

[[arm]]
harness = "pi"
model = "laguna-s-2.1"

[[arm]]
harness = "aider"
model = "inkling"

[[task_group]]
tasks = ["hello-world", "fibonacci"]
""")
        spec = mq.load_spec(path)
        self.assertEqual(len(spec["arm"]), 2)
        self.assertEqual(len(spec["task_group"]), 1)
        self.assertEqual(spec["arm"][0]["harness"], "pi")
        self.assertEqual(spec["arm"][1]["model"], "inkling")

    def test_rejects_missing_arms(self):
        path = self.make_spec("""
results_path = "results.jsonl"
[[task_group]]
tasks = ["hello"]
""")
        with self.assertRaises(mq.SpecError):
            mq.load_spec(path)

    def test_rejects_missing_harness(self):
        path = self.make_spec("""
results_path = "results.jsonl"
[[arm]]
model = "x"
[[task_group]]
tasks = ["hello"]
""")
        with self.assertRaises(mq.SpecError):
            mq.load_spec(path)

    def test_rejects_missing_task_groups(self):
        path = self.make_spec("""
results_path = "results.jsonl"
[[arm]]
harness = "pi"
model = "x"
""")
        with self.assertRaises(mq.SpecError):
            mq.load_spec(path)

    def test_rejects_missing_results_path(self):
        path = self.make_spec("""
[[arm]]
harness = "pi"
model = "x"
[[task_group]]
tasks = ["hello"]
""")
        with self.assertRaises(mq.SpecError):
            mq.load_spec(path)

    def test_custom_retry_budgets(self):
        path = self.make_spec("""
results_path = "results.jsonl"
[retry]
infra = 5
stalled = 2
rate_limited = 1

[[arm]]
harness = "pi"
model = "x"
[[task_group]]
tasks = ["hello"]
""")
        spec = mq.load_spec(path)
        retry = spec.get("retry", {})
        self.assertEqual(retry["infra"], 5)
        self.assertEqual(retry["stalled"], 2)
        self.assertEqual(retry["rate_limited"], 1)

    def test_expand_cells(self):
        arms = [{"harness": "pi", "model": "a"}, {"harness": "codex", "model": "b"}]
        tasks = ["t1", "t2"]
        cells = mq.expand_cells(arms, tasks, 2)
        self.assertEqual(len(cells), 8)  # 2 arms * 2 tasks * 2 trials
        ids = [c["run_id"] for c in cells]
        self.assertIn("pi:t1:a:trial1", ids)
        self.assertIn("codex:t2:b:trial2", ids)

    def test_expand_cells_run_id_format(self):
        arms = [{"harness": "pi", "model": "laguna-s-2.1"}]
        cells = mq.expand_cells(arms, ["hello"], 1)
        self.assertEqual(cells[0]["run_id"], "pi:hello:laguna-s-2.1:trial1")


class QueueStateTests(unittest.TestCase):
    """Verify persistent queue state JSON."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mq_state_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_state_is_empty(self):
        path = os.path.join(self.tmp, "queue-state.json")
        state = mq.QueueState(path)
        self.assertEqual(state.data, {})

    def test_save_and_restore(self):
        path = os.path.join(self.tmp, "queue-state.json")
        state = mq.QueueState(path)
        state.set("hello", "world")
        state.save()

        state2 = mq.QueueState(path)
        self.assertEqual(state2.get("hello"), "world")

    def test_arm_state_roundtrip(self):
        a = mq.ArmState("pi")
        a.satisfied_cells.update(f"c{i}" for i in range(5))
        a.planned = 10
        a.consecutive_excluded = 3
        a.exhausted_cells = ["pi:t1:m:trial1"]
        restored = mq.ArmState.from_dict(a.to_dict())
        self.assertEqual(restored.name, "pi")
        self.assertEqual(restored.satisfied, 5)
        self.assertEqual(restored.planned, 10)
        self.assertEqual(restored.consecutive_excluded, 3)
        self.assertEqual(restored.exhausted_cells, ["pi:t1:m:trial1"])
        self.assertFalse(restored.paused)


class CellSatisfactionTests(unittest.TestCase):
    """Verify cell satisfaction logic."""

    def test_solved_is_satisfied(self):
        self.assertTrue(mq.cell_is_satisfied({"failure_class": "solved"}))

    def test_wrong_answer_is_satisfied(self):
        self.assertTrue(mq.cell_is_satisfied({"failure_class": "wrong_answer"}))

    def test_infra_is_not_satisfied(self):
        self.assertFalse(mq.cell_is_satisfied({"failure_class": "infra"}))

    def test_rate_limited_is_not_satisfied(self):
        self.assertFalse(mq.cell_is_satisfied({"failure_class": "rate_limited"}))

    def test_stalled_is_not_satisfied(self):
        self.assertFalse(mq.cell_is_satisfied({"failure_class": "stalled"}))

    def test_timeout_is_satisfied(self):
        self.assertTrue(mq.cell_is_satisfied({"failure_class": "timeout"}))

    def test_no_row_is_not_satisfied(self):
        self.assertFalse(mq.cell_is_satisfied(None))

    def test_no_failure_class_derived(self):
        """A row with no failure_class uses class_for_report to derive it."""
        row = {"success": False, "checker_exit": 1, "completed": True,
               "error": "something", "tokens": 100}
        # Should derive as wrong_answer (satisfied)
        self.assertTrue(mq.cell_is_satisfied(row))


class LoadResultsIdsTests(unittest.TestCase):
    """Verify results JSONL loading."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mq_results_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_file_returns_empty(self):
        path = os.path.join(self.tmp, "results.jsonl")
        self.assertEqual(mq.load_results_ids(path), {})

    def test_loads_rows_keyed_by_run_id(self):
        path = os.path.join(self.tmp, "results.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": "a", "failure_class": "solved"}) + "\n")
            fh.write(json.dumps({"run_id": "b", "failure_class": "infra"}) + "\n")
        rows = mq.load_results_ids(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows["a"]["failure_class"], "solved")
        self.assertEqual(rows["b"]["failure_class"], "infra")

    def test_skips_corrupt_lines(self):
        path = os.path.join(self.tmp, "results.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write(json.dumps({"run_id": "a", "failure_class": "solved"}) + "\n")
        rows = mq.load_results_ids(path)
        self.assertEqual(len(rows), 1)

    def test_last_row_wins_on_duplicate_id(self):
        path = os.path.join(self.tmp, "results.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": "a", "failure_class": "infra"}) + "\n")
            fh.write(json.dumps({"run_id": "a", "failure_class": "solved"}) + "\n")
        rows = mq.load_results_ids(path)
        self.assertEqual(rows["a"]["failure_class"], "solved")

    def test_snapshot_counts_only_trailing_excluded_attempts(self):
        path = os.path.join(self.tmp, "results.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"run_id": "a", "failure_class": "infra"}) + "\n")
            fh.write(json.dumps({"run_id": "a", "failure_class": "solved"}) + "\n")
            fh.write(json.dumps({"run_id": "a", "failure_class": "stalled"}) + "\n")
            fh.write(json.dumps({"run_id": "a", "failure_class": "stalled"}) + "\n")
        rows, attempts = mq.load_results_snapshot(path)
        self.assertEqual(rows["a"]["failure_class"], "stalled")
        self.assertEqual(attempts["a"], 2)


class RetryBudgetTests(unittest.TestCase):
    """Verify retry budget logic."""

    def test_default_budgets(self):
        arm = mq.ArmState("pi")
        self.assertEqual(arm.retry_budget("infra"), 2)
        self.assertEqual(arm.retry_budget("stalled"), 1)
        self.assertEqual(arm.retry_budget("stall"), 0)
        self.assertEqual(arm.retry_budget("rate_limited"), 3)
        self.assertEqual(arm.retry_budget("wrong_answer"), 0)
        self.assertEqual(arm.retry_budget("solved"), 0)
        self.assertFalse(arm.should_exhaust("stalled", 1))
        self.assertTrue(arm.should_exhaust("stalled", 2))
        self.assertEqual(arm.retries_remaining("stalled", 1), 1)
        self.assertEqual(arm.retries_remaining("stalled", 2), 0)

    def test_custom_budgets(self):
        arm = mq.ArmState("pi", {"infra": 5, "stalled": 0})
        self.assertEqual(arm.retry_budget("infra"), 5)
        self.assertEqual(arm.retry_budget("stalled"), 0)

    def test_legacy_stall_config_maps_to_stalled_failure_class(self):
        budgets = mq.resolve_retry_budgets({"stall": 2})
        self.assertEqual(budgets["stalled"], 2)
        self.assertNotIn("stall", budgets)

    def test_canonical_stalled_config_wins_over_legacy_alias(self):
        budgets = mq.resolve_retry_budgets({"stall": 5, "stalled": 2})
        self.assertEqual(budgets["stalled"], 2)

    def test_legacy_persisted_arm_budget_is_canonicalized(self):
        arm = mq.ArmState.from_dict({
            "name": "pi x m",
            "budgets": {"infra": 2, "stall": 1, "rate_limited": 3},
        })
        self.assertEqual(arm.retry_budget("stalled"), 1)
        self.assertEqual(arm.retry_budget("stall"), 0)

    def test_active_spec_budgets_override_persisted_arm_budget(self):
        arm = mq.ArmState.from_dict(
            {"name": "pi x m", "budgets": {"stall": 5}},
            retry_budgets={"infra": 2, "stalled": 2, "rate_limited": 3},
        )
        self.assertEqual(arm.retry_budget("stalled"), 2)

    def test_seeded_stalled_row_consumes_original_attempt(self):
        row = {"run_id": "cell", "failure_class": "stalled"}
        failed_attempts = mq.effective_failed_attempts(row, persisted_attempts=0)
        arm = mq.ArmState("pi x m", {"stalled": 1})
        self.assertEqual(failed_attempts, 1)
        self.assertFalse(arm.should_exhaust("stalled", failed_attempts))
        self.assertEqual(
            arm.retries_remaining("stalled", failed_attempts + 1), 0)

    def test_seeded_corrected_row_uses_effective_retry_class(self):
        row = {
            "run_id": "cell",
            "failure_class": "wrong_answer",
            "success": False,
            "completed": True,
            "checker_exit": 1,
            "turns": 2,
            "tokens_output": 10,
            "sampling_observed": [{"max_completion_tokens": 1}],
        }
        self.assertEqual(mq.row_failure_class(row), "infra")
        self.assertEqual(
            mq.effective_failed_attempts(row, persisted_attempts=0), 1)

    def test_seeded_legacy_row_derives_effective_retry_class(self):
        row = {
            "run_id": "cell",
            "success": False,
            "completed": False,
            "checker_exit": None,
            "error": "docker daemon not reachable (is Docker Desktop running?)",
            "tokens": None,
        }
        self.assertEqual(mq.row_failure_class(row), "infra")
        self.assertEqual(
            mq.effective_failed_attempts(row, persisted_attempts=0), 1)

    def test_result_history_wins_if_state_save_missed_latest_attempt(self):
        row = {"run_id": "cell", "failure_class": "stalled"}
        failed_attempts = mq.effective_failed_attempts(
            row, persisted_attempts=1, result_attempts=2)
        arm = mq.ArmState("pi x m", {"stalled": 1})
        self.assertEqual(failed_attempts, 2)
        self.assertTrue(arm.should_exhaust("stalled", failed_attempts))

    def test_backoff_rate_limited(self):
        d1 = mq.backoff_for_failure("rate_limited", 1, 60)
        self.assertAlmostEqual(d1, 60.0)
        d2 = mq.backoff_for_failure("rate_limited", 2, 60)
        self.assertAlmostEqual(d2, 120.0)
        d3 = mq.backoff_for_failure("rate_limited", 3, 60)
        self.assertAlmostEqual(d3, 240.0)

    def test_backoff_other_failures(self):
        d = mq.backoff_for_failure("infra", 1, 60)
        self.assertEqual(d, 10.0)  # fixed 10s for non-rate-limited


class RunnerCommandBuildingTests(unittest.TestCase):
    """Verify runner subprocess argv construction."""

    def test_basic_command(self):
        cell = {"harness": "pi", "model": "a", "task": "t1", "trial": 1}
        cmd = mq.build_runner_command(cell, "/path/results.jsonl", "/tasks", 2400, None, "local")
        self.assertIn("--force", cmd)
        self.assertIn("--harness", cmd)
        self.assertIn("pi", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("a", cmd)
        self.assertIn("--task", cmd)
        self.assertIn("t1", cmd)
        self.assertIn("--trial", cmd)
        self.assertIn("1", cmd)
        self.assertIn("--timeout", cmd)
        self.assertIn("2400", cmd)
        self.assertIn("--results-path", cmd)
        self.assertIn("/path/results.jsonl", cmd)
        self.assertIn("--tasks-dir", cmd)
        self.assertIn("/tasks", cmd)

    def test_docker_mode(self):
        cell = {"harness": "pi", "model": "a", "task": "t1", "trial": 1}
        cmd = mq.build_runner_command(cell, "r.jsonl", "/t", 2400, 600, "docker")
        self.assertIn("--exec", cmd)
        self.assertIn("docker", cmd)
        self.assertIn("--stall-timeout", cmd)
        self.assertIn("600", cmd)
        self.assertIn("--proxy", cmd)

    def test_no_stall_timeout_no_proxy(self):
        cell = {"harness": "pi", "model": "a", "task": "t1", "trial": 1}
        cmd = mq.build_runner_command(cell, "r.jsonl", "/t", 2400, None, "local")
        self.assertNotIn("--stall-timeout", cmd)
        # proxy is only added when stall_timeout is set
        self.assertNotIn("--proxy", cmd)

    def test_version_drift_flag_omitted_by_default(self):
        cell = {"harness": "codex", "model": "a", "task": "t1", "trial": 1}
        cmd = mq.build_runner_command(cell, "r.jsonl", "/t", 2400, None, "local")
        self.assertNotIn("--allow-version-drift", cmd)

    def test_version_drift_flag_passed_when_requested(self):
        cell = {"harness": "codex", "model": "a", "task": "t1", "trial": 1}
        cmd = mq.build_runner_command(
            cell, "r.jsonl", "/t", 2400, None, "local", allow_version_drift=True)
        self.assertIn("--allow-version-drift", cmd)


class MissingTaskImagesTests(unittest.TestCase):
    """Preflight for pinned per-task docker images."""

    def test_local_group_without_tasks_dir_is_skipped_not_crashed(self):
        # A local task group omits tasks_dir (runner resolves via discovery) and
        # pins no docker image. The preflight must skip it, not os.path.join a
        # None tasks_dir -- that TypeError killed an all-local spec before any
        # cell ran.
        spec = {
            "task_group": [{"tasks": ["make-it-run", "webcore"]}],
        }
        # docker_runner must never be consulted for a local group.
        def _fail(_ref):
            raise AssertionError("docker inspect should not run for a local group")
        self.assertEqual(mq.missing_task_images(spec, "/spec/dir", docker_runner=_fail), [])


class CoverageSummaryTests(unittest.TestCase):
    """Verify coverage output format."""

    def test_full_coverage(self):
        arms = [{"harness": "pi", "model": "a"}]
        tasks = ["t1"]
        cells = mq.expand_cells(arms, tasks, 1)
        arm_states = {a["harness"]: mq.ArmState(a["harness"]) for a in arms}
        arm_states["pi"].planned = 1
        arm_states["pi"].satisfied_cells.add("pi:t1:m:trial1")

        self.assertEqual(arm_states["pi"].satisfied, 1)
        self.assertEqual(arm_states["pi"].planned, 1)

    def test_partial_coverage_exhausted(self):
        arm = mq.ArmState("pi")
        arm.planned = 3
        arm.satisfied_cells.add("pi:t1:m:trial1")
        arm.exhausted_cells = ["pi:t1:a:trial2"]
        self.assertEqual(len(arm.exhausted_cells), 1)
        self.assertEqual(arm.satisfied, 1)

    def test_record_satisfied_clears_stale_exhaustion(self):
        run_id = "pi:t1:m:trial1"
        arm = mq.ArmState("pi")
        arm.exhausted_cells = [run_id]
        arm.consecutive_excluded = 2
        arm.record_satisfied(run_id)
        self.assertEqual(arm.satisfied_cells, {run_id})
        self.assertEqual(arm.exhausted_cells, [])
        self.assertEqual(arm.consecutive_excluded, 0)


if __name__ == "__main__":
    unittest.main()


class FirstAttemptIsNotARetryTests(unittest.TestCase):
    """A cell with no prior row must be RUN, not declared exhausted.

    Regression: on the queue's first live use it reported 0/9 satisfied,
    marking every cell "EXHAUSTED (fc=None attempts=0 budget=0)" before any
    of them executed. A fresh cell has no failure class, retry_budget(None)
    is 0 by design, and the check was ``attempts >= budget`` -- so 0 >= 0
    exhausted the cell up front. Retry budgets govern RETRIES only.
    """

    def test_fresh_cell_is_not_exhausted(self):
        arm = mq.ArmState("pi x laguna-s-2.1")
        self.assertFalse(arm.should_exhaust(None, 0),
                         "a cell that has never run must get its first attempt")

    def test_retry_budget_still_bounds_retries(self):
        arm = mq.ArmState("pi x m", {"rate_limited": 2})
        self.assertFalse(arm.should_exhaust("rate_limited", 1))
        self.assertFalse(arm.should_exhaust("rate_limited", 2))
        self.assertTrue(arm.should_exhaust("rate_limited", 3))

    def test_unknown_failure_class_exhausts_after_one_attempt(self):
        arm = mq.ArmState("pi x m", {"rate_limited": 3})
        self.assertTrue(arm.should_exhaust("wrong_answer", 1),
                        "no budget for a class means no retry after attempt 1")
        self.assertTrue(arm.should_exhaust(None, 1))


class ArmIdentityIncludesModelTests(unittest.TestCase):
    """Arms must be keyed by (harness, model), not harness alone.

    Regression: pi x laguna and pi x inkling collapsed to one arm named "pi",
    so they shared pause state and their coverage lines were printed twice as
    indistinguishable "pi: 0/6".
    """

    def test_same_harness_different_models_are_distinct_arms(self):
        arms = [{"harness": "pi", "model": "laguna-s-2.1"},
                {"harness": "pi", "model": "inkling"}]
        cells = mq.expand_cells(arms, ["webcore"], 1)
        names = {c["arm"] for c in cells}
        self.assertEqual(len(names), 2, f"expected two distinct arms, got {names}")
        self.assertIn("pi x laguna-s-2.1", names)
        self.assertIn("pi x inkling", names)


class RunnerInvokedAsModuleTests(unittest.TestCase):
    """The runner must be invoked as ``-m obench.run``, not as a file path.

    Regression: obench/run.py uses relative imports, so running it as a script
    path died with "attempted relative import with no known parent package" on
    the queue's first live cell.
    """

    def test_command_uses_module_form(self):
        cell = {"harness": "pi", "model": "m", "task": "t", "trial": 1,
                "run_id": "pi:t:m:trial1", "arm": "pi x m", "arm_idx": 0}
        cmd = mq.build_runner_command(
            cell, results_path="/tmp/r.jsonl", tasks_dir="tasks",
            timeout=60, stall_timeout=None, exec_mode="local")
        self.assertIn("-m", cmd, f"runner must be a module invocation: {cmd}")
        self.assertIn("obench.run", cmd)
        self.assertFalse(any(str(part).endswith("run.py") for part in cmd),
                         f"must not invoke run.py by path: {cmd}")

    def test_command_forces_rerun_over_excluded_rows(self):
        cell = {"harness": "pi", "model": "m", "task": "t", "trial": 2,
                "run_id": "pi:t:m:trial2", "arm": "pi x m", "arm_idx": 0}
        cmd = mq.build_runner_command(
            cell, results_path="/tmp/r.jsonl", tasks_dir="tasks",
            timeout=60, stall_timeout=None, exec_mode="local")
        self.assertIn("--force", cmd,
                      "without --force the runner skips cells that already have "
                      "an excluded row, which is exactly the coverage gap")


class TasksDirOmittedWhenUnsetTests(unittest.TestCase):
    """--tasks-dir must be OMITTED (not emitted empty) when no dir is set.

    Regression #4: with a spec that omits tasks_dir, the command still carried
    ``--tasks-dir`` with an empty/None value; the runner then failed and wrote
    no row, and the queue misread the missing row as an unclassifiable exhausted
    cell. When no tasks dir is known the flag must be absent so the runner
    resolves it via config/discovery.
    """

    def test_none_tasks_dir_omits_flag(self):
        cell = {"harness": "null", "model": "m", "task": "t", "trial": 1}
        cmd = mq.build_runner_command(
            cell, "/r.jsonl", None, 60, None, "local")
        self.assertNotIn("--tasks-dir", cmd, cmd)

    def test_cell_tasks_dir_overrides_default(self):
        cell = {"harness": "null", "model": "m", "task": "t", "trial": 1,
                "tasks_dir": "/group/dir", "exec_mode": "docker"}
        cmd = mq.build_runner_command(cell, "/r.jsonl", None, 60, None, "local")
        self.assertIn("--tasks-dir", cmd)
        self.assertIn("/group/dir", cmd)
        # Per-group exec_mode on the cell wins over the default.
        self.assertIn("--exec", cmd)
        self.assertIn("docker", cmd)

    def test_cell_local_exec_mode_overrides_docker_default(self):
        cell = {"harness": "null", "model": "m", "task": "t", "trial": 1,
                "tasks_dir": "/g", "exec_mode": "local"}
        cmd = mq.build_runner_command(cell, "/r.jsonl", None, 60, None, "docker")
        self.assertNotIn("--exec", cmd, "cell's local exec_mode must override docker default")


class GroupedCellExpansionTests(unittest.TestCase):
    """expand_cells_grouped carries per-group tasks_dir + exec_mode onto cells."""

    def test_cells_carry_group_tasks_dir_and_exec_mode(self):
        arms = [{"harness": "null", "model": "m"}]
        groups = [
            {"tasks": ["core"], "tasks_dir": "/core", "exec_mode": "local"},
            {"tasks": ["tb"], "tasks_dir": "/tb", "exec_mode": "docker"},
        ]
        cells = mq.expand_cells_grouped(arms, groups, 1)
        by_task = {c["task"]: c for c in cells}
        self.assertEqual(by_task["core"]["tasks_dir"], "/core")
        self.assertEqual(by_task["core"]["exec_mode"], "local")
        self.assertEqual(by_task["tb"]["tasks_dir"], "/tb")
        self.assertEqual(by_task["tb"]["exec_mode"], "docker")

    def test_group_without_tasks_dir_yields_none(self):
        arms = [{"harness": "null", "model": "m"}]
        groups = [{"tasks": ["t"], "tasks_dir": None, "exec_mode": "local"}]
        cells = mq.expand_cells_grouped(arms, groups, 1)
        self.assertIsNone(cells[0]["tasks_dir"])

    def test_resolve_group_tasks_dir_precedence(self):
        spec = {"tasks_dir": "spec-tasks"}
        # Group-level wins.
        self.assertTrue(mq.resolve_group_tasks_dir(
            {"tasks_dir": "grp"}, spec, "/base").endswith("/base/grp"))
        # Falls back to spec-level.
        self.assertTrue(mq.resolve_group_tasks_dir(
            {}, spec, "/base").endswith("/base/spec-tasks"))
        # Neither set -> None (runner resolves).
        self.assertIsNone(mq.resolve_group_tasks_dir({}, {}, "/base"))


class CoverageCannotExceedPlannedTests(unittest.TestCase):
    """Coverage must be idempotent: re-verifying a cell cannot inflate it.

    Regression: satisfied was an incrementing int persisted across resumes, so
    a second pass over already-satisfied cells double-counted and the report
    printed impossible coverage ("Total: 4/2 satisfied", and 300% in another
    run). Coverage is the control protecting solve-rate honesty; a control that
    can print 200% is not a control. Satisfied is now the SET of cells with a
    verdict.
    """

    def test_duplicate_satisfaction_counts_once(self):
        arm = mq.ArmState("pi x m")
        arm.planned = 2
        for _ in range(5):
            arm.satisfied_cells.add("pi:t:m:trial1")
        arm.satisfied_cells.add("pi:t:m:trial2")
        self.assertEqual(arm.satisfied, 2)
        self.assertLessEqual(arm.satisfied, arm.planned)

    def test_satisfied_survives_state_roundtrip_without_growing(self):
        arm = mq.ArmState("pi x m")
        arm.planned = 3
        arm.satisfied_cells.update({"a", "b"})
        restored = mq.ArmState.from_dict(arm.to_dict())
        self.assertEqual(restored.satisfied, 2)
        # Re-adding the same cells after a resume must not inflate the count.
        restored.satisfied_cells.update({"a", "b"})
        self.assertEqual(restored.satisfied, 2)


class PerGroupTimeoutTests(unittest.TestCase):
    """A task group may set its own timeout, overriding the spec-global one."""

    def test_cell_timeout_overrides_global(self):
        cell = {"harness": "codex", "model": "m", "task": "t", "trial": 1, "timeout": 600}
        cmd = mq.build_runner_command(cell, "r.jsonl", "/t", 2400, None, "local")
        i = cmd.index("--timeout")
        self.assertEqual(cmd[i + 1], "600")

    def test_cell_without_timeout_uses_global(self):
        cell = {"harness": "codex", "model": "m", "task": "t", "trial": 1}
        cmd = mq.build_runner_command(cell, "r.jsonl", "/t", 2400, None, "local")
        i = cmd.index("--timeout")
        self.assertEqual(cmd[i + 1], "2400")

    def test_resolve_groups_carries_group_timeout(self):
        spec = {"task_group": [{"tasks": ["a"], "timeout": 600}, {"tasks": ["b"]}]}
        groups = mq.resolve_groups(spec, "/spec", "local")
        self.assertEqual(groups[0]["timeout"], 600)
        self.assertIsNone(groups[1]["timeout"])

    def test_expand_cells_grouped_carries_timeout(self):
        arms = [{"harness": "codex", "model": "m"}]
        groups = [{"tasks": ["a"], "tasks_dir": None, "exec_mode": "local", "timeout": 600}]
        cells = mq.expand_cells_grouped(arms, groups, 1)
        self.assertEqual(cells[0]["timeout"], 600)


class CumulativeWallTests(unittest.TestCase):
    """load_cumulative_wall sums wall_time_s across every attempt of a run_id."""

    def test_sums_wall_across_attempts(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"run_id": "a", "wall_time_s": 400}) + "\n")
            f.write(json.dumps({"run_id": "a", "wall_time_s": 410}) + "\n")
            f.write(json.dumps({"run_id": "b", "wall_time_s": 30}) + "\n")
            path = f.name
        try:
            w = mq.load_cumulative_wall(path)
            self.assertAlmostEqual(w["a"], 810)
            self.assertAlmostEqual(w["b"], 30)
        finally:
            os.unlink(path)

    def test_missing_file_is_empty(self):
        self.assertEqual(mq.load_cumulative_wall("/no/such/file.jsonl"), {})


class WallCapTests(unittest.TestCase):
    """wall_cap_exceeded gates re-queue on cumulative retry wall time."""

    def test_disabled_when_cap_none(self):
        self.assertFalse(mq.wall_cap_exceeded(10_000, None))

    def test_not_exceeded_below_cap(self):
        self.assertFalse(mq.wall_cap_exceeded(100, 3600))

    def test_exceeded_at_or_above_cap(self):
        self.assertTrue(mq.wall_cap_exceeded(3600, 3600))
        self.assertTrue(mq.wall_cap_exceeded(5000, 3600))
