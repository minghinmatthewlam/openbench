#!/usr/bin/env python3
"""Tests for result-row failure classification semantics."""

import os
import tempfile
import unittest


import sys

from obench import failure_class  # noqa: E402
from obench import run  # noqa: E402


MOONSHOT_429 = (
    "APIError: HTTP 429 rate_limit: TPD rate limit, current 1502271, "
    "limit 1500000. Please retry later."
)


class TestClassifyFailure(unittest.TestCase):
    def test_solved_when_checker_passed(self):
        row = {"success": True, "checker_exit": 0, "error": "timeout after 1s"}
        self.assertEqual(failure_class.classify_failure(row, MOONSHOT_429), "solved")

    def test_moonshot_429_signature_is_rate_limited(self):
        row = {"success": False, "completed": False, "tokens": 0, "turns": 1}
        self.assertEqual(failure_class.classify_failure(row, MOONSHOT_429), "rate_limited")

    def test_rate_limited_beats_timeout_and_wrong_answer(self):
        row = {"success": False, "error": "timeout after 600s", "checker_exit": 1}
        self.assertEqual(
            failure_class.classify_failure(row, "APIError: quota exhausted"),
            "rate_limited",
        )

    def test_domain_timeout_text_does_not_force_timeout_class(self):
        row = {"success": False, "completed": True, "checker_exit": 1, "error": None}
        transcript = "The task source defines timeout_s and a TimeoutError helper."
        self.assertEqual(failure_class.classify_failure(row, transcript), "wrong_answer")

    def test_domain_text_about_rate_limiting_is_not_provider_rate_limit(self):
        row = {"success": False, "completed": True, "checker_exit": 1, "error": None}
        transcript = (
            "README says the webcore middleware returns 429 Too Many Requests "
            "when the application's token-bucket rate limit exceeded path is tested."
        )
        self.assertEqual(failure_class.classify_failure(row, transcript), "wrong_answer")

    def test_provider_http_429_context_is_rate_limited(self):
        row = {"success": False, "completed": False, "checker_exit": 1, "error": None}
        self.assertEqual(
            failure_class.classify_failure(row, "HTTP 429 Too Many Requests from API response"),
            "rate_limited",
        )

    def test_infra_markers(self):
        cases = [
            "docker daemon not reachable (is Docker Desktop running?)",
            "container produced no result sentinel (exit 1)",
            "SETUP-NEEDED: export MOONSHOT_API_KEY to use kimi-k2.7-code",
            "missing pi auth at /home/me/.pi/agent/auth.json",
            "No such image: openbench-harness:latest",
            '"stopReason":"error","errorMessage":"No API key for provider: openai-codex"',
        ]
        for text in cases:
            with self.subTest(text=text):
                row = {"success": False, "error": text, "checker_exit": 1}
                self.assertEqual(failure_class.classify_failure(row, ""), "infra")

    def test_infra_beats_timeout(self):
        row = {"success": False, "error": "container timeout; No such image: openbench-harness:latest"}
        self.assertEqual(failure_class.classify_failure(row, ""), "infra")

    def test_timeout_from_error_checker_or_wall_cap(self):
        self.assertEqual(
            failure_class.classify_failure({"success": False, "error": "timeout after 30s"}, ""),
            "timeout",
        )
        self.assertEqual(
            failure_class.classify_failure({"success": False, "checker_exit": "timeout"}, ""),
            "timeout",
        )
        self.assertEqual(
            failure_class.classify_failure({"success": False, "wall_time_s": 599.0, "tokens": 10}, "", timeout_s=600),
            "timeout",
        )

    def test_silent_cap_riding_retry_loop_is_infra(self):
        row = {
            "success": False,
            "completed": False,
            "wall_time_s": 1201.0,
            "tokens": None,
            "turns": None,
            "output_tail": "",
            "error": "timeout after 1200s",
            "checker_exit": None,
        }
        self.assertEqual(failure_class.classify_failure(row, "", timeout_s=1200), "infra")

    def test_cap_riding_with_work_evidence_stays_timeout(self):
        row = {
            "success": False,
            "completed": False,
            "wall_time_s": 1201.0,
            "tokens": 90000,
            "turns": 20,
            "output_tail": "",
            "error": "timeout after 1200s",
            "checker_exit": None,
        }
        self.assertEqual(failure_class.classify_failure(row, "", timeout_s=1200), "timeout")

    def test_cap_riding_with_meaningful_output_stays_timeout(self):
        row = {
            "success": False,
            "completed": False,
            "wall_time_s": 1201.0,
            "tokens": None,
            "turns": None,
            "output_tail": "",
            "error": "timeout after 1200s",
            "checker_exit": None,
        }
        output = "I inspected the failing tests and started rewriting the parser. " * 4
        self.assertEqual(failure_class.classify_failure(row, output, timeout_s=1200), "timeout")

    def test_wrong_answer_when_agent_finished_and_checker_failed(self):
        row = {"success": False, "completed": True, "checker_exit": 1, "error": None}
        self.assertEqual(failure_class.classify_failure(row, "normal transcript"), "wrong_answer")

    def test_completed_silent_run_is_infra_not_wrong_answer(self):
        row = {"success": False, "completed": True, "checker_exit": 1,
               "wall_time_s": 343.0, "tokens": None, "tokens_output": 0}
        self.assertEqual(failure_class.classify_failure(row, "", timeout_s=1200), "infra")
        self.assertEqual(failure_class.classify_failure_reason(row, ""),
                         "silent-no-model-call")

    def test_incomplete_zero_work_run_is_infra_not_wrong_answer(self):
        # Real shape from an OpenRouter provider-overload abandonment: the model
        # answered "high demand / Reconnecting 1..5/5" for minutes, then exited
        # with no work done. completed=False, no tokens/turns, workspace
        # untouched, and it did NOT ride the cap (419s of a 2400s budget). The
        # throttle text lives only in the transcript, not in any row field, so
        # marker matching cannot see it -- the structural no-work shape must.
        row = {"success": False, "completed": False, "checker_exit": 1,
               "tokens": None, "tokens_output": None, "turns": None,
               "workspace_changed": False, "wall_time_s": 419.68, "error": "exit 1"}
        self.assertEqual(failure_class.classify_failure(row, "", timeout_s=2400), "infra")

    def test_incomplete_run_that_did_work_is_not_swallowed_as_infra(self):
        # Guard: a cell cut off AFTER real model work (tokens/turns) must not be
        # reclassified by the zero-work gate -- only genuine no-work runs are infra.
        row = {"success": False, "completed": False, "checker_exit": 1,
               "tokens": 500, "tokens_output": 200, "turns": 8, "wall_time_s": 419.0}
        self.assertNotEqual(failure_class.classify_failure(row, "", timeout_s=2400), "infra")

    def test_real_wrong_answer_with_tokens_stays_wrong_answer(self):
        row = {"success": False, "completed": True, "checker_exit": 1,
               "tokens": 100, "tokens_output": 25}
        self.assertEqual(failure_class.classify_failure(row, ""), "wrong_answer")

    def test_token_parse_failure_with_long_output_stays_wrong_answer(self):
        row = {"success": False, "completed": True, "checker_exit": 1,
               "tokens": None, "tokens_output": None}
        output = "Model inspected the repository and attempted a repair. " * 6
        self.assertEqual(failure_class.classify_failure(row, output), "wrong_answer")

    def test_explicit_workspace_change_prevents_silent_reclassification(self):
        row = {"success": False, "completed": True, "checker_exit": 1,
               "tokens": None, "workspace_changed": True}
        self.assertEqual(failure_class.classify_failure(row, ""), "wrong_answer")

    def test_intentional_null_negative_control_stays_wrong_answer(self):
        row = {"harness": "null", "success": False, "completed": True,
               "checker_exit": 1, "tokens": None, "turns": None}
        self.assertEqual(failure_class.classify_failure(row, ""), "wrong_answer")

    def test_proxy_upstream_failure_is_infra_even_with_long_error(self):
        row = {"success": False, "completed": False, "checker_exit": 1}
        self.assertEqual(failure_class.classify_failure(
            row, "API error (status 502): proxy_upstream_failed"), "infra")

    def test_stronger_marker_does_not_get_silent_failure_reason(self):
        row = {"success": False, "completed": True, "checker_exit": 1,
               "tokens": None, "error": "proxy_upstream_failed"}
        self.assertEqual(failure_class.classify_failure(row, ""), "infra")
        self.assertIsNone(failure_class.classify_failure_reason(row, ""))

    def test_instant_bare_cli_exit_without_tokens_is_infra(self):
        row = {
            "run_id": "codex:terminal-bench/cancel-async-tasks:gpt-5.6-sol:trial1",
            "harness": "codex",
            "model": "gpt-5.6-sol",
            "task": "terminal-bench/cancel-async-tasks",
            "trial": 1,
            "success": False,
            "completed": False,
            "failure_class": "wrong_answer",
            "error": "exit 1",
            "tokens": None,
            "tokens_fresh": None,
            "turns": None,
            "wall_time_s": 4.344,
            "checker_exit": 1,
            "score": 0.0,
        }
        self.assertEqual(failure_class.classify_failure(row, ""), "infra")
        self.assertTrue(failure_class.has_instant_cli_exit_shape(row))

    def test_slow_exit_with_model_tokens_stays_wrong_answer(self):
        row = {
            "success": False,
            "completed": False,
            "error": "exit 1",
            "tokens": 12345,
            "wall_time_s": 300.0,
            "checker_exit": 1,
        }
        self.assertEqual(failure_class.classify_failure(row, ""), "wrong_answer")
        self.assertFalse(failure_class.has_instant_cli_exit_shape(row))


class TestRunnerWriteTimeClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="bench_classify_")
        self.tasks_dir = os.path.join(self.tmp, "tasks")
        self.task = "tiny"
        task_dir = os.path.join(self.tasks_dir, self.task)
        os.makedirs(os.path.join(task_dir, "workspace"))
        with open(os.path.join(task_dir, "instruction.md"), "w", encoding="utf-8") as fh:
            fh.write("do it")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_runner_scans_full_output_not_just_tail(self):
        orig_invoke, orig_checker = run.invoke_adapter, run.run_checker

        def fake_invoke(*args, **kwargs):
            return {
                "completed": False,
                "error": "exit 1",
                "output_tail": "tail without marker",
                "full_output": MOONSHOT_429 + "\n" + ("x" * 3000),
                "tokens": 0,
                "turns": 1,
                "cmd": ["fake"],
            }, "local"

        try:
            run.invoke_adapter = fake_invoke
            run.run_checker = lambda *a, **k: (1, None, "", "")
            row = run.run_cell(
                "fake", self.task, "kimi-k2.7-code", 1, 600,
                self.tasks_dir, self.tmp, 30,
            )
        finally:
            run.invoke_adapter, run.run_checker = orig_invoke, orig_checker

        self.assertNotIn("output_tail", run.ROW_FIELDS)
        self.assertEqual(row["output_tail"], "tail without marker")  # internal only, not persisted
        self.assertEqual(row["failure_class"], "rate_limited")

    def test_runner_workspace_change_is_real_model_work_evidence(self):
        orig_invoke, orig_checker = run.invoke_adapter, run.run_checker

        def fake_invoke(_mode, _harness, _instruction, workdir, *_args, **_kwargs):
            with open(os.path.join(workdir, "model-edit.txt"), "w", encoding="utf-8") as fh:
                fh.write("changed")
            return ({"completed": True, "error": None, "output_tail": "",
                     "full_output": "", "tokens": None, "turns": None,
                     "cmd": ["fake"]}, "local")

        try:
            run.invoke_adapter = fake_invoke
            run.run_checker = lambda *a, **k: (1, None, "", "")
            row = run.run_cell("fake", self.task, "model", 1, 600,
                               self.tasks_dir, self.tmp, 30)
        finally:
            run.invoke_adapter, run.run_checker = orig_invoke, orig_checker

        self.assertTrue(row["workspace_changed"])
        self.assertEqual(row["failure_class"], "wrong_answer")

    def test_runner_uses_docker_host_wall_time_when_larger(self):
        orig_invoke, orig_checker = run.invoke_adapter, run.run_checker

        def fake_invoke(*args, **kwargs):
            return {
                "completed": False,
                "error": "container timeout after 600s (+grace); killed",
                "output_tail": "",
                "full_output": "",
                "tokens": None,
                "turns": None,
                "cmd": ["fake"],
                "host_wall_time_s": 1234.567,
            }, "docker"

        try:
            run.invoke_adapter = fake_invoke
            run.run_checker = lambda *a, **k: (1, None, "", "")
            row = run.run_cell(
                "fake", self.task, "deepseek-v4-flash", 1, 600,
                self.tasks_dir, self.tmp, 30,
            )
        finally:
            run.invoke_adapter, run.run_checker = orig_invoke, orig_checker

        self.assertEqual(row["exec_mode"], "docker")
        self.assertEqual(row["wall_time_s"], 1234.567)

    def test_runner_ignores_local_adapter_host_wall_time_extra(self):
        orig_invoke, orig_checker = run.invoke_adapter, run.run_checker

        def fake_invoke(*args, **kwargs):
            return {
                "completed": False,
                "error": "exit 1",
                "output_tail": "",
                "full_output": "",
                "tokens": None,
                "turns": None,
                "cmd": ["fake"],
                "host_wall_time_s": 1234.567,
            }, "local"

        try:
            run.invoke_adapter = fake_invoke
            run.run_checker = lambda *a, **k: (1, None, "", "")
            row = run.run_cell(
                "fake", self.task, "deepseek-v4-flash", 1, 600,
                self.tasks_dir, self.tmp, 30,
            )
        finally:
            run.invoke_adapter, run.run_checker = orig_invoke, orig_checker

        self.assertEqual(row["exec_mode"], "local")
        self.assertLess(row["wall_time_s"], 10)


    def test_adapter_traceback_missing_binary_is_infra(self):
        row = {"success": False, "completed": False, "tokens": None,
               "error": "Traceback (most recent call last):\n  ...\nFileNotFoundError: [Errno 2] No such file or directory: 'aider'",
               "wall_time_s": 0.002}
        self.assertEqual(failure_class.classify_failure(row), "infra")

    def test_task_debugging_traceback_with_real_work_is_wrong_answer(self):
        # Regression: agents debugging Python tasks print tracebacks in their
        # transcripts; a worked cell (tokens + turns) must never be reclassified
        # as infra by the adapter-crash markers (10 raman-fitting cells were).
        row = {"success": False, "completed": True, "tokens": 35768, "turns": 16,
               "output_tail": "ran fit.py\nTraceback (most recent call last):\n  ...\nValueError: bad fit",
               "wall_time_s": 234.0, "checker_exit": 1}
        self.assertEqual(failure_class.classify_failure(row), "wrong_answer")

    def test_vendor_authentication_error_is_infra(self):
        row = {"success": False, "completed": True, "tokens": 0,
               "output_tail": 'Fails, Your api key: ****ogus is invalid","type":"authentication_error"',
               "wall_time_s": 2.0}
        self.assertEqual(failure_class.classify_failure(row), "infra")

if __name__ == "__main__":
    unittest.main()
