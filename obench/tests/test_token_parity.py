#!/usr/bin/env python3
"""Fixture-backed tests for TOKEN_PARITY.md adapter usage normalization."""

import importlib.util
import json
import os
import subprocess

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import tempfile
import unittest
from unittest import mock

ADAPTERS_DIR = os.path.join(BENCH_DIR, "adapters")
FIXTURE_DIR = os.path.join(BENCH_DIR, "tests", "fixtures", "usage", "deepseek-v4-flash")


def load_adapter(name):
    path = os.path.join(ADAPTERS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"bench_adapter_{name}_token_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def expected(name):
    with open(os.path.join(FIXTURE_DIR, "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)[name]


SPLIT_KEYS = (
    "tokens_input_uncached",
    "tokens_cache_read",
    "tokens_cache_write",
    "tokens_output",
    "tokens_reasoning",
    "token_basis",
)


class TokenParityFixtureTests(unittest.TestCase):
    def assert_usage_matches(self, actual, exp):
        for key in SPLIT_KEYS:
            self.assertEqual(actual.get(key), exp.get(key), key)
        self.assertEqual(actual.get("usage_raw"), exp.get("usage_raw"))
        if actual.get("tokens_input_uncached") is None or actual.get("tokens_output") is None:
            self.assertIsNone(actual.get("tokens"))
        else:
            self.assertEqual(
                actual.get("tokens"),
                actual["tokens_input_uncached"] + actual["tokens_output"],
            )

    def test_pi_probe_fixture_sums_all_turns(self):
        pi = load_adapter("pi")
        tokens, turns, tail, usage = pi._parse_json_with_usage(read_fixture("pi-stream.txt"))
        actual = {"tokens": tokens, **usage}
        self.assertEqual(turns, 3)
        self.assert_usage_matches(actual, expected("pi"))

    def test_pi_multiturn_sum_not_last_event(self):
        pi = load_adapter("pi")
        stream = "\n".join([
            json.dumps({"type": "turn_end", "message": {"usage": {
                "input": 2, "cacheRead": 3, "cacheWrite": 5,
                "output": 7, "reasoning": 1, "totalTokens": 17}}}),
            json.dumps({"type": "turn_end", "message": {"usage": {
                "input": 11, "cacheRead": 13, "cacheWrite": 17,
                "output": 19, "reasoning": 3, "totalTokens": 60}}}),
        ])
        tokens, turns, tail, usage = pi._parse_json_with_usage(stream)
        self.assertEqual(turns, 2)
        self.assertEqual(tokens, 39)
        self.assertEqual(usage["tokens_input_uncached"], 13)
        self.assertEqual(usage["tokens_cache_read"], 16)
        self.assertEqual(usage["tokens_cache_write"], 22)
        self.assertEqual(usage["tokens_output"], 26)
        self.assertEqual(usage["tokens_reasoning"], 4)
        self.assertEqual(usage["token_basis"], "vendor_split")

    def test_pi_agent_end_usage_fallback(self):
        pi = load_adapter("pi")
        stream = json.dumps({"type": "agent_end", "messages": [{
            "role": "assistant",
            "usage": {"input": 5, "cacheRead": 7, "cacheWrite": 11,
                      "output": 13, "reasoning": 3, "totalTokens": 36},
            "content": [{"type": "text", "text": "done"}],
        }]})
        tokens, turns, tail, usage = pi._parse_json_with_usage(stream)
        self.assertIsNone(turns)
        self.assertEqual(tokens, 18)
        self.assertEqual(tail, "done")
        self.assertEqual(usage["tokens_input_uncached"], 5)
        self.assertEqual(usage["tokens_cache_read"], 7)
        self.assertEqual(usage["tokens_cache_write"], 11)
        self.assertEqual(usage["tokens_output"], 13)
        self.assertEqual(usage["tokens_reasoning"], 3)
        self.assertEqual(usage["token_basis"], "vendor_split")

    def test_opencode_probe_fixture_sums_steps_and_normalizes_reasoning_output(self):
        opencode = load_adapter("opencode")
        tokens, turns, tail, usage = opencode._parse_json_with_usage(read_fixture("opencode-stream.txt"))
        actual = {"tokens": tokens, **usage}
        self.assertEqual(turns, 3)
        self.assert_usage_matches(actual, expected("opencode"))

    def test_opencode_multistep_sum_not_last_event(self):
        opencode = load_adapter("opencode")
        stream = "\n".join([
            json.dumps({"type": "step_finish", "part": {"tokens": {
                "input": 2, "output": 3, "reasoning": 5,
                "cache": {"read": 7, "write": 11}, "total": 28}}}),
            json.dumps({"type": "step_finish", "part": {"tokens": {
                "input": 13, "output": 17, "reasoning": 19,
                "cache": {"read": 23, "write": 29}, "total": 101}}}),
        ])
        tokens, turns, tail, usage = opencode._parse_json_with_usage(stream)
        self.assertEqual(turns, 2)
        self.assertEqual(tokens, 59)
        self.assertEqual(usage["tokens_input_uncached"], 15)
        self.assertEqual(usage["tokens_cache_read"], 30)
        self.assertEqual(usage["tokens_cache_write"], 40)
        self.assertEqual(usage["tokens_output"], 44)  # output + reasoning
        self.assertEqual(usage["tokens_reasoning"], 24)
        self.assertEqual(usage["token_basis"], "vendor_split")

    def test_claude_probe_fixture_prefers_model_usage_and_reasoning_unknown(self):
        claude = load_adapter("claude")
        tokens, turns, tail, ok, usage = claude._parse_json_with_usage(read_fixture("claude-stream.txt"))
        actual = {"tokens": tokens, **usage}
        self.assertEqual(turns, 1)
        self.assertTrue(ok)
        self.assert_usage_matches(actual, expected("claude"))
        self.assertIsNone(usage["tokens_reasoning"])

    def test_codex_probe_fixture_uses_final_cache_inclusive_aggregate(self):
        codex = load_adapter("codex")
        tokens, turns, tail, usage = codex._parse_json_with_usage(read_fixture("codex-stream.txt"))
        actual = {"tokens": tokens, **usage}
        self.assertEqual(turns, 1)
        self.assert_usage_matches(actual, expected("codex"))
        self.assertEqual(tokens, 508)  # do not double-count reasoning subset

    def test_codex_153_cache_write_field_survives_run_normalization(self):
        codex = load_adapter("codex")
        fixture = os.path.join(BENCH_DIR, "tests", "fixtures", "usage",
                               "codex-0.153.0", "turn-completed.json")
        with open(fixture, encoding="utf-8") as fh:
            event = json.load(fh)
        # The captured event reports zero. Missing and positive values below
        # are synthetic boundary cases for the same schema.
        for writes in (0, None, 30):
            with self.subTest(cache_write_input_tokens=writes):
                usage = dict(event["usage"])
                if writes is None:
                    usage.pop("cache_write_input_tokens")
                else:
                    usage["cache_write_input_tokens"] = writes
                stream = json.dumps({"type": "turn.completed", "usage": usage})
                proc = subprocess.CompletedProcess([], 0, stdout=stream, stderr="")
                with tempfile.TemporaryDirectory() as directory:
                    with mock.patch.object(codex.subprocess, "run", return_value=proc):
                        result = codex.run(
                            "offline replay", directory, "gpt-5.6-sol", 1,
                            env_override={"CODEX_HOME": directory},
                        )
                self.assertTrue(result["completed"])
                self.assertEqual(result["usage_raw"], usage)
                self.assertEqual(result["tokens_cache_write"], writes)
                self.assertEqual(result["token_basis"],
                                 "estimated" if writes is None else "vendor_split")
                self.assertEqual(result["tokens_input_uncached"], 17667 - (writes or 0))
                self.assertEqual(result["tokens_cache_read"], 106368)
                self.assertEqual(result["tokens_output"], 4774)
                self.assertEqual(result["tokens_reasoning"], 2278)
                self.assertEqual(result["tokens"], 22441 - (writes or 0))

    def test_invariant_violation_downgrades_basis(self):
        pi = load_adapter("pi")
        stream = json.dumps({"type": "turn_end", "message": {"usage": {
            "input": 2, "cacheRead": 3, "cacheWrite": 5,
            "output": 7, "reasoning": 1, "totalTokens": 999}}})
        tokens, turns, tail, usage = pi._parse_json_with_usage(stream)
        self.assertEqual(tokens, 9)
        self.assertEqual(usage["token_basis"], "estimated")

    def test_cursor_split_unknown_harness_reported_scalar_preserved(self):
        cursor = load_adapter("cursor")
        tokens, turns, tail, usage = cursor._parse_json_with_usage(json.dumps({
            "result": "ok",
            "usage": {"inputTokens": 12, "outputTokens": 34,
                      "cacheReadTokens": 56, "cacheWriteTokens": 78},
        }))
        self.assertEqual(tokens, 46)
        self.assertIsNone(turns)
        self.assertEqual(usage["token_basis"], "harness_reported")
        self.assertEqual(usage["tokens_input_uncached"], 12)
        self.assertEqual(usage["tokens_cache_read"], 56)
        self.assertEqual(usage["tokens_cache_write"], 78)
        self.assertEqual(usage["tokens_output"], 34)

    def test_cursor_empty_output_is_defensive(self):
        cursor = load_adapter("cursor")
        self.assertEqual(cursor._parse_json(""), (None, None, ""))
        tokens, turns, tail, usage = cursor._parse_json_with_usage("")
        self.assertIsNone(tokens)
        self.assertIsNone(turns)
        self.assertEqual(tail, "")
        self.assertIsNone(usage["token_basis"])

    def test_devin_split_unknown_estimated_scalar_preserved(self):
        devin = load_adapter("devin")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as fh:
            json.dump({
                "final_metrics": {
                    "total_prompt_tokens": 100,
                    "total_completion_tokens": 7,
                    "total_cached_tokens": 0,
                },
                "steps": [
                    {"metrics": {"prompt_tokens": 40}},
                    {"metrics": {"prompt_tokens": 60}},
                ],
            }, fh)
            path = fh.name
        try:
            tokens, turns, usage = devin._parse_export_with_usage(path)
        finally:
            os.unlink(path)
        self.assertEqual(tokens, 67)
        self.assertEqual(turns, 2)
        self.assertEqual(usage["token_basis"], "estimated")
        self.assertIsNone(usage["tokens_input_uncached"])
        self.assertIsNone(usage["tokens_output"])
if __name__ == "__main__":
    unittest.main()
