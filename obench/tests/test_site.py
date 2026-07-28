#!/usr/bin/env python3
"""Tests for the unified harness + gateway leaderboard site."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from obench import gateway_probe_publish, publish, site


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _make_task(root, name):
    task_dir = os.path.join(root, name)
    os.makedirs(os.path.join(task_dir, "workspace"), exist_ok=True)
    _write(os.path.join(task_dir, "instruction.md"), f"# {name}\nDo the thing.\n")
    _write(os.path.join(task_dir, "checker.sh"), "#!/bin/sh\nexit 0\n")
    _write(os.path.join(task_dir, "workspace", "main.py"), "print('hi')\n")
    return task_dir


def _row(harness, task, trial, success, *, model="model-x", wall=10.0, tokens=100):
    return {
        "run_id": f"{harness}:{task}:{model}:trial{trial}",
        "harness": harness,
        "model": model,
        "task": task,
        "trial": trial,
        "success": success,
        "score": 1.0 if success else 0.0,
        "failure_class": "solved" if success else "wrong_answer",
        "wall_time_s": wall,
        "tokens_input_uncached": tokens - 20,
        "tokens_output": 20,
        "tokens_cache_read": 50,
        "tokens_cache_write": 5,
        "tokens": tokens,
        "token_basis": "vendor_split",
        "harness_version": "1.0",
        "timeout_s": 60,
        "completed": True,
        "candidate_provenance": None,
    }


def _publish_bundle(out_dir, tasks_dir, rows, *, title="Test card"):
    results = os.path.join(out_dir, "_src.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    with open(results, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return publish.create_bundle(results, out_dir, tasks_dirs=[tasks_dir], title=title)


class _SiteFixture(unittest.TestCase):
    """A docs/ root holding one verified two-arm harness bundle."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.site_dir = os.path.join(self.root, "docs")
        tasks_dir = os.path.join(self.root, "tasks")
        _make_task(tasks_dir, "alpha")
        _make_task(tasks_dir, "beta")

        rows = []
        for task in ("alpha", "beta"):
            for trial in (1, 2):
                rows.append(_row("fast", task, trial, True, wall=5.0))
                rows.append(_row("slow", task, trial, task == "alpha", wall=40.0))
        _publish_bundle(
            os.path.join(self.site_dir, "releases", "b1"),
            tasks_dir,
            rows,
            title="Two harnesses",
        )
        _write(
            os.path.join(self.site_dir, "releases.json"),
            json.dumps([{
                "id": "b1", "title": "Two harnesses", "date": "2026-07-24",
                "models": ["model-x"], "path": "releases/b1/index.html",
            }]),
        )


class HarnessFamilyTests(_SiteFixture):
    def test_bundle_is_ranked_and_enriched(self):
        doc = site.build_board(self.site_dir, community_dir=None)
        family = doc["harness"]
        self.assertEqual(family["bundle_count"], 1)
        arms = family["bundles"][0]["arms"]
        self.assertEqual([a["harness"] for a in arms], ["fast", "slow"])
        self.assertEqual(arms[0]["solve_rate"], 1.0)
        self.assertEqual(arms[1]["solve_rate"], 0.5)
        # Enrichment: median wall time comes from solved cells only.
        self.assertAlmostEqual(arms[0]["median_wall_s"], 5.0)
        self.assertAlmostEqual(arms[1]["median_wall_s"], 40.0)
        # No configured price for model-x, so cost stays absent rather than 0.
        self.assertIsNone(arms[0]["cost_per_solve_usd"])

    def test_enrichment_matches_published_denominators(self):
        """Enriched rows describe the same cells the solve rate was built from."""
        doc = site.build_board(self.site_dir, community_dir=None)
        for arm in doc["harness"]["bundles"][0]["arms"]:
            self.assertIsNotNone(arm["median_wall_s"])
            self.assertEqual(arm["n"], 4)

    def test_unverified_release_is_skipped_not_ranked(self):
        _write(os.path.join(self.site_dir, "releases", "html-only", "index.html"), "<p>x</p>")
        doc = site.build_board(self.site_dir, community_dir=None)
        self.assertEqual(doc["harness"]["bundle_count"], 1)
        self.assertIn("html-only", [s["id"] for s in doc["harness"]["skipped"]])


class GatewayFamilyTests(_SiteFixture):
    def test_gateway_board_labels_latency_as_median(self):
        metric = {"estimate": 1.0, "low": 0.5, "high": 1.5}
        bundle = {
            "title": "Gateway fixture",
            "track": "fixed_model_provider",
            "blocks_included": 1,
            "blocks_observed": 1,
            "tasks_included": 1,
            "blocks_excluded": {},
            "blocks_max_calls_affected": 0,
            "budget_max_calls": 20,
            "arms": [{
                "arm_id": "direct",
                "role": "direct",
                "requested_provider": "openai",
                "solve_rate": metric,
                "mean_checker_score": metric,
                "availability": metric,
                "latency_s": metric,
                "route_distribution": {},
                "max_calls": {"cells": 0, "total_cells": 1, "ratio": 0.0},
                "cost": None,
            }],
            "contrasts": [{
                "arm_id": "gateway",
                "direct_arm": "direct",
                "solve_rate": metric,
                "mean_checker_score": metric,
                "availability": metric,
                "latency_s": metric,
            }],
        }

        page = site._gateway_board(bundle)

        self.assertIn("Median E2E cell latency", page)
        self.assertIn("Δ median E2E cell latency", page)

    def test_legacy_gateway_family_is_not_built_or_rendered(self):
        legacy = os.path.join(self.site_dir, "gateway", "legacy-workload")
        _write(
            os.path.join(legacy, "provenance.json"),
            json.dumps({"bundle_kind": "gateway_bench"}),
        )
        with mock.patch.object(site, "build_gateway_family") as build_legacy:
            doc = site.build_board(
                self.site_dir,
                gateway_dirs=[os.path.join(self.site_dir, "gateway")],
            )
        build_legacy.assert_not_called()
        self.assertEqual(doc["gateway"]["bundle_count"], 0)
        self.assertNotIn("gateway_probe", doc)
        page = site.render_board_html(doc)
        self.assertNotIn("LEGACY GATEWAY WORKLOAD", page)

    def test_tampered_bundle_fails_verification(self):
        bundle = os.path.join(self.site_dir, "gateway", "tampered")
        _write(os.path.join(bundle, "provenance.json"), json.dumps({
            "schema_version": 1,
            "bundle_kind": "gateway_bench",
            "artifacts": {"results.jsonl": "0" * 64},
        }))
        _write(os.path.join(bundle, "results.jsonl"), "{}\n")
        self.assertIsNotNone(site.gateway_verification_error(bundle))


class GatewayProbeFamilyTests(_SiteFixture):
    def test_checked_in_models_publish_kimi_first_and_gpt4o_as_v3(self):
        repo_docs = str(Path(__file__).resolve().parents[2] / "docs")

        doc = site.build_board(repo_docs)
        bundles = doc["gateway"]["bundles"]

        self.assertEqual(
            [bundle["model"] for bundle in bundles[:2]],
            ["Kimi K3", "GPT-4o mini"],
        )
        self.assertEqual(bundles[0]["result_schema_version"], 4)
        self.assertEqual(bundles[1]["result_schema_version"], 3)
        self.assertIsNone(bundles[1]["retry_summary"])
        self.assertIsNone(bundles[1]["completion_integrity"])
        self.assertIsNone(bundles[1]["output_token_limit_equalities"])
        page = site.render_board_html(doc)
        self.assertIn(
            'id="gateway-model-tab-2026-07-28-kimi-k3-managed-100" '
            'aria-controls="gateway-model-panel-2026-07-28-kimi-k3-managed-100" '
            'aria-selected="true"',
            page,
        )
        self.assertIn(
            'id="gateway-model-tab-2026-07-27-gpt4o-mini-managed-30" '
            'aria-controls="gateway-model-panel-2026-07-27-gpt4o-mini-managed-30" '
            'aria-selected="false"',
            page,
        )

    def _publish_probe(self, bundle_id="probe-v4"):
        from obench.tests.test_gateway_probe_publish import (
            TEST_COMMIT,
            build_private_run,
        )

        source_root = os.path.join(self.root, "probe-source-" + bundle_id)
        os.makedirs(source_root)
        private_run = build_private_run(source_root, max_output_tokens=3)
        bundle = os.path.join(self.site_dir, "gateway-probe", bundle_id)
        gateway_probe_publish.publish_bundle(
            private_run,
            bundle,
            verified_with_commit=TEST_COMMIT,
        )
        return bundle

    def test_verified_request_bundle_is_the_public_gateway_bench(self):
        self._publish_probe()
        _write(
            os.path.join(self.site_dir, "gateway-probe.json"),
            json.dumps([{
                "id": "probe-v4",
                "title": "Gateway Probe: managed request probe",
                "model": "Synthetic model",
                "date": "2026-07-27",
            }]),
        )

        doc = site.build_board(self.site_dir)
        family = doc["gateway"]

        self.assertEqual(family["bundle_count"], 1)
        self.assertNotIn("gateway_probe", doc)
        bundle = family["bundles"][0]
        self.assertEqual(bundle["title"], "Gateway Bench: managed request benchmark")
        self.assertEqual(bundle["model"], "Synthetic model")
        self.assertEqual(bundle["complete_blocks"], {"cold": 2, "warm": 2})
        self.assertEqual(bundle["scheduled_blocks_per_condition"], 2)
        self.assertEqual(bundle["model_match"], "exact_revision")
        self.assertEqual(len(bundle["arms"]), 2)
        self.assertEqual(len(bundle["contrasts"]), 1)

        _, _, facts = site._lede(doc, "gateway")
        self.assertIn("1 published bundle", facts)
        self.assertIn("1 benchmarked model", facts)
        self.assertIn("2 routes per bundle", facts)
        self.assertIn("model-specific denominators below", facts)
        self.assertNotIn("cold blocks", " ".join(facts))
        self.assertNotIn("requests", " ".join(facts))
        self.assertIn("updated 2026-07-27", facts)

        page = site.render_board_html(doc)
        self.assertIn('href="#gateway">Gateway Bench</a>', page)
        self.assertIn("Gateway Bench: managed request benchmark", page)
        self.assertNotIn("Gateway Probe", page)
        self.assertNotIn('id="view-gateway-probe"', page)

    def test_multiple_probe_bundles_render_as_accessible_model_tabs(self):
        self._publish_probe("gpt-bundle")
        self._publish_probe("kimi-bundle")
        _write(
            os.path.join(self.site_dir, "gateway-probe.json"),
            json.dumps([
                {
                    "id": "gpt-bundle",
                    "title": "GPT benchmark",
                    "model": "GPT-4o mini",
                    "date": "2026-07-27",
                },
                {
                    "id": "kimi-bundle",
                    "title": "Kimi benchmark",
                    "model": "Kimi K3",
                    "date": "2026-07-27",
                },
            ]),
        )

        doc = site.build_board(self.site_dir)
        page = site.render_board_html(doc)

        self.assertEqual(
            [bundle["model"] for bundle in doc["gateway"]["bundles"]],
            ["GPT-4o mini", "Kimi K3"],
        )
        self.assertEqual(page.count('role="tab"'), 2)
        self.assertEqual(page.count('role="tabpanel"'), 2)
        self.assertIn('role="tablist" aria-label="Benchmark model"', page)
        self.assertIn(
            'id="gateway-model-tab-gpt-bundle" '
            'aria-controls="gateway-model-panel-gpt-bundle"',
            page,
        )
        self.assertIn(
            'id="gateway-model-panel-kimi-bundle" '
            'aria-labelledby="gateway-model-tab-kimi-bundle"',
            page,
        )
        self.assertIn("GPT benchmark", page)
        self.assertIn("Kimi benchmark", page)
        self.assertIn("2 cold + 2 warm matched blocks", page)
        self.assertIn('ev.key === "ArrowRight"', page)
        self.assertIn('ev.key === "Home"', page)
        self.assertIn("selectGatewayModel(modelTabs[0], false)", page)

    def test_run_note_is_carried_and_escaped_below_evidence_depth(self):
        self._publish_probe()
        run_note = (
            'Recovered split-session run; <script>alert("unsafe")</script> '
            "remained outside metrics."
        )
        _write(
            os.path.join(self.site_dir, "gateway-probe.json"),
            json.dumps([{
                "id": "probe-v4",
                "title": "Gateway benchmark",
                "run_note": run_note,
            }]),
        )

        doc = site.build_board(self.site_dir)
        bundle = doc["gateway"]["bundles"][0]
        page = site.render_board_html(doc)

        self.assertEqual(bundle["run_note"], run_note)
        self.assertIn(
            "&lt;script&gt;alert(&quot;unsafe&quot;)&lt;/script&gt;",
            page,
        )
        self.assertNotIn("<script>alert", page)
        self.assertLess(page.index("Evidence depth:"), page.index("Run note:"))
        self.assertLess(page.index("Run note:"), page.index("Gateway leaderboard"))

    def test_run_note_is_absent_when_unspecified_or_malformed(self):
        bundle_dir = self._publish_probe()

        bundle = site.aggregate_gateway_probe_bundle(bundle_dir)
        self.assertNotIn("run_note", bundle)
        self.assertNotIn("Run note:", site._gateway_probe_board(bundle))

        for run_note in ({"html": "<script>alert(1)</script>"}, "x" * 501):
            with self.subTest(run_note_type=type(run_note).__name__):
                bundle = site.aggregate_gateway_probe_bundle(
                    bundle_dir,
                    manifest_entry={"run_note": run_note},
                )
                self.assertNotIn("run_note", bundle)
                self.assertNotIn("Run note:", site._gateway_probe_board(bundle))

    def test_single_block_bundle_is_plainly_labelled_as_a_spike(self):
        bundle = site.aggregate_gateway_probe_bundle(self._publish_probe())
        bundle["complete_blocks"] = {"cold": 1, "warm": 1}
        bundle["scheduled_blocks_per_condition"] = 1
        bundle["baseline_arm_id"] = "direct-moonshot"

        page = site._gateway_probe_board(bundle)

        self.assertIn("Evidence depth: 1 cold + 1 warm matched blocks per route.", page)
        self.assertIn("This is a spike denominator", page)
        self.assertIn("not maturity-equivalent", page)
        self.assertIn('class="evidence-depth is-spike"', page)
        self.assertIn("Every delta is gateway minus Direct Moonshot.", page)

    def test_completion_integrity_requires_explicit_finish_reason_evidence(self):
        rows = [{
            "identity": {
                "arm": {"id": "legacy-route"},
                "schedule": {"condition": "warm"},
            },
            "request_metrics": {
                "stream": {"terminal_status": "completed"},
                "usage": {"output_tokens": 3},
            },
            "reuse_evidence": {"stream": {"done": True}},
        }]

        self.assertIsNone(site._gateway_probe_completion_integrity(rows))

        rows[0]["request_metrics"]["stream"]["finish_reason"] = None
        integrity = site._gateway_probe_completion_integrity(rows)

        self.assertEqual(
            integrity["arms"]["legacy-route"]["warm"]["measured"],
            {
                "natural_stop": 0,
                "length": 0,
                "missing": 1,
                "other": 0,
                "total": 1,
            },
        )
        self.assertEqual(
            integrity["arms"]["legacy-route"]["warm"]["warm_primer"],
            {
                "natural_stop": 0,
                "length": 0,
                "missing": 1,
                "other": 0,
                "total": 1,
            },
        )

    def test_completion_integrity_aggregates_and_renders_future_rows(self):
        bundle_dir = self._publish_probe()
        rows = site._read_jsonl(os.path.join(bundle_dir, "results.jsonl"))
        measured_reasons = {
            ("direct", "cold", 1): "stop",
            ("direct", "cold", 2): "length",
            ("direct", "warm", 1): None,
            ("direct", "warm", 2): "tool_calls",
            ("gateway", "cold", 1): "stop",
            ("gateway", "cold", 2): "stop",
            ("gateway", "warm", 1): "length",
            ("gateway", "warm", 2): None,
        }
        primer_reasons = {
            ("direct", 1): "stop",
            ("direct", 2): "length",
            ("gateway", 1): "stop",
            ("gateway", 2): None,
        }
        for row in rows:
            identity = row["identity"]
            arm_id = identity["arm"]["id"]
            schedule = identity["schedule"]
            condition = schedule["condition"]
            repetition = schedule["repetition"]
            row["request_metrics"]["stream"]["finish_reason"] = (
                measured_reasons[(arm_id, condition, repetition)]
            )
            if condition == "warm":
                row["reuse_evidence"]["stream"] = {
                    "finish_reason": primer_reasons[(arm_id, repetition)],
                }

        with mock.patch.object(site, "_read_jsonl", return_value=rows):
            bundle = site.aggregate_gateway_probe_bundle(
                bundle_dir,
            )

        integrity = bundle["completion_integrity"]["arms"]
        self.assertEqual(integrity["direct"]["cold"]["measured"], {
            "natural_stop": 1,
            "length": 1,
            "missing": 0,
            "other": 0,
            "total": 2,
        })
        self.assertEqual(integrity["direct"]["warm"]["measured"], {
            "natural_stop": 0,
            "length": 0,
            "missing": 1,
            "other": 1,
            "total": 2,
        })
        self.assertEqual(integrity["gateway"]["warm"]["warm_primer"], {
            "natural_stop": 1,
            "length": 0,
            "missing": 1,
            "other": 0,
            "total": 2,
        })

        page = site._gateway_probe_board(bundle)
        self.assertIn("Completion integrity", page)
        self.assertIn("Measured natural-stop", page)
        self.assertIn("Measured length", page)
        self.assertIn("Measured missing", page)
        self.assertIn("Measured other", page)
        self.assertIn("Warm-primer natural-stop", page)
        self.assertIn("Natural stop means the response ended normally", page)
        self.assertIn("provider reported a length-based termination", page)
        self.assertIn("missing means no finish reason was reported", page)
        self.assertLess(
            page.index("Gateway leaderboard"),
            page.index("Completion integrity"),
        )
        completion_html = page[page.index("Completion integrity"):]
        self.assertEqual(completion_html.count(">1/2<"), 2)
        self.assertNotIn("Configured output-limit equality", page)

    def test_output_limit_disclosure_is_data_driven(self):
        bundle = site.aggregate_gateway_probe_bundle(
            self._publish_probe(),
        )

        self.assertIsNotNone(bundle["completion_integrity"])
        disclosure = bundle["output_token_limit_equalities"]
        self.assertEqual(disclosure["configured_limit"], 3)
        self.assertEqual(disclosure["arms"]["direct"]["cold"], {
            "equal": 2,
            "measured": 2,
        })
        self.assertEqual(disclosure["arms"]["gateway"]["warm"], {
            "equal": 2,
            "measured": 2,
        })

        page = site._gateway_probe_board(bundle)
        self.assertIn("Completion integrity", page)
        self.assertNotIn("Configured output-limit equality", page)
        self.assertNotIn("Configured request output limit", page)
        self.assertNotIn("Cold responses at 3-token limit", page)
        self.assertNotIn("Warm responses at 3-token limit", page)

    def test_output_limit_disclosure_ignores_page_metadata_override(self):
        bundle_dir = self._publish_probe()

        for value in (None, True, 0, -1, "3", 3.0):
            with self.subTest(value=value):
                entry = {} if value is None else {"output_token_limit": value}
                bundle = site.aggregate_gateway_probe_bundle(
                    bundle_dir,
                    manifest_entry=entry,
                )
                self.assertEqual(
                    bundle["output_token_limit_equalities"]["configured_limit"],
                    3,
                )
                page = site._gateway_probe_board(bundle)
                self.assertNotIn("Configured output-limit equality", page)
                self.assertNotIn("Cold responses at 3-token limit", page)

    def test_output_limit_disclosure_counts_only_measured_output_tokens(self):
        rows = [
            {
                "identity": {
                    "arm": {"id": "route"},
                    "schedule": {"condition": "cold"},
                },
                "request_metrics": {"usage": {"output_tokens": 128}},
            },
            {
                "identity": {
                    "arm": {"id": "route"},
                    "schedule": {"condition": "cold"},
                },
                "request_metrics": {"usage": {"output_tokens": None}},
            },
        ]

        disclosure = site._output_token_limit_equalities(rows, 128)

        self.assertEqual(disclosure["arms"]["route"]["cold"], {
            "equal": 1,
            "measured": 1,
        })

    def test_retry_evidence_is_derived_from_verified_public_rows(self):
        bundle_dir = self._publish_probe()
        rows_path = os.path.join(bundle_dir, "results.jsonl")
        with open(rows_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]

        for index, row in enumerate(rows):
            success = index % 4 != 0
            rescued = index % 4 == 1
            attempts = [{
                "attempt_number": 1,
                "retry": {
                    "retry_after_s": 1.5 if rescued else None,
                    "wait_actual_s": 1.6 if rescued else None,
                },
            }]
            if rescued or not success:
                attempts.append({
                    "attempt_number": 2,
                    "retry": {
                        "retry_after_s": None,
                        "wait_actual_s": None,
                    },
                })
            row["retry_evidence"] = {
                "max_total_attempts": 2,
                "max_output_tokens": 3,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "first_attempt_outcome": {"success": success and not rescued},
                "eventual_outcome": {"success": success},
                "recovered": rescued,
                "recovery_timing": {
                    "initial_request_to_completion_s": 2.5,
                    "final_attempt_request_start_offset_s": 1.75,
                },
            }
            row["request_metrics"]["usage"]["output_tokens"] = (
                3 if index % 3 == 0 else 2
            )

        stale = json.loads(json.dumps(rows[0]))
        stale_schedule = stale["identity"]["schedule"]
        for row in rows:
            schedule = row["identity"]["schedule"]
            if (
                row["identity"]["case"]["id"]
                == stale["identity"]["case"]["id"]
                and schedule["condition"] == stale_schedule["condition"]
                and schedule["repetition"] == stale_schedule["repetition"]
            ):
                schedule["block_attempt"] += 1
        stale["retry_evidence"]["attempt_count"] = 2
        stale["retry_evidence"]["attempts"].append({
            "attempt_number": 2,
            "retry": {
                "retry_after_s": 99.0,
                "wait_actual_s": 99.0,
            },
        })
        rows.append(stale)

        manifest = {
            "verification": {"verified_with_commit": "f" * 40},
            "result_count": len(rows),
            "result_schema_version": 4,
        }
        with mock.patch.object(
            gateway_probe_publish, "verify_bundle", return_value=manifest
        ) as verify:
            with mock.patch.object(site, "_read_jsonl", return_value=rows):
                bundle = site.aggregate_gateway_probe_bundle(bundle_dir)

        verify.assert_called_once_with(bundle_dir)
        retry = bundle["retry_summary"]["overall"]
        self.assertEqual(retry["logical_cells"], 8)
        self.assertEqual(retry["first_attempt_successes"], 4)
        self.assertEqual(retry["eventual_successes"], 6)
        self.assertEqual(retry["retried"], 4)
        self.assertEqual(retry["rescued"], 2)
        self.assertEqual(retry["exhausted"], 2)
        self.assertEqual(retry["attempts"], 12)
        self.assertEqual(
            retry["attempt_distribution"], {"1": 4, "2": 4}
        )
        self.assertEqual(retry["retry_after_s"]["count"], 2)
        self.assertEqual(retry["wait_actual_s"]["median"], 1.6)
        self.assertEqual(
            retry["output_limits"]["3"], {"equal": 3, "measured": 8}
        )
        self.assertEqual(
            bundle["output_token_limit_equalities"]["configured_limit"],
            3,
        )

        page = site._gateway_probe_board(bundle)
        self.assertNotIn("Retry evidence:", page)
        self.assertNotIn("physical attempts across", page)
        self.assertNotIn("First → eventual success", page)
        self.assertNotIn("Configured request output limit", page)

    def test_retry_summary_rejects_partial_or_malformed_evidence(self):
        base = {
            "identity": {
                "arm": {"id": "route"},
                "schedule": {"condition": "cold"},
            },
        }
        self.assertIsNone(site._gateway_probe_retry_summary([base]))

        malformed = dict(base)
        malformed["retry_evidence"] = {
            "max_total_attempts": 2,
            "max_output_tokens": 7,
            "attempt_count": 2,
            "attempts": [{}],
            "first_attempt_outcome": {},
            "eventual_outcome": {},
            "recovered": False,
            "recovery_timing": {},
        }
        self.assertIsNone(
            site._gateway_probe_retry_summary([malformed])
        )

    def test_tampered_probe_bundle_fails_closed(self):
        bundle = self._publish_probe()
        _write(os.path.join(bundle, "report.json"), "{}\n")

        doc = site.build_board(self.site_dir)

        self.assertEqual(doc["gateway"]["bundle_count"], 0)
        self.assertEqual(
            [item["id"] for item in doc["gateway"]["skipped"]],
            ["probe-v4"],
        )
        self.assertIn(
            "bundle verification failed",
            doc["gateway"]["skipped"][0]["reason"],
        )

    def test_probe_board_renders_factual_30_by_30_contract(self):
        bundle_dir = self._publish_probe()
        bundle = site.aggregate_gateway_probe_bundle(bundle_dir)
        self.assertIsNotNone(bundle)
        bundle["complete_blocks"] = {"cold": 30, "warm": 30}
        bundle["scheduled_blocks_per_condition"] = 30
        availability_low, availability_high = site.stats.wilson_ci(30, 30)
        for arm in bundle["arms"]:
            for condition in ("cold", "warm"):
                item = arm["conditions"][condition]
                item["denominators"].update({
                    "scheduled": 30,
                    "attempted": 30,
                    "success": 30,
                    "route_verified": 30,
                })
                item["availability"].update({
                    "successes": 30,
                    "attempted": 30,
                    "rate": 1.0,
                    "wilson95": {
                        "confidence": 0.95,
                        "low": availability_low,
                        "high": availability_high,
                    },
                })
                for name, metric in item["metrics"].items():
                    covered = 0 if name in {
                        "cached_input_tokens",
                        "cache_write_input_tokens",
                    } else 30
                    metric["coverage"] = {
                        "covered": covered,
                        "total": 30,
                        "ratio": covered / 30,
                    }
        for contrast in bundle["contrasts"]:
            for condition in ("cold", "warm"):
                for metric in contrast["conditions"][condition].values():
                    metric["coverage"] = {
                        "covered": 30,
                        "total": 30,
                        "ratio": 1.0,
                    }

        page = site._gateway_probe_board(bundle)

        for label in (
            "Cold requests",
            "Warm requests",
            "OpenBench Composite",
            "TTFT p50 / p95",
            "Response headers p50 / p95",
            "First body byte p50 / p95",
            "Stream total p50 / p95",
            "Throughput tok/s p50 / p95",
            "Total / cached / cache-write tokens p50 / p95",
            "Gateway leaderboard",
            "Cold setup",
            "DNS p50 / p95",
            "TCP p50 / p95",
            "TLS p50 / p95",
            "Paired request deltas",
            "Δ response headers",
            "Δ TTFT",
        ):
            self.assertIn(label, page)
        request_labels = (
            "Route",
            "TTFT p50 / p95",
            "Stream total p50 / p95",
            "Response headers p50 / p95",
            "First body byte p50 / p95",
            "Throughput tok/s p50 / p95",
            "Total / cached / cache-write tokens p50 / p95",
        )
        cold_table = page[
            page.index("Cold requests"):page.index("Warm requests")
        ]
        warm_table = page[
            page.index("Warm requests"):page.index("Cold setup")
        ]
        for table in (cold_table, warm_table):
            positions = [table.index(label) for label in request_labels]
            self.assertEqual(positions, sorted(positions))
        self.assertIn("<b>cold blocks </b>30/30", page)
        self.assertIn("<b>warm blocks </b>30/30", page)
        self.assertIn("Complete blocks: 30/30.", page)
        self.assertIn(
            "TTFT begins when the measured request is sent; connection "
            "setup is reported separately.",
            page,
        )
        self.assertIn("95% CI", page)
        self.assertIn("(0/30)", page)
        self.assertNotIn("Success / availability", page)
        self.assertNotIn("Route verification", page)
        self.assertNotIn("verified 30/30", page)
        self.assertNotIn("unverifiable", page)
        self.assertNotIn("coverage 30/30", page)
        self.assertNotIn("Cost p50 / p95", page)
        self.assertIn('class="route-logo"', page)
        self.assertIn('class="provider-cell"', page)
        self.assertNotIn("<img", page)
        self.assertNotIn("Semantic TTFT p50 / p95", page)
        self.assertNotIn("CI 88.6%–100.0%", page)
        self.assertNotIn("TTFB", page)
        self.assertNotIn("exploratory", page.lower())
        self.assertNotIn("confirmatory", page.lower())
        self.assertNotIn("Δ stream total", page)
        self.assertNotIn("Provider variance", page)
        paired = page[page.index("Paired request deltas"):]
        self.assertIn("Every delta is gateway minus direct.", paired)
        self.assertIn("positive means slower/worse", paired)
        self.assertIn("negative means faster/better", paired)
        self.assertNotIn(">vs direct<", paired)
        self.assertLess(
            page.index("Gateway leaderboard"),
            page.index("Cold requests"),
        )
        self.assertGreater(
            page.index("Completion integrity"),
            page.index("Paired request deltas"),
        )
        self.assertNotIn("Retry evidence:", page)
        self.assertNotIn("Configured output-limit equality", page)

    def test_probe_route_labels_hide_provider_suffix(self):
        labels = {
            "cloudflare-openai": "Cloudflare",
            "concentrate-openai": "Concentrate",
            "direct-openai": "Direct OpenAI",
            "direct-moonshot": "Direct Moonshot",
            "cloudflare-moonshot": "Cloudflare",
            "concentrate-moonshot": "Concentrate",
            "openrouter-moonshot": "OpenRouter",
            "vercel-moonshot": "Vercel",
            "openrouter-openai": "OpenRouter",
            "vercel-openai": "Vercel",
            "custom-route": "custom-route",
        }
        for arm_id, expected in labels.items():
            self.assertEqual(site._gateway_probe_route_name(arm_id), expected)

    def _composite_bundle(self):
        def arm(arm_id, cold, warm, throughput, successes):
            def condition(ttft, success):
                coverage = {
                    "covered": success,
                    "total": 10,
                    "ratio": success / 10,
                }
                return {
                    "denominators": {
                        "scheduled": 10,
                        "attempted": 10,
                        "success": success,
                        "route_verified": success,
                    },
                    "availability": {
                        "successes": success,
                        "attempted": 10,
                    },
                    "metrics": {
                        "request_to_semantic_ttft_s": {
                            "p50": ttft[0],
                            "p95": ttft[1],
                            "coverage": coverage,
                        },
                        "cold_end_to_end_semantic_ttft_s": {
                            "p50": ttft[0],
                            "p95": ttft[1],
                            "coverage": coverage,
                        },
                        "throughput_tokens_per_s": {
                            "p50": throughput,
                            "p95": throughput,
                            "coverage": coverage,
                        },
                    },
                }

            cold_condition = condition(cold, successes[0])
            cold_condition["metrics"][
                "cold_end_to_end_semantic_ttft_s"
            ].update({
                "p50": cold[0] + 5.0,
                "p95": cold[1] + 5.0,
            })
            return {
                "arm_id": arm_id,
                "conditions": {
                    "cold": cold_condition,
                    "warm": condition(warm, successes[1]),
                },
            }

        return {
            "baseline_arm_id": "direct-openai",
            "arms": [
                arm("direct-openai", (1.0, 2.0), (1.0, 2.0), 200.0, (10, 10)),
                arm("vercel-openai", (2.0, 4.0), (1.0, 3.0), 200.0, (9, 9)),
            ],
        }

    def test_probe_composite_uses_absolute_scales_and_linear_success(self):
        bundle = self._composite_bundle()
        scores = {
            row["arm_id"]: row["score"]
            for row in site._gateway_probe_composite_scores(bundle)
        }

        # Latency: 30%*90 + 15%*80 + 30%*95 + 15%*85.
        # Throughput: 10%*100. Combined request success is 18/20.
        self.assertAlmostEqual(scores["vercel-openai"], 81.225)
        self.assertEqual(
            site._GATEWAY_COMPOSITE_WEIGHTS,
            {
                ("cold", "p50"): 0.30,
                ("cold", "p95"): 0.15,
                ("warm", "p50"): 0.30,
                ("warm", "p95"): 0.15,
            },
        )

        # The gateway score is absolute: changing Direct does not change it.
        direct = bundle["arms"][0]
        for condition in direct["conditions"].values():
            for name in (
                "request_to_semantic_ttft_s",
                "cold_end_to_end_semantic_ttft_s",
            ):
                condition["metrics"][name].update({
                    "p50": 19.0,
                    "p95": 20.0,
                })
            condition["metrics"]["throughput_tokens_per_s"]["p50"] = 5.0
        changed = {
            row["arm_id"]: row["score"]
            for row in site._gateway_probe_composite_scores(bundle)
        }
        self.assertAlmostEqual(changed["vercel-openai"], 81.225)

    def test_probe_leaderboard_excludes_direct_and_cost(self):
        page = site._gateway_probe_leaderboard(self._composite_bundle())

        self.assertIn("OpenBench Composite", page)
        self.assertIn("Vercel", page)
        self.assertIn(">81.2<", page)
        self.assertNotIn("price", page.lower())
        self.assertEqual(page.count('class="route-position"'), 1)
        ranked_rows = page[page.index('class="route-leaderboard"'):]
        self.assertNotIn("Direct OpenAI", ranked_rows)
        self.assertNotIn("cost", ranked_rows.lower())

    def test_probe_leaderboard_names_the_bundle_baseline(self):
        bundle = self._composite_bundle()
        bundle["baseline_arm_id"] = "direct-moonshot"
        bundle["arms"][0]["arm_id"] = "direct-moonshot"

        page = site._gateway_probe_leaderboard(bundle)

        self.assertIn("Direct Moonshot is an unranked reference.", page)
        self.assertNotIn("Direct OpenAI is an unranked reference.", page)

    def test_probe_composite_withholds_incomplete_verified_telemetry(self):
        bundle = self._composite_bundle()
        gateway = bundle["arms"][1]
        gateway["conditions"]["warm"]["denominators"]["route_verified"] = 8

        scores = {
            row["arm_id"]: row["score"]
            for row in site._gateway_probe_composite_scores(bundle)
        }

        self.assertNotIn("vercel-openai", scores)
        self.assertIn("direct-openai", scores)

    def test_probe_leaderboard_wraps_evidence_on_small_screens(self):
        mobile_css = site._CSS[site._CSS.index("@media(max-width:680px){"):]
        self.assertIn(".route-detail{white-space:normal}", mobile_css)


class CostBasisTests(unittest.TestCase):
    def _basis(self, covered, total=10, per_solve=1.0):
        return {
            "attempted_cost_usd": {"estimate": per_solve},
            "cost_per_solve_usd": per_solve,
            "basis_coverage": {
                "covered_calls": covered,
                "total_calls": total,
                "ratio": covered / total,
                "complete": covered == total,
            },
        }

    def test_common_cost_basis_requires_complete_frozen_list_for_every_arm(self):
        complete = {"costs": {"frozen_list_estimate": self._basis(10)}}
        incomplete = {"costs": {"frozen_list_estimate": self._basis(9)}}

        self.assertEqual(
            site._common_complete_cost_basis([complete, complete]),
            "frozen_list_estimate",
        )
        self.assertIsNone(
            site._common_complete_cost_basis([complete, incomplete])
        )

    def test_gateway_reported_never_substitutes_for_missing_frozen_list(self):
        arms = [
            {"costs": {"frozen_list_estimate": self._basis(10)}},
            {"costs": {"gateway_reported": self._basis(10)}},
        ]

        self.assertIsNone(site._common_complete_cost_basis(arms))

    def test_zero_solve_arm_keeps_complete_attempted_cost_basis(self):
        zero_solve = self._basis(10, per_solve=None)

        self.assertEqual(
            site._common_complete_cost_basis([
                {"costs": {"frozen_list_estimate": zero_solve}},
            ]),
            "frozen_list_estimate",
        )

    def test_cost_dto_preserves_separate_evidence_bases(self):
        costs = site._cost_dtos({
            "gateway_reported": self._basis(10, per_solve=2.0),
            "frozen_list_estimate": self._basis(10, per_solve=1.0),
        })

        self.assertEqual(set(costs), {"gateway_reported", "frozen_list_estimate"})
        self.assertEqual(costs["gateway_reported"]["cost_per_solve_usd"], 2.0)
        self.assertEqual(costs["frozen_list_estimate"]["cost_per_solve_usd"], 1.0)

    def test_metric_dto_preserves_metric_coverage(self):
        dto = site._metric_dto({
            "estimate": 3.0,
            "interval": {"low": 2.0, "high": 4.0},
            "call_coverage": {"covered": 8, "total": 10, "ratio": 0.8},
            "paired_block_coverage": {"covered": 4, "total": 5, "ratio": 0.8},
        })

        self.assertEqual(dto["call_coverage"]["covered"], 8)
        self.assertEqual(dto["paired_block_coverage"]["total"], 5)


class RenderTests(_SiteFixture):
    def test_gateway_board_renders_call_cap_denominators(self):
        metric = {"estimate": 1.0, "low": 0.8, "high": 1.0}
        bundle = {
            "title": "Gateway run",
            "track": "fixed_model_provider",
            "harness": "pi",
            "blocks_included": 5,
            "blocks_observed": 5,
            "blocks_excluded": {},
            "blocks_max_calls_affected": 2,
            "tasks_included": 1,
            "budget_max_calls": 20,
            "arms": [{
                "arm_id": "direct",
                "role": "direct",
                "requested_provider": "openai",
                "solve_rate": metric,
                "mean_checker_score": metric,
                "availability": metric,
                "latency_s": metric,
                "route_distribution": {},
                "max_calls": {"cells": 2, "total_cells": 5, "ratio": 0.4},
                "cost": None,
            }],
            "contrasts": [],
        }

        page = site._gateway_board(bundle)

        self.assertIn("<b>cap-affected blocks </b>2/5", page)
        self.assertIn("20-call cap", page)
        self.assertIn("2/5 (40.0%)", page)

    def test_gateway_board_splits_outcomes_from_serving_telemetry(self):
        metric = {
            "estimate": 0.5,
            "low": 0.4,
            "high": 0.6,
            "cell_coverage": {"covered": 5, "total": 5, "ratio": 1.0},
            "call_coverage": {"covered": 9, "total": 10, "ratio": 0.9},
            "paired_block_coverage": {"covered": 4, "total": 5, "ratio": 0.8},
        }
        arm = {
            "arm_id": "gateway",
            "role": "gateway",
            "requested_provider": "openai",
            "solve_rate": metric,
            "mean_checker_score": metric,
            "availability": metric,
            "latency_s": metric,
            "ttfb_s": metric,
            "semantic_ttft_s": metric,
            "throughput_tokens_per_s": metric,
            "mean_input_tokens_per_call": metric,
            "mean_output_tokens_per_call": metric,
            "cache_hit_call_rate": metric,
            "cached_input_fraction": metric,
            "mean_cached_input_tokens_per_call": metric,
            "mean_cache_write_input_tokens_per_call": {
                **metric,
                "estimate": None,
                "call_coverage": {"covered": 0, "total": 10, "ratio": 0.0},
            },
            "route_distribution": {
                "OpenAI/gpt-4o-mini-2024-07-18": {
                    "share": 1.0,
                    "task_coverage": {"covered": 4, "total": 5, "ratio": 0.8},
                },
            },
            "max_calls": {"cells": 1, "total_cells": 5, "ratio": 0.2},
            "cost": {
                "cost_per_solve_usd": 0.0123,
                "coverage": {"covered_calls": 10, "total_calls": 10},
            },
        }
        contrast = {
            "arm_id": "gateway",
            "direct_arm": "direct",
            "solve_rate": metric,
            "mean_checker_score": metric,
            "availability": metric,
            "latency_s": metric,
            "ttfb_s": metric,
            "semantic_ttft_s": metric,
            "throughput_tokens_per_s": metric,
            "mean_input_tokens_per_call": metric,
            "mean_output_tokens_per_call": metric,
            "attempted_cost_usd": metric,
        }
        bundle = {
            "title": "Gateway run",
            "track": "gateway_tax",
            "model_match": "rolling_alias",
            "provider_prompt_mode": "provider_default",
            "blocks_included": 5,
            "blocks_observed": 5,
            "blocks_excluded": {},
            "blocks_max_calls_affected": 1,
            "tasks_included": 1,
            "budget_max_calls": 20,
            "arms": [arm],
            "contrasts": [contrast],
        }

        page = site._gateway_board(bundle)

        for label in (
            "Median E2E cell latency",
            "List-est. $/solve",
            "Serving telemetry",
            "Served route",
            "TTFB",
            "Semantic TTFT",
            "Output tok/s",
            "Cache read / write tok/call",
            "Serving telemetry tax",
            "Δ list cost / attempted cell",
        ):
            self.assertIn(label, page)
        self.assertIn("rolling_alias", page)
        self.assertIn("provider_default", page)
        self.assertIn("OpenAI/gpt-4o-mini-2024-07-18", page)
        self.assertIn("100% · tasks 4/5", page)
        self.assertIn('class="provider-cell"', page)
        self.assertIn('class="route-logo"', page)
        self.assertIn("cache write 0/10", page)
        self.assertIn("gateway response caching was disabled", page)
        self.assertIn("not cost per solve", page)
        self.assertNotIn("Δ list cost / solve", page)

    def test_harness_token_columns_render_factual_split_semantics(self):
        page = site.render_board_html(site.build_board(self.site_dir))
        for label in (
            "Fresh tokens/solve",
            "Uncached input/solve",
            "Output/solve",
            "Cache-read/solve",
            "Cache-write/solve",
            "Telemetry source / basis",
            "Telemetry coverage",
        ):
            self.assertIn(label, page)
        self.assertIn(">100<", page)
        self.assertIn(">80<", page)
        self.assertIn(">20<", page)
        self.assertIn(">50<", page)
        self.assertIn(">5<", page)
        self.assertIn(">proxy 0/4 · native 4/4<", page)
        self.assertIn(">native<", page)
        self.assertIn(">vendor_split<", page)
        self.assertNotIn(">Tokens/solve<", page)
        self.assertNotIn("cache-hit", page.lower())

    def test_native_fallback_renders_partial_proxy_coverage(self):
        doc = site.build_board(self.site_dir)
        arm = doc["harness"]["bundles"][0]["arms"][0]
        arm["token_telemetry_coverage"]["proxy_covered_rows"] = 2
        page = site.render_board_html(doc)
        row = page[page.index('data-harness="fast"'):]
        row = row[:row.index("</tr>")]
        self.assertIn(">native<", row)
        self.assertIn(">proxy 2/4 · native 4/4<", row)

    def test_incomplete_token_telemetry_renders_coverage_and_no_metrics(self):
        doc = site.build_board(self.site_dir)
        arm = doc["harness"]["bundles"][0]["arms"][0]
        for field in (
            "fresh_tokens_per_solve",
            "tokens_input_uncached_per_solve",
            "tokens_output_per_solve",
            "tokens_cache_read_per_solve",
            "tokens_cache_write_per_solve",
        ):
            arm[field] = None
        arm["token_telemetry_source"] = None
        arm["token_telemetry_bases"] = []
        arm["token_telemetry_coverage"] = {
            "total_rows": 4,
            "covered_rows": 0,
            "ratio": 0.0,
            "proxy_covered_rows": 1,
            "native_covered_rows": 2,
        }
        page = site.render_board_html(doc)
        row = page[page.index('data-harness="fast"'):]
        row = row[:row.index("</tr>")]
        self.assertEqual(row.count("<td>—</td>"), 5)
        self.assertIn(">unavailable<", row)
        self.assertIn(">proxy 1/4 · native 2/4<", row)

    def test_page_loads_no_external_resources(self):
        """Self-contained means no resource *loads*, not no links.

        A community bundle legitimately links its source on github.com, so
        forbidding the substring "https://" outright would be a false
        invariant that only passes while no fixture has a link.
        """
        doc = site.build_board(self.site_dir)
        page = site.render_board_html(doc)
        self.assertIn("<!doctype html>", page)
        for forbidden in (
            "<script src", "<link rel=\"stylesheet",
            "<img", "<iframe", "@import", "url(http", "src=\"http",
            "fetch(", "XMLHttpRequest", "WebSocket", "cdn.",
        ):
            self.assertNotIn(forbidden, page)

    def test_contact_links_to_issue_creation_and_x(self):
        page = site.render_board_html(site.build_board(self.site_dir))

        self.assertIn(">Contact<", page)
        self.assertIn(
            'href="https://github.com/minghinmatthewlam/openbench/issues/new"',
            page,
        )
        self.assertIn('href="https://x.com/mattlam_"', page)

    def test_site_metadata_emits_safe_social_preview_tags(self):
        _write(
            os.path.join(self.site_dir, "site-meta.json"),
            json.dumps({
                "canonical_url": "https://openbench.example/",
                "social_image_url": "https://openbench.example/og.png",
            }),
        )
        doc = site.build_board(self.site_dir)
        page = site.render_board_html(doc)

        self.assertEqual(doc["schema_version"], 5)
        self.assertIn(
            '<link rel="canonical" href="https://openbench.example/">',
            page,
        )
        self.assertIn(
            '<meta property="og:image" '
            'content="https://openbench.example/og.png">',
            page,
        )

        doc["site_metadata"] = {
            "canonical_url": "javascript:alert(1)",
            "social_image_url": "data:image/png,unsafe",
        }
        unsafe_page = site.render_board_html(doc)
        self.assertNotIn("javascript:", unsafe_page)
        self.assertNotIn("data:image", unsafe_page)

    def test_hostile_manifest_cannot_produce_a_scripting_href(self):
        """The renderer enforces the scheme itself.

        Ingest validates community links, but `community.json`, `releases.json`
        and `gateway.json` are committed files a pull request can edit
        directly, so the guard cannot live only at ingest.
        """
        import re
        doc = site.build_board(self.site_dir)
        doc["community"] = [{
            "id": "evil", "title": "Evil bundle", "submitter": "someone",
            "path": "javascript:alert(1)",
            "link": "JaVaScRiPt:alert(document.cookie)",
        }]
        doc["releases"] = [{
            "id": "r", "title": "Rel", "path": "  javascript:alert(2)",
        }]
        doc["harness"]["bundles"][0]["path"] = "data:text/html,<script>x</script>"
        doc["harness"]["bundles"][0]["results_path"] = "//evil.example/x"
        page = site.render_board_html(doc)

        for href in re.findall(r'href="([^"]*)"', page):
            self.assertFalse(
                href.lower().replace("\t", "").strip().startswith(
                    ("javascript:", "data:", "vbscript:", "//")),
                f"unsafe href rendered: {href!r}",
            )
        # Dropping the link must not drop the text.
        self.assertIn("Evil bundle", page)
        self.assertIn("results.jsonl", page)

    def test_safe_href_accepts_relative_and_http_only(self):
        for allowed in ("https://e.example/x", "http://e.example",
                        "releases/a/index.html", "/abs", "#packs"):
            self.assertEqual(site._safe_href(allowed), allowed.strip())
        for blocked in ("javascript:alert(1)", "JaVaScRiPt:alert(1)",
                        "  javascript:alert(1)", "java\tscript:alert(1)",
                        "data:text/html,x", "vbscript:x", "//evil.example",
                        "mailto:a@b.c", "", None, 42):
            self.assertIsNone(site._safe_href(blocked), blocked)

    def test_anchor_links_are_allowed_and_scheme_checked(self):
        """External links are fine; they must still be http(s)."""
        import re
        doc = site.build_board(self.site_dir)
        doc["community"] = [{
            "id": "c1", "title": "Community bundle", "submitter": "someone",
            "path": "community/c1/index.html",
            "link": "https://example.com/proof",
        }]
        page = site.render_board_html(doc)
        self.assertIn('href="https://example.com/proof"', page)
        for href in re.findall(r'href="([^"]+)"', page):
            self.assertFalse(
                href.lower().startswith(("javascript:", "data:", "vbscript:")),
                f"unsafe scheme in href: {href}",
            )

    def test_bundle_supplied_text_is_escaped(self):
        """Titles and caveats come from bundles; they are content, not markup."""
        doc = site.build_board(self.site_dir)
        bundle = doc["harness"]["bundles"][0]
        bundle["title"] = '<img src=x onerror="alert(1)">'
        bundle["caveats"] = ["</table><script>alert(2)</script>"]
        bundle["has_caveats"] = True
        page = site.render_board_html(doc)
        self.assertNotIn("<img src=x", page)
        self.assertNotIn("<script>alert(2)", page)
        self.assertIn("&lt;img src=x", page)

    def test_tables_are_rendered_without_javascript(self):
        """The script only enhances; the data must be in the document."""
        page = site.render_board_html(site.build_board(self.site_dir))
        head, _, script = page.partition("<script>")
        self.assertIn(
            '<main id="view-harness"><div class="lede" '
            'data-lede="harness">', head
        )
        self.assertIn(
            '<main id="view-gateway"><div class="lede" '
            'data-lede="gateway">', head
        )
        for view in ("releases", "methodology", "contact"):
            self.assertIn(
                f'<main id="view-{view}"><div class="lede" '
                'data-lede="general">',
                head,
            )
        self.assertIn("<tbody>", head)
        self.assertIn('data-harness="fast"', head)
        self.assertIn('data-harness="slow"', head)
        # Two arms rendered as two real rows, before any JS runs.
        self.assertEqual(head.count("<tr data-harness="), 2)
        self.assertNotIn("<tbody>", script)

    def test_public_families_and_methodology_use_gateway_bench_only(self):
        page = site.render_board_html(site.build_board(self.site_dir))
        self.assertIn('id="view-harness"', page)
        self.assertIn('id="view-gateway"', page)
        self.assertIn('href="#gateway">Gateway Bench</a>', page)
        self.assertNotIn('id="view-gateway-probe"', page)
        self.assertNotIn("Gateway Probe", page)
        self.assertIn('if (hash === "gateway-probe")', page)
        self.assertIn('window.history.replaceState(null, "", "#gateway")', page)
        self.assertIn(
            'document.getElementById("view-" + name).hidden = name !== view;',
            page,
        )
        self.assertIn(
            "var target = document.getElementById(hash);", page
        )
        self.assertIn(
            "var host = target && target.closest", page
        )
        self.assertIn('id="view-methodology"', page)
        self.assertNotIn("Gateway Tax", page)
        self.assertIn(
            "Harness Bench denominators are countable cells", page
        )
        self.assertIn(
            "HTTP 429 responses and timeouts remain in\n"
            "    that denominator",
            page,
        )
        self.assertNotIn(
            "Denominators are countable cells. Infrastructure", page
        )

    def test_write_board_emits_the_landing_page_and_data(self):
        info = site.write_board(self.site_dir)
        self.assertEqual(info["html_path"], os.path.join(self.site_dir, "index.html"))
        self.assertTrue(os.path.isfile(info["json_path"]))
        self.assertTrue(os.path.isfile(info["html_path"]))
        with open(info["json_path"], encoding="utf-8") as fh:
            doc = json.load(fh)
        self.assertEqual(doc["schema_version"], site.SCHEMA_VERSION)
        self.assertEqual(info["harness_bundles"], 1)
        self.assertEqual(info["gateway_bundles"], 0)
        self.assertNotIn("gateway_probe_bundles", info)

    def test_write_board_rolls_back_both_outputs_on_replace_failure(self):
        info = site.write_board(self.site_dir)
        with open(info["html_path"], "rb") as fh:
            html_before = fh.read()
        with open(info["json_path"], "rb") as fh:
            json_before = fh.read()
        real_replace = os.replace
        calls = 0

        def fail_second_replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic board publish failure")
            return real_replace(source, destination)

        with mock.patch.object(site.os, "replace", side_effect=fail_second_replace):
            with self.assertRaisesRegex(OSError, "synthetic board publish failure"):
                site.write_board(self.site_dir)
        with open(info["html_path"], "rb") as fh:
            self.assertEqual(fh.read(), html_before)
        with open(info["json_path"], "rb") as fh:
            self.assertEqual(fh.read(), json_before)

    def test_build_is_deterministic(self):
        first = site.write_board(self.site_dir)
        with open(first["json_path"], encoding="utf-8") as fh:
            one = fh.read()
        site.write_board(self.site_dir)
        with open(first["json_path"], encoding="utf-8") as fh:
            two = fh.read()
        self.assertEqual(one, two)


class DesignContractTests(_SiteFixture):
    """The page ships one stylesheet; these are the parts easy to break."""

    def setUp(self):
        super().setUp()
        self.page = site.render_board_html(site.build_board(self.site_dir))

    def test_wordmark_is_the_project_name(self):
        self.assertIn("OpenBench", self.page)
        self.assertNotIn('class="cmd"', self.page)
        self.assertNotIn('content:"$ "', self.page)

    def test_both_theme_scopes_are_defined(self):
        # The media query carries the OS preference; the data-theme scopes carry
        # the viewer's toggle and must be able to win in both directions.
        self.assertIn("@media (prefers-color-scheme:dark)", self.page)
        self.assertIn(':root[data-theme="dark"]', self.page)
        self.assertIn(':root[data-theme="light"]', self.page)
        self.assertIn(':root:where(:not([data-theme="light"]))', self.page)

    def test_interval_marks_are_styled_and_placed(self):
        """A class rename that misses the markup silently unstyles the bars."""
        self.assertIn('class="iv"', self.page)
        self.assertIn('.iv .track{', self.page)
        self.assertIn('class="span"', self.page)

    def test_no_separate_leaderboard_page_is_referenced(self):
        self.assertNotIn("leaderboard.html", self.page)

    def test_reduced_motion_and_focus_are_honoured(self):
        self.assertIn("prefers-reduced-motion", self.page)
        self.assertIn(":focus-visible", self.page)

    def test_wide_content_scrolls_inside_its_own_container(self):
        self.assertIn("overflow-x:auto", self.page)
        self.assertIn('class="scroll"', self.page)

    def test_tinted_text_uses_the_text_safe_pole_step(self):
        """Marks may wear the validated hue; text must clear 4.5:1.

        The validated mark colours sit near 3:1, which is fine for a bar and
        not fine for a number. Tinted values use the darker `*-ink` step.
        """
        self.assertIn("--pole-better-ink:", self.page)
        self.assertIn("--pole-worse-ink:", self.page)
        self.assertIn(".dv .val.better{color:var(--pole-better-ink)}", self.page)
        self.assertIn(".dv .val.worse{color:var(--pole-worse-ink)}", self.page)
        # The raw mark hue must never be assigned to a text colour.
        self.assertNotIn(".val.worse{color:var(--pole-worse)}", self.page)

    def test_plot_width_is_budgeted_per_table(self):
        """Four contrast columns cannot each be as wide as a lone one."""
        self.assertIn("--plot-w:", self.page)

    def test_lede_states_coverage_and_draws_no_conclusion(self):
        """The page reports what is covered; the boards carry the results."""
        title, deck, facts = site._lede(
            site.build_board(self.site_dir), "harness"
        )
        self.assertTrue(any("result-sealed bundles" in f for f in facts))
        self.assertTrue(any("valid result rows" in f for f in facts))
        self.assertTrue(any("matched result rows" in f for f in facts))
        self.assertIn("result-sealed bundle", deck)
        self.assertNotIn("verified bundle", deck)
        # No interpretation of the numbers, and no claimed cause.
        for forbidden in ("because", "due to", "caused by", "proves",
                          "clusters", "spans", "wins", "best", "fastest"):
            self.assertNotIn(forbidden, (title + " " + deck).lower())

    def test_lede_survives_an_empty_site(self):
        with tempfile.TemporaryDirectory() as td:
            doc = site.build_board(td)
        for family in ("harness", "gateway", "general"):
            title, deck, facts = site._lede(doc, family)
            self.assertTrue(title)
            self.assertTrue(deck)
            self.assertTrue(facts)

    def test_family_ledes_isolate_coverage_facts(self):
        doc = site.build_board(self.site_dir)
        doc["gateway"]["bundle_count"] = 1
        doc["gateway"]["bundles"] = [{
            "arms": [{"arm_id": f"route-{i}"} for i in range(5)],
            "model": "Synthetic model",
            "model_match": "rolling_alias",
            "complete_blocks": {"cold": 30, "warm": 30},
            "scheduled_blocks_per_condition": 30,
            "result_count": 300,
            "date": "2026-07-27",
        }]

        harness_title, harness_deck, harness_facts = site._lede(
            doc, "harness"
        )
        self.assertEqual(harness_title, "Coding-agent harness benchmarks")
        self.assertIn("2 harnesses", harness_facts)
        self.assertIn("updated 2026-07-24", harness_facts)
        self.assertNotIn("routes", " ".join(harness_facts))
        self.assertNotIn("requests", " ".join(harness_facts))
        self.assertNotIn("gateway", harness_deck.lower())

        gateway_title, gateway_deck, gateway_facts = site._lede(
            doc, "gateway"
        )
        self.assertEqual(gateway_title, "AI gateway benchmarks")
        self.assertEqual(gateway_facts, [
            "1 published bundle",
            "1 benchmarked model",
            "5 routes per bundle",
            "model-specific denominators below",
            "updated 2026-07-27",
        ])
        self.assertIn("Select a model below", gateway_deck)
        self.assertNotIn("harness", gateway_deck.lower())
        self.assertNotIn("result rows", " ".join(gateway_facts))

        general_title, _, general_facts = site._lede(doc, "general")
        self.assertEqual(general_title, "OpenBench benchmark results")
        self.assertNotIn("requests", " ".join(general_facts))
        self.assertNotIn("complete cold", " ".join(general_facts))

    def test_harness_metadata_names_counts_and_digests_precisely(self):
        page = site.render_board_html(site.build_board(self.site_dir))
        self.assertIn("<b>matched rows </b>8", page)
        self.assertIn("<b>common task/trials </b>4", page)
        self.assertIn("<b>results SHA </b>", page)
        self.assertIn("<b>task-set SHA </b>", page)
        self.assertNotIn("<b>ranked cells </b>", page)


class TableOrderingTests(_SiteFixture):
    """A header that claims a sort must be telling the truth."""

    def _rows(self, page, after):
        import re
        chunk = page[page.index(after):]
        chunk = chunk[:chunk.index("</tbody>")]
        return re.findall(r'class="name">([^<]+)<', chunk)

    def test_declared_sort_is_actually_applied(self):
        columns = [
            {"label": "Name", "cls": "name", "type": "str",
             "cell": lambda r: r["n"], "key": lambda r: r["n"]},
            {"label": "Score", "cell": lambda r: str(r["v"]), "key": lambda r: r["v"]},
        ]
        rows = [{"n": "a", "v": 1}, {"n": "b", "v": 9}, {"n": "c", "v": 5}]
        html = site._render_table(columns, rows, "Score")
        self.assertEqual(self._rows(html, "<tbody>"), ["b", "c", "a"])
        self.assertIn('aria-sort="descending"', html)

    def test_ascending_column_sorts_and_labels_ascending(self):
        columns = [
            {"label": "Name", "cls": "name", "type": "str",
             "cell": lambda r: r["n"], "key": lambda r: r["n"]},
            {"label": "Wall", "dir": "asc",
             "cell": lambda r: str(r["v"]), "key": lambda r: r["v"]},
        ]
        rows = [{"n": "a", "v": 9}, {"n": "b", "v": 1}]
        html = site._render_table(columns, rows, "Wall")
        self.assertEqual(self._rows(html, "<tbody>"), ["b", "a"])
        self.assertIn('aria-sort="ascending"', html)

    def test_rows_without_a_value_sink_and_do_not_crash(self):
        columns = [
            {"label": "Name", "cls": "name", "type": "str",
             "cell": lambda r: r["n"], "key": lambda r: r["n"]},
            {"label": "Score", "cell": lambda r: "-", "key": lambda r: r["v"]},
        ]
        rows = [{"n": "a", "v": None}, {"n": "b", "v": 3}]
        html = site._render_table(columns, rows, "Score")
        self.assertEqual(self._rows(html, "<tbody>"), ["b", "a"])

    def test_no_declared_sort_preserves_caller_order(self):
        """The router arms table leads with a control, not a rank."""
        columns = [{"label": "Name", "cls": "name", "type": "str",
                    "cell": lambda r: r["n"], "key": lambda r: r["n"]}]
        rows = [{"n": "z"}, {"n": "a"}]
        html = site._render_table(columns, rows)
        self.assertEqual(self._rows(html, "<tbody>"), ["z", "a"])
        self.assertNotIn("aria-sort", html)

    def test_tables_do_not_assign_ordinal_rank_or_lead_treatment(self):
        page = site.render_board_html(site.build_board(self.site_dir))
        self.assertNotIn('class="lead"', page)
        self.assertNotIn('class="rank"', page)
        self.assertNotIn('<th scope="col">#</th>', page)


class ContrastPlotTests(unittest.TestCase):
    """The gateway-tax cell is the page's only inferential graphic."""

    def _metric(self, estimate, low, high):
        return {"estimate": estimate, "low": low, "high": high}

    def test_interval_spanning_zero_reads_as_no_effect(self):
        cell = site._delta_cell(
            self._metric(-0.028, -0.111, 0.056), site._fmt_pct, True, 0.2)
        self.assertIn("val null", cell)
        self.assertNotIn("better", cell)
        self.assertNotIn("worse", cell)

    def test_direction_uses_the_diverging_poles(self):
        better = site._delta_cell(
            self._metric(0.08, 0.02, 0.14), site._fmt_pct, True, 0.2)
        self.assertIn("val better", better)
        # Lower latency is better, so a positive delta is the "worse" pole.
        worse = site._delta_cell(
            self._metric(3.45, 1.10, 5.90), lambda v: f"{v:.2f}s", False, 6.0)
        self.assertIn("val worse", worse)

    def test_sign_and_interval_survive_without_colour(self):
        cell = site._delta_cell(
            self._metric(3.45, 1.10, 5.90), lambda v: f"{v:.2f}s", False, 6.0)
        self.assertIn("+3.45s", cell)
        self.assertIn("95% CI +1.10s to +5.90s", cell)

    def test_domain_is_shared_across_a_column(self):
        rows = [
            {"d": self._metric(1.0, 0.5, 2.0)},
            {"d": self._metric(-4.0, -6.0, -2.0)},
        ]
        self.assertEqual(site._delta_domain(rows, "d"), 6.0)

    def test_empty_column_domain_never_divides_by_zero(self):
        self.assertEqual(site._delta_domain([], "d"), 1.0)
        self.assertEqual(
            site._delta_domain([{"d": self._metric(None, None, None)}], "d"), 1.0)


class CliTests(_SiteFixture):
    def test_build_subcommand(self):
        from obench.cli import main as cli_main
        code = cli_main(["site", "build", "--site-dir", self.site_dir,
                         "--no-community-dir"])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.site_dir, "index.html")))
        self.assertTrue(os.path.isfile(os.path.join(self.site_dir, "board.json")))


if __name__ == "__main__":
    unittest.main()
