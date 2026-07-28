#!/usr/bin/env python3
"""Unified static leaderboard site for OpenBench benchmark families.

``obench site build`` scans the GitHub Pages root and emits two artifacts:

* ``board.json`` — one machine-readable document covering **Harness Bench**
  (verified ``results.jsonl`` publish bundles, aggregated by
  :mod:`obench.leaderboard`) and request-level **Gateway Bench** bundles.
* ``index.html`` — the site's landing page, which *is* the leaderboard: family
  tabs, per-board sortable tables, model/harness filters, Wilson and bootstrap
  confidence intervals drawn as bars, and paired route contrasts.

Every table is rendered here, in Python, at build time. The page's script only
enhances what is already in the document — it re-orders rows, hides them, and
switches tabs — so the page is complete with JavaScript switched off and there
is exactly one renderer to keep honest. No server, no build step, and no
third-party assets, the same constraints as every other page this repo
publishes.

Comparability rule, unchanged from ``obench leaderboard``: cells from different
bundles are never blended into one score. Each bundle is its own ranked board.
Gateway Bench request measurements and Harness Bench cells have different units
and denominators and are never merged.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from collections import defaultdict

from . import leaderboard, report_page, stats
from .paths import SOURCE_ROOT

SCHEMA_VERSION = 5
_MAX_GATEWAY_RUN_NOTE_LENGTH = 500

HARNESS_NOTE = (
    "Scores are not comparable across bundles: task sets, trial counts, and "
    "timeout caps differ."
)

GATEWAY_NOTE = (
    "Each board is one Gateway Tax experiment: a direct control arm against "
    "gateway arms under one declared model-match policy. Intervals are "
    "task-bootstrap."
)

GATEWAY_PROBE_NOTE = (
    "Gateway Bench measures request-level transport and serving telemetry. "
    "Cold and warm denominators are separate and are never merged."
)

CROSS_FAMILY_NOTE = (
    "Each board is one result-sealed bundle, shown with its interval and its "
    "provenance. Benchmark families share no denominators."
)


def _public_gateway_title(value):
    """Map legacy public titles onto the sole Gateway Bench product name."""
    value = re.sub(r"\bgateway probe\b", "Gateway Bench", str(value), flags=re.I)
    return re.sub(r"\brequest probe\b", "request benchmark", value, flags=re.I)


def _gateway_run_note(value):
    """Return a short manifest-supplied run disclosure, or ``None``."""
    if not isinstance(value, str):
        return None
    note = value.strip()
    if not note or len(note) > _MAX_GATEWAY_RUN_NOTE_LENGTH:
        return None
    return note


# --------------------------------------------------------------------------
# Harness Bench
# --------------------------------------------------------------------------


def _load_pricing():
    """Repo price sheet, when present. Missing prices simply omit $/solve."""
    for candidate in (
        os.path.join(SOURCE_ROOT, "prices.json"),
        os.path.join(os.getcwd(), "prices.json"),
    ):
        if os.path.isfile(candidate):
            try:
                return stats.load_pricing(candidate)
            except (OSError, ValueError):
                return {}
    return {}


def _arm_rows(bundle_dir):
    """Regroup a bundle's countable rows by ``(harness, model)``.

    Mirrors :func:`obench.leaderboard.aggregate_bundle` exactly so the enriched
    speed/cost columns describe the same cells as the published solve rate.
    """
    results_path = os.path.join(bundle_dir, "results.jsonl")
    rows = stats.load_rows([results_path])
    countable = stats.filter_rows(rows, [])["countable_rows"]
    fields = ("harness", "model")
    matched, diagnostics = stats.matched_rows(countable, fields)
    table_rows = matched if diagnostics is not None else countable
    grouped = defaultdict(list)
    for row in table_rows:
        key = (str(row.get("harness") or "-"), str(row.get("model") or "-"))
        grouped[key].append(row)
    return grouped


def enrich_harness_arms(bundle, bundle_dir, pricing):
    """Attach median wall time and $/solve to an aggregated bundle's arms."""
    try:
        grouped = _arm_rows(bundle_dir)
    except (OSError, ValueError):
        return bundle
    for arm in bundle["arms"]:
        rows = grouped.get((arm["harness"], arm["model"]))
        if not rows:
            arm["median_wall_s"] = None
            arm["cost_per_solve_usd"] = None
            continue
        solved_rows = [r for r in rows if r.get("success")]
        walls = [
            float(r["wall_time_s"]) for r in solved_rows
            if stats.is_nonnegative_number(r.get("wall_time_s"))
        ]
        arm["median_wall_s"] = stats.median(walls) if walls else None
        arm["cost_per_solve_usd"] = report_page._cost_per_solve(
            rows, len(solved_rows), pricing
        )
    return bundle


def build_harness_family(site_dir, community_dir=None):
    """Aggregated Harness Bench boards, enriched with speed and cost."""
    doc = leaderboard.build_leaderboard(site_dir, community_dir=community_dir)
    pricing = _load_pricing()
    for bundle in doc["bundles"]:
        bundle_dir = _bundle_dir_for(site_dir, community_dir, bundle)
        if bundle_dir:
            enrich_harness_arms(bundle, bundle_dir, pricing)
        bundle["family"] = "harness"
    return {
        "note": HARNESS_NOTE,
        "bundle_count": doc["bundle_count"],
        "bundles": doc["bundles"],
        "skipped": doc.get("skipped") or [],
    }


def _bundle_dir_for(site_dir, community_dir, bundle):
    """Resolve an aggregated bundle back to the directory it was read from."""
    results_rel = bundle.get("results_path") or ""
    if results_rel:
        candidate = os.path.join(site_dir, results_rel)
        if os.path.isfile(candidate):
            return os.path.dirname(candidate)
    for parent in ("releases", "community"):
        candidate = os.path.join(site_dir, parent, bundle["id"])
        if os.path.isfile(os.path.join(candidate, "results.jsonl")):
            return candidate
    if community_dir:
        candidate = os.path.join(community_dir, bundle["id"])
        if os.path.isfile(os.path.join(candidate, "results.jsonl")):
            return candidate
    return None


# --------------------------------------------------------------------------
# Gateway Bench
# --------------------------------------------------------------------------


def gateway_verification_error(bundle_dir):
    """Return why a directory is not a verified gateway bundle, else ``None``."""
    provenance_path = os.path.join(bundle_dir, "provenance.json")
    if not os.path.isfile(provenance_path):
        return "no provenance.json (not a gateway evidence bundle)"
    try:
        with open(provenance_path, encoding="utf-8") as fh:
            provenance = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return "invalid provenance.json"
    if not isinstance(provenance, dict):
        return "invalid provenance.json"
    if provenance.get("bundle_kind") != "gateway_bench":
        return "not a gateway_bench bundle"
    from . import gateway_publish
    try:
        gateway_publish.verify_bundle(bundle_dir)
    except Exception as exc:  # noqa: BLE001 - report any verification failure
        return f"bundle verification failed: {exc}"
    return None


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


GATEWAY_METRIC_NAMES = (
    "solve_rate",
    "mean_checker_score",
    "availability",
    "latency_s",
    "ttfb_s",
    "semantic_ttft_s",
    "throughput_tokens_per_s",
    "mean_input_tokens_per_call",
    "mean_output_tokens_per_call",
    "mean_total_tokens_per_call",
    "cache_hit_call_rate",
    "cached_input_fraction",
    "mean_cached_input_tokens_per_call",
    "mean_cache_write_input_tokens_per_call",
)


def _cost_dtos(costs):
    """Preserve every evidenced cost basis without selecting across arms."""
    if not isinstance(costs, dict):
        return {}
    result = {}
    for basis, entry in sorted(costs.items()):
        if not isinstance(entry, dict):
            continue
        coverage = entry.get("basis_coverage") or {}
        if (coverage.get("covered_calls") or 0) <= 0:
            continue
        interval = entry.get("cost_per_solve_interval") or {}
        result[basis] = {
            "basis": basis,
            "attempted_cost_usd": _metric_dto(entry.get("attempted_cost_usd")),
            "cost_per_solve_usd": entry.get("cost_per_solve_usd"),
            "cost_per_solve_low": interval.get("low"),
            "cost_per_solve_high": interval.get("high"),
            "coverage": dict(coverage),
            "currencies": list(entry.get("currencies") or []),
            "effective_at": list(entry.get("effective_at") or []),
        }
    return result


def _common_complete_cost_basis(arms):
    """Return the one comparable cost basis used by the route table.

    Gateway-reported amounts are not available for every serving route. The
    frozen list estimate is the experiment-wide standardized basis, so the
    comparison column appears only when it completely covers every arm.
    """
    arms = list(arms)
    basis = "frozen_list_estimate"
    if not arms:
        return None
    for arm in arms:
        entry = (arm.get("costs") or {}).get(basis) or {}
        coverage = entry.get("basis_coverage") or {}
        if not coverage.get("complete"):
            return None
    return basis


def _metric_dto(metric):
    if not isinstance(metric, dict):
        return {
            "estimate": None,
            "low": None,
            "high": None,
            "aggregation": None,
        }
    interval = metric.get("interval") or {}
    result = {
        "estimate": metric.get("estimate"),
        "low": interval.get("low"),
        "high": interval.get("high"),
        "aggregation": metric.get("aggregation"),
    }
    for name in (
        "task_coverage",
        "cell_coverage",
        "call_coverage",
        "paired_task_coverage",
        "paired_block_coverage",
    ):
        coverage = metric.get(name)
        if isinstance(coverage, dict):
            result[name] = dict(coverage)
    return result


def aggregate_gateway_bundle(bundle_dir, *, site_dir=None, manifest_entry=None):
    """Aggregate one verified gateway bundle, or ``None`` when unusable."""
    from . import gateway_report

    if gateway_verification_error(bundle_dir) is not None:
        return None
    results_path = os.path.join(bundle_dir, "results.jsonl")
    try:
        rows = _read_jsonl(results_path)
        report = gateway_report.aggregate(rows)
    except (OSError, ValueError) as exc:
        del exc
        return None

    experiment = {}
    experiment_path = os.path.join(bundle_dir, "experiment.json")
    if os.path.isfile(experiment_path):
        try:
            with open(experiment_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                experiment = loaded
        except (OSError, json.JSONDecodeError):
            experiment = {}

    arm_spec = {
        a.get("arm_id"): a for a in (experiment.get("arms") or [])
        if isinstance(a, dict) and a.get("arm_id")
    }

    report_arms = report.get("arms") or {}
    common_cost_basis = _common_complete_cost_basis(report_arms.values())
    arms = []
    for arm_id, arm in report_arms.items():
        metrics = arm.get("metrics") or {}
        spec = arm_spec.get(arm_id, {})
        cost_dtos = _cost_dtos(arm.get("costs"))
        projected = {
            "arm_id": arm_id,
            "role": arm.get("role"),
            "route_kind": spec.get("route_kind"),
            "requested_model": spec.get("requested_model"),
            "requested_provider": spec.get("requested_provider"),
            "max_calls": arm.get("max_calls") or {
                "cells": 0,
                "total_cells": arm.get("attempted_cells", 0),
                "ratio": 0.0,
            },
            "route_distribution": dict(arm.get("route_distribution") or {}),
            "costs": cost_dtos,
            "cost": cost_dtos.get(common_cost_basis),
        }
        for name in GATEWAY_METRIC_NAMES:
            projected[name] = _metric_dto(metrics.get(name))
        arms.append(projected)
    # Baseline first, then slowest-to-fastest is meaningless before sorting in
    # the UI; keep a deterministic order: direct arms first, then arm_id.
    arms.sort(key=lambda a: (0 if a["role"] == "direct" else 1, a["arm_id"]))

    contrasts = []
    for arm_id, contrast in (report.get("paired_contrasts") or {}).items():
        metrics = contrast.get("metrics") or {}
        projected = {
            "arm_id": arm_id,
            "direct_arm": contrast.get("direct_arm"),
        }
        for name in GATEWAY_METRIC_NAMES:
            projected[name] = _metric_dto(metrics.get(name))
        projected["attempted_cost_usd"] = _metric_dto(
            metrics.get(f"cost:{common_cost_basis}") if common_cost_basis else None
        )
        contrasts.append(projected)
    contrasts.sort(key=lambda c: c["arm_id"])

    entry = dict(manifest_entry or {})
    bundle_id = entry.get("id") or os.path.basename(os.path.normpath(bundle_dir))
    blocks = report.get("blocks") or {}
    return {
        "family": "gateway",
        "id": bundle_id,
        "kind": entry.get("kind") or "release",
        "title": entry.get("title") or bundle_id,
        "date": entry.get("date") or "",
        "link": entry.get("link"),
        "submitter": entry.get("submitter"),
        "path": entry.get("path"),
        "results_path": (
            leaderboard._rel_under(site_dir, results_path) if site_dir else results_path
        ),
        "track": report.get("track"),
        "model_match": report.get("model_match"),
        "provider_prompt_mode": report.get("provider_prompt_mode"),
        "harness": experiment.get("harness"),
        "experiment_id": experiment.get("experiment_id"),
        "experiment_digest": report.get("experiment_digest"),
        "execution_lane": experiment.get("execution_lane"),
        "budget": dict(experiment.get("budget") or {}),
        "budget_max_calls": (report.get("budget") or {}).get("max_calls"),
        "common_cost_basis": common_cost_basis,
        "blocks_included": blocks.get("included"),
        "blocks_observed": blocks.get("observed"),
        "blocks_excluded": blocks.get("excluded_by_reason") or {},
        "blocks_max_calls_affected": blocks.get("max_calls_affected", 0),
        "tasks_included": (report.get("tasks") or {}).get("included"),
        "arms": arms,
        "contrasts": contrasts,
    }


def build_gateway_family(site_dir, gateway_dirs=None):
    """Scan gateway bundle roots and aggregate every verified bundle."""
    site_dir = os.path.abspath(site_dir)
    roots = list(gateway_dirs or [])
    if not roots:
        default_root = os.path.join(site_dir, "gateway")
        roots = [default_root] if os.path.isdir(default_root) else []

    manifest = {
        e["id"]: e
        for e in leaderboard._load_manifest_list(
            os.path.join(site_dir, "gateway.json"))
        if isinstance(e, dict) and e.get("id")
    }

    bundles = []
    skipped = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            bundle_dir = os.path.join(root, name)
            if not os.path.isdir(bundle_dir):
                continue
            real = os.path.realpath(bundle_dir)
            if real in seen:
                continue
            seen.add(real)
            error = gateway_verification_error(bundle_dir)
            if error:
                skipped.append({"id": name, "kind": "gateway", "reason": error})
                continue
            aggregated = aggregate_gateway_bundle(
                bundle_dir, site_dir=site_dir, manifest_entry=manifest.get(name)
            )
            if aggregated is None:
                skipped.append({
                    "id": name,
                    "kind": "gateway",
                    "reason": "rows did not aggregate into a Gateway Tax report",
                })
                continue
            bundles.append(aggregated)

    bundles.sort(key=lambda b: (-leaderboard._date_key(b.get("date")), b.get("id") or ""))
    skipped.sort(key=lambda s: s.get("id") or "")
    return {
        "note": GATEWAY_NOTE,
        "bundle_count": len(bundles),
        "bundles": bundles,
        "skipped": skipped,
    }


def gateway_probe_verification_error(bundle_dir):
    """Return a fail-closed public Probe verification error, else ``None``."""
    from . import gateway_probe_publish

    try:
        gateway_probe_publish.verify_bundle(bundle_dir)
    except Exception as exc:  # noqa: BLE001 - surface any verifier rejection
        return f"bundle verification failed: {exc}"
    return None


def _output_token_limit_equalities(rows, configured_limit):
    if (
        not isinstance(configured_limit, int)
        or isinstance(configured_limit, bool)
        or configured_limit <= 0
    ):
        return None

    counts = {}
    for row in rows:
        identity = row.get("identity") or {}
        arm_id = (identity.get("arm") or {}).get("id")
        condition = (identity.get("schedule") or {}).get("condition")
        if not arm_id or condition not in {"cold", "warm"}:
            continue
        usage = (row.get("request_metrics") or {}).get("usage") or {}
        output_tokens = usage.get("output_tokens")
        if (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 0
        ):
            continue
        item = counts.setdefault(arm_id, {}).setdefault(
            condition, {"equal": 0, "measured": 0}
        )
        item["measured"] += 1
        if output_tokens == configured_limit:
            item["equal"] += 1

    return {
        "configured_limit": configured_limit,
        "arms": {
            arm_id: {
                condition: dict(values)
                for condition, values in sorted(conditions.items())
            }
            for arm_id, conditions in sorted(counts.items())
        },
    }


def _finish_reason_bucket(stream):
    if not isinstance(stream, dict) or "finish_reason" not in stream:
        return "missing", False
    finish_reason = stream.get("finish_reason")
    if finish_reason == "stop":
        return "natural_stop", True
    if finish_reason == "length":
        return "length", True
    if finish_reason is None or finish_reason == "":
        return "missing", True
    return "other", True


def _gateway_probe_completion_integrity(rows):
    counts = {}
    has_finish_reason_evidence = False

    def increment(item, bucket):
        item["total"] += 1
        item[bucket] += 1

    for row in rows:
        identity = row.get("identity") or {}
        arm_id = (identity.get("arm") or {}).get("id")
        condition = (identity.get("schedule") or {}).get("condition")
        if not arm_id or condition not in {"cold", "warm"}:
            continue

        item = counts.setdefault(arm_id, {}).setdefault(condition, {
            "measured": {
                "natural_stop": 0,
                "length": 0,
                "missing": 0,
                "other": 0,
                "total": 0,
            },
        })
        request_stream = (
            (row.get("request_metrics") or {}).get("stream")
        )
        bucket, has_evidence = _finish_reason_bucket(request_stream)
        has_finish_reason_evidence = (
            has_finish_reason_evidence or has_evidence
        )
        increment(item["measured"], bucket)

        if condition == "warm":
            primer = item.setdefault("warm_primer", {
                "natural_stop": 0,
                "length": 0,
                "missing": 0,
                "other": 0,
                "total": 0,
            })
            primer_stream = (
                (row.get("reuse_evidence") or {}).get("stream")
            )
            bucket, has_evidence = _finish_reason_bucket(primer_stream)
            has_finish_reason_evidence = (
                has_finish_reason_evidence or has_evidence
            )
            increment(primer, bucket)

    if not has_finish_reason_evidence:
        return None
    return {
        "arms": {
            arm_id: {
                condition: conditions[condition]
                for condition in ("cold", "warm")
                if condition in conditions
            }
            for arm_id, conditions in sorted(counts.items())
        },
    }


def _retry_timing_summary(values):
    values = [
        value for value in values
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
        )
    ]
    return {
        "count": len(values),
        "median": stats.median(values) if values else None,
        "max": max(values) if values else None,
    }


def _gateway_probe_retry_summary(rows, expected_output_limit=None):
    """Derive retry diagnostics from verified public result rows."""
    if not rows or not all(
        isinstance(row.get("retry_evidence"), dict) for row in rows
    ):
        return None

    latest_by_coordinate = {}
    for row in rows:
        identity = row.get("identity") or {}
        case_id = (identity.get("case") or {}).get("id")
        schedule = identity.get("schedule") or {}
        condition = schedule.get("condition")
        repetition = schedule.get("repetition")
        block_attempt = schedule.get("block_attempt")
        if (
            not case_id
            or condition not in {"cold", "warm"}
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not isinstance(block_attempt, int)
            or isinstance(block_attempt, bool)
            or block_attempt < 0
        ):
            return None
        coordinate = (case_id, condition, repetition)
        latest = latest_by_coordinate.get(coordinate)
        if latest is None or block_attempt > latest[0]:
            latest_by_coordinate[coordinate] = (block_attempt, [row])
        elif block_attempt == latest[0]:
            latest[1].append(row)

    rows = [
        row
        for _attempt, selected in latest_by_coordinate.values()
        for row in selected
    ]
    groups = defaultdict(list)
    for row in rows:
        identity = row.get("identity") or {}
        arm_id = (identity.get("arm") or {}).get("id")
        condition = (identity.get("schedule") or {}).get("condition")
        retry = row["retry_evidence"]
        attempts = retry.get("attempts")
        attempt_count = retry.get("attempt_count")
        max_attempts = retry.get("max_total_attempts")
        output_limit = retry.get("max_output_tokens")
        if (
            not arm_id
            or condition not in {"cold", "warm"}
            or not isinstance(attempts, list)
            or not attempts
            or not isinstance(attempt_count, int)
            or isinstance(attempt_count, bool)
            or attempt_count != len(attempts)
            or not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < attempt_count
            or not isinstance(output_limit, int)
            or isinstance(output_limit, bool)
            or output_limit <= 0
            or (
                expected_output_limit is not None
                and output_limit != expected_output_limit
            )
            or not isinstance(retry.get("first_attempt_outcome"), dict)
            or not isinstance(retry.get("eventual_outcome"), dict)
            or not isinstance(retry.get("recovery_timing"), dict)
            or not isinstance(retry.get("recovered"), bool)
        ):
            return None
        groups[(arm_id, condition)].append(row)

    def summarize(selected):
        attempt_distribution = defaultdict(int)
        retry_after_values = []
        wait_values = []
        recovery_completion_values = []
        final_attempt_offset_values = []
        limits = defaultdict(lambda: {"equal": 0, "measured": 0})
        attempts_total = 0
        first_successes = 0
        eventual_successes = 0
        retried = 0
        rescued = 0
        exhausted = 0

        for row in selected:
            retry = row["retry_evidence"]
            attempt_count = retry["attempt_count"]
            eventual_success = (
                retry["eventual_outcome"].get("success") is True
            )
            attempts_total += attempt_count
            attempt_distribution[attempt_count] += 1
            first_successes += (
                retry["first_attempt_outcome"].get("success") is True
            )
            eventual_successes += eventual_success
            retried += attempt_count > 1
            rescued += retry["recovered"] is True
            exhausted += (
                attempt_count == retry["max_total_attempts"]
                and not eventual_success
            )

            if attempt_count > 1:
                timing = retry["recovery_timing"]
                recovery_completion_values.append(
                    timing.get("initial_request_to_completion_s")
                )
                final_attempt_offset_values.append(
                    timing.get("final_attempt_request_start_offset_s")
                )

            for attempt in retry["attempts"]:
                retry_attempt = (
                    attempt.get("retry")
                    if isinstance(attempt, dict) else None
                ) or {}
                retry_after_values.append(retry_attempt.get("retry_after_s"))
                wait_values.append(retry_attempt.get("wait_actual_s"))

            limit = retry["max_output_tokens"]
            usage = (row.get("request_metrics") or {}).get("usage") or {}
            output_tokens = usage.get("output_tokens")
            if (
                isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and output_tokens >= 0
            ):
                limits[limit]["measured"] += 1
                if output_tokens == limit:
                    limits[limit]["equal"] += 1

        return {
            "logical_cells": len(selected),
            "attempts": attempts_total,
            "first_attempt_successes": first_successes,
            "eventual_successes": eventual_successes,
            "retried": retried,
            "rescued": rescued,
            "exhausted": exhausted,
            "attempt_distribution": {
                str(count): frequency
                for count, frequency in sorted(attempt_distribution.items())
            },
            "retry_after_s": _retry_timing_summary(retry_after_values),
            "wait_actual_s": _retry_timing_summary(wait_values),
            "recovery_completion_s": _retry_timing_summary(
                recovery_completion_values
            ),
            "final_attempt_start_offset_s": _retry_timing_summary(
                final_attempt_offset_values
            ),
            "output_limits": {
                str(limit): dict(counts)
                for limit, counts in sorted(limits.items())
            },
        }

    return {
        "overall": summarize(rows),
        "groups": [
            {
                "arm_id": arm_id,
                "condition": condition,
                **summarize(selected),
            }
            for (arm_id, condition), selected in sorted(groups.items())
        ],
    }


def aggregate_gateway_probe_bundle(
    bundle_dir, *, site_dir=None, manifest_entry=None
):
    """Project one verified public Gateway Probe report-v4 bundle for the site."""
    from . import gateway_probe_publish

    try:
        manifest = gateway_probe_publish.verify_bundle(bundle_dir)
        with open(
            os.path.join(bundle_dir, "experiment.json"),
            encoding="ascii",
        ) as fh:
            experiment = json.load(fh)
        with open(os.path.join(bundle_dir, "report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
        rows = _read_jsonl(os.path.join(bundle_dir, "results.jsonl"))
    except Exception:  # noqa: BLE001 - any drift makes the board unusable
        return None

    if (
        report.get("schema_version") != 4
        or report.get("benchmark") != "gateway_probe"
        or not isinstance(report.get("arms"), dict)
        or not isinstance(report.get("paired_contrasts"), dict)
    ):
        return None

    arms = []
    for arm_id, arm in sorted(
        report["arms"].items(),
        key=lambda item: (
            0 if item[1].get("baseline") else 1,
            item[0],
        ),
    ):
        arms.append({
            "arm_id": arm_id,
            "role": arm.get("role"),
            "baseline": arm.get("baseline") is True,
            "conditions": dict(arm.get("conditions") or {}),
        })

    contrasts = [
        {
            "arm_id": arm_id,
            "direct_arm": report.get("baseline_arm_id"),
            "conditions": dict(conditions or {}),
        }
        for arm_id, conditions in sorted(report["paired_contrasts"].items())
    ]

    entry = dict(manifest_entry or {})
    configured_output_limit = experiment["budget"]["max_output_tokens"]
    retry_summary = _gateway_probe_retry_summary(
        rows,
        expected_output_limit=configured_output_limit,
    )
    output_limit_equalities = _output_token_limit_equalities(
        rows,
        configured_output_limit,
    )
    completion_integrity = _gateway_probe_completion_integrity(rows)
    bundle_id = entry.get("id") or os.path.basename(os.path.normpath(bundle_dir))
    results_path = os.path.join(bundle_dir, "results.jsonl")
    verification = manifest.get("verification") or {}
    first_row = rows[0]
    track = (
        ((first_row.get("identity") or {}).get("benchmark") or {}).get("track")
    )
    bundle = {
        "family": "gateway_probe",
        "id": bundle_id,
        "kind": entry.get("kind") or "release",
        "title": _public_gateway_title(entry.get("title") or bundle_id),
        "model": entry.get("model") or entry.get("title") or bundle_id,
        "date": entry.get("date") or "",
        "path": entry.get("path"),
        "link": entry.get("link"),
        "submitter": entry.get("submitter"),
        "results_path": (
            leaderboard._rel_under(site_dir, results_path)
            if site_dir else results_path
        ),
        "track": track,
        "model_match": first_row.get("model_match"),
        "experiment_id": report.get("experiment_id"),
        "experiment_digest": report.get("experiment_digest"),
        "schedule_digest": report.get("schedule_digest"),
        "price_digest": report.get("price_digest"),
        "verified_with_commit": verification.get("verified_with_commit"),
        "result_count": manifest.get("result_count"),
        "complete_blocks": dict(report.get("complete_blocks") or {}),
        "scheduled_blocks_per_condition": report.get(
            "scheduled_blocks_per_condition"
        ),
        "baseline_arm_id": report.get("baseline_arm_id"),
        "completion_integrity": completion_integrity,
        "retry_summary": retry_summary,
        "output_token_limit_equalities": output_limit_equalities,
        "arms": arms,
        "contrasts": contrasts,
    }
    run_note = _gateway_run_note(entry.get("run_note"))
    if run_note:
        bundle["run_note"] = run_note
    return bundle


def build_gateway_probe_family(site_dir, gateway_probe_dirs=None):
    """Scan and fail-closed verify public Gateway Probe bundles."""
    site_dir = os.path.abspath(site_dir)
    roots = list(gateway_probe_dirs or [])
    if not roots:
        default_root = os.path.join(site_dir, "gateway-probe")
        roots = [default_root] if os.path.isdir(default_root) else []

    manifest = {
        entry["id"]: entry
        for entry in leaderboard._load_manifest_list(
            os.path.join(site_dir, "gateway-probe.json")
        )
        if isinstance(entry, dict) and entry.get("id")
    }

    bundles = []
    skipped = []
    seen = set()
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            bundle_dir = os.path.join(root, name)
            if not os.path.isdir(bundle_dir):
                continue
            real = os.path.realpath(bundle_dir)
            if real in seen:
                continue
            seen.add(real)
            error = gateway_probe_verification_error(bundle_dir)
            if error:
                skipped.append({
                    "id": name,
                    "kind": "gateway_probe",
                    "reason": error,
                })
                continue
            aggregated = aggregate_gateway_probe_bundle(
                bundle_dir,
                site_dir=site_dir,
                manifest_entry=manifest.get(name),
            )
            if aggregated is None:
                skipped.append({
                    "id": name,
                    "kind": "gateway_probe",
                    "reason": "verified report did not match Gateway Bench schema v4",
                })
                continue
            bundles.append(aggregated)

    bundles.sort(
        key=lambda bundle: (
            -leaderboard._date_key(bundle.get("date")),
            bundle.get("id") or "",
        )
    )
    skipped.sort(key=lambda item: item.get("id") or "")
    return {
        "note": GATEWAY_PROBE_NOTE,
        "bundle_count": len(bundles),
        "bundles": bundles,
        "skipped": skipped,
    }


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------


def build_board(
    site_dir, community_dir=None, gateway_dirs=None, gateway_probe_dirs=None
):
    """Build the combined benchmark-family board document."""
    site_dir = os.path.abspath(site_dir)
    harness = build_harness_family(site_dir, community_dir=community_dir)
    # Legacy coding-agent gateway workloads are intentionally excluded from
    # the public board. Keep the argument as an inert compatibility input.
    del gateway_dirs
    gateway_probe = build_gateway_probe_family(
        site_dir, gateway_probe_dirs=gateway_probe_dirs
    )
    releases = leaderboard._load_manifest_list(os.path.join(site_dir, "releases.json"))
    community = leaderboard._load_manifest_list(os.path.join(site_dir, "community.json"))
    packs = leaderboard._load_manifest_list(os.path.join(site_dir, "packs.json"))
    site_metadata = {}
    metadata_path = os.path.join(site_dir, "site-meta.json")
    try:
        with open(metadata_path, encoding="utf-8") as fh:
            candidate = json.load(fh)
        if isinstance(candidate, dict):
            site_metadata = {
                key: value
                for key, value in candidate.items()
                if key in {"canonical_url", "social_image_url"}
                and isinstance(value, str)
            }
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return {
        "generated_by": "obench site",
        "schema_version": SCHEMA_VERSION,
        "site_metadata": site_metadata,
        "cross_family_note": CROSS_FAMILY_NOTE,
        "harness": harness,
        "gateway": gateway_probe,
        "releases": [e for e in releases if isinstance(e, dict)],
        "community": [e for e in community if isinstance(e, dict)],
        "packs": [e for e in packs if isinstance(e, dict)],
    }


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

_CSS = """
/* ---------------------------------------------------------------------------
   OpenBench leaderboards.

   Treated as a published measurement rather than a dashboard.

   Type      three roles. A serif display face carries headlines and board
             titles; a sans carries prose and small labels; a mono carries
             every identifier the benchmark names and every value it measures,
             because those are readings rather than writing.
   Colour    the page is ink on paper. Colour is reserved for data: the
             interval marks, and the validated blue/orange diverging pair on
             signed contrasts. Chrome never spends it. Marks carry the
             validated hue; any text tinted by a pole uses the darker
             `*-ink` step, so no reading depends on a sub-4.5:1 colour.
   Structure rules, not boxes. Boards are records separated by hairlines, so
             the data sits on the page instead of inside a card.
--------------------------------------------------------------------------- */
:root{
  color-scheme:light;
  --font-display:"Iowan Old Style","Palatino Linotype",Palatino,Charter,
    "Bitstream Charter",Georgia,"Liberation Serif",serif;
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    "Helvetica Neue",Arial,sans-serif;
  --font-mono:ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",
    "Cascadia Mono",Menlo,Consolas,"Liberation Mono",monospace;

  --paper:#ffffff;
  --ground:#f2f2f0;
  --ink:#0f1113;
  --ink-2:#54595f;
  --ink-3:#6b7280;
  --rule:#dfe0dd;
  --rule-strong:#b9bcb8;
  --wash:#f7f7f5;

  --data:#2a78d6;
  --data-rgb:42,120,214;
  --pole-better:#2a78d6;
  --pole-better-rgb:42,120,214;
  --pole-worse:#eb6834;
  --pole-worse-rgb:235,104,52;
  --pole-better-ink:#1f66bd;
  --pole-worse-ink:#ad4218;
  --pole-null:#6b7280;
  --pole-null-rgb:107,114,128;

  --warn-ink:#8a5300;
  --warn-bg:#fbf1dc;
  --warn-rule:#e0c58a;

  --track:rgba(15,17,19,.08);
  --grid:rgba(15,17,19,.14);
  --plot-w:140px;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --paper:#101215;
    --ground:#08090b;
    --ink:#eceef1;
    --ink-2:#a2a9b2;
    --ink-3:#8b939d;
    --rule:#22262b;
    --rule-strong:#39404a;
    --wash:#161a1f;

    --data:#3987e5;
    --data-rgb:57,135,229;
    --pole-better:#3987e5;
    --pole-better-rgb:57,135,229;
    --pole-worse:#d95926;
    --pole-worse-rgb:217,89,38;
    --pole-better-ink:#3987e5;
    --pole-worse-ink:#e06a3c;
    --pole-null:#8b939d;
    --pole-null-rgb:139,147,157;

    --warn-ink:#f0b545;
    --warn-bg:#241d10;
    --warn-rule:#4b3c17;

    --track:rgba(236,238,241,.10);
    --grid:rgba(236,238,241,.16);
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --paper:#101215;
  --ground:#08090b;
  --ink:#eceef1;
  --ink-2:#a2a9b2;
  --ink-3:#8b939d;
  --rule:#22262b;
  --rule-strong:#39404a;
  --wash:#161a1f;

  --data:#3987e5;
  --data-rgb:57,135,229;
  --pole-better:#3987e5;
  --pole-better-rgb:57,135,229;
  --pole-worse:#d95926;
  --pole-worse-rgb:217,89,38;
  --pole-better-ink:#3987e5;
  --pole-worse-ink:#e06a3c;
  --pole-null:#8b939d;
  --pole-null-rgb:139,147,157;

  --warn-ink:#f0b545;
  --warn-bg:#241d10;
  --warn-rule:#4b3c17;

  --track:rgba(236,238,241,.10);
  --grid:rgba(236,238,241,.16);
}
:root[data-theme="light"]{
  color-scheme:light;
  --paper:#ffffff;
  --ground:#f2f2f0;
  --ink:#0f1113;
  --ink-2:#54595f;
  --ink-3:#6b7280;
  --rule:#dfe0dd;
  --rule-strong:#b9bcb8;
  --wash:#f7f7f5;

  --data:#2a78d6;
  --data-rgb:42,120,214;
  --pole-better:#2a78d6;
  --pole-better-rgb:42,120,214;
  --pole-worse:#eb6834;
  --pole-worse-rgb:235,104,52;
  --pole-better-ink:#1f66bd;
  --pole-worse-ink:#ad4218;
  --pole-null:#6b7280;
  --pole-null-rgb:107,114,128;

  --warn-ink:#8a5300;
  --warn-bg:#fbf1dc;
  --warn-rule:#e0c58a;

  --track:rgba(15,17,19,.08);
  --grid:rgba(15,17,19,.14);
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 var(--font-sans);-webkit-font-smoothing:antialiased;
  font-variant-numeric:tabular-nums}
[hidden]{display:none !important}
a{color:inherit;text-decoration-color:var(--rule-strong);text-underline-offset:3px}
a:hover{text-decoration-color:currentColor}
:focus-visible{outline:2px solid var(--data);outline-offset:3px;border-radius:2px}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important;transition-duration:.01ms !important}
}

.wrap{max-width:1180px;margin:0 auto;padding:0 32px}

/* --- masthead ----------------------------------------------------------- */
header.top{border-bottom:1px solid var(--ink);background:var(--paper);
  position:sticky;top:0;z-index:30}
.top .wrap{display:flex;align-items:center;gap:26px;min-height:56px;flex-wrap:wrap}
.brand{display:flex;align-items:baseline;gap:9px;white-space:nowrap;
  margin-right:auto}
.brand .cmd{font:600 13px/1 var(--font-mono);letter-spacing:.16em;
  text-transform:uppercase}
.brand .what{font:italic 15px/1 var(--font-display);color:var(--ink-2)}
nav.tabs{display:flex;gap:24px;flex-wrap:wrap}
nav.tabs a{font:600 13px/1 var(--font-sans);letter-spacing:.03em;
  text-decoration:none;color:var(--ink-3);padding:19px 0;
  border-bottom:2px solid transparent;margin-bottom:-1px}
nav.tabs a:hover{color:var(--ink)}
nav.tabs a[aria-current="page"]{color:var(--ink);border-bottom-color:var(--ink)}
button.theme{background:none;border:0;color:var(--ink-3);cursor:pointer;
  padding:6px;display:inline-flex;align-items:center}
button.theme:hover{color:var(--ink)}
button.theme svg{width:15px;height:15px}

/* --- the lede ----------------------------------------------------------- */
.lede{padding:56px 0 36px;border-bottom:1px solid var(--rule)}
.lede h1{margin:0;max-width:26ch;font:400 clamp(30px,4vw,46px)/1.08
  var(--font-display);letter-spacing:-.02em;text-wrap:balance}
.lede .deck{margin:22px 0 0;max-width:56ch;font-size:18.5px;line-height:1.55;
  color:var(--ink-2)}
.lede .dateline{margin-top:30px;padding-top:14px;border-top:1px solid var(--rule);
  font:11.5px/1.7 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;
  color:var(--ink-3);display:flex;flex-wrap:wrap;gap:0 22px}

/* --- controls ----------------------------------------------------------- */
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;
  padding:20px 0;border-bottom:1px solid var(--rule)}
.controls input[type=search],.controls select{background:transparent;
  color:var(--ink);border:0;border-bottom:1px solid var(--rule-strong);
  padding:6px 2px;font:13px var(--font-mono);border-radius:0}
.controls input[type=search]{min-width:230px;flex:1}
.controls select{font-family:var(--font-sans);font-size:13.5px;cursor:pointer}
.controls label{display:flex;align-items:center;gap:8px;color:var(--ink-2);
  font-size:13.5px}

/* --- asides ------------------------------------------------------------- */
.note{margin:22px 0 0;padding:0;color:var(--ink-2);font-size:15px;max-width:78ch}
.note strong{color:var(--ink);font-weight:600}
.gateway-model-tabs{display:flex;gap:0;margin-top:26px;border-bottom:1px solid var(--rule);
  overflow-x:auto}
.gateway-model-tabs button{appearance:none;background:none;border:0;
  border-bottom:2px solid transparent;color:var(--ink-3);cursor:pointer;
  padding:12px 18px 10px;font:600 14px/1.2 var(--font-sans);white-space:nowrap}
.gateway-model-tabs button:first-child{padding-left:0}
.gateway-model-tabs button[aria-selected="true"]{border-bottom-color:var(--ink);
  color:var(--ink)}
.gateway-model-tabs .tab-depth{display:block;margin-top:5px;
  font:10.5px/1.25 var(--font-mono);font-weight:400;color:var(--ink-3)}
.evidence-depth{border-left:3px solid var(--rule-strong);padding-left:12px}
.evidence-depth.is-spike{border-left-color:var(--warn-rule);color:var(--warn-ink)}

/* --- boards, as records rather than cards ------------------------------- */
.board{padding:52px 0 8px;border-bottom:1px solid var(--rule)}
.board:last-child{border-bottom:0}
.board .head{padding:0 0 18px}
.board .head:last-child{padding:16px 0 0;border-top:1px solid var(--rule)}
.scroll+.head{padding-top:40px}
.board h2{margin:0;font:400 27px/1.2 var(--font-display);letter-spacing:-.012em;
  max-width:70ch;text-wrap:balance;hyphens:none}
.board h2 a{text-decoration:none}
.board h2 a:hover{text-decoration:underline}
.board .head p{margin:10px 0 0;color:var(--ink-2);font-size:14.5px;max-width:74ch}
.meta{margin-top:11px;font:11.5px/1.9 var(--font-mono);letter-spacing:.05em;
  color:var(--ink-3);display:flex;flex-wrap:wrap;gap:0 20px;
  text-transform:uppercase}
.meta b{font-weight:400;color:var(--ink-3);opacity:.65}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
.chip{border:1px solid var(--rule-strong);color:var(--ink-2);
  padding:2px 8px;font:500 11px/1.6 var(--font-mono);white-space:nowrap;
  letter-spacing:.02em}
.chip.warn{background:var(--warn-bg);border-color:var(--warn-rule);
  color:var(--warn-ink)}
.chip.role-direct{border-color:var(--ink);color:var(--ink)}
td .chips{flex-wrap:nowrap;justify-content:flex-end;margin-top:0}

details.caveats{margin-top:14px;font-size:14.5px}
details.caveats summary{cursor:pointer;color:var(--warn-ink);font-weight:600;
  list-style:none;display:flex;align-items:center;gap:7px;font-size:13.5px}
details.caveats summary::-webkit-details-marker{display:none}
details.caveats summary::before{content:"+";font:600 15px var(--font-mono)}
details.caveats[open] summary::before{content:"\\2212"}
details.caveats ul{margin:12px 0 0;padding-left:19px;color:var(--ink-2)}
details.caveats li{margin:8px 0;max-width:80ch}

/* --- tables ------------------------------------------------------------- */
.scroll{overflow-x:auto;margin:0 -4px;padding:0 4px}
table{border-collapse:collapse;width:100%}
th,td{padding:13px 11px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;padding-left:0}
th:last-child,td:last-child{padding-right:0}
thead th{color:var(--ink-3);font:600 10.5px/1.4 var(--font-sans);
  text-transform:uppercase;letter-spacing:.1em;vertical-align:bottom;
  padding-bottom:11px;border-bottom:1px solid var(--ink)}
thead th.sortable{cursor:pointer;user-select:none}
thead th.sortable:hover{color:var(--ink)}
thead th .arrow{opacity:.35;margin-left:5px;font-size:9px;vertical-align:1px}
thead th[aria-sort]{color:var(--ink)}
thead th[aria-sort] .arrow{opacity:1}
tbody td{font:14px/1.4 var(--font-mono);color:var(--ink-2);
  border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--wash)}
td.name{color:var(--ink);font-weight:600;letter-spacing:-.01em}
/* Identity stays put while the measures scroll. */
thead th:first-child,tbody td:first-child{position:sticky;left:0;z-index:1;
  background:var(--paper)}
tbody tr:hover td:first-child{background:var(--wash)}

/* --- interval plot: one shared 0-100% scale for the whole column -------- */
.scroll[data-dense="1"] .iv .range{display:none}
.scroll[data-dense="1"] th,.scroll[data-dense="1"] td{padding-left:9px;
  padding-right:9px}
.axis{display:flex;justify-content:space-between;width:var(--plot-w);
  margin:7px 0 0 auto;font:400 9.5px/1 var(--font-mono);color:var(--ink-3);
  letter-spacing:.02em;text-transform:none}
.iv{display:flex;align-items:center;gap:11px;justify-content:flex-end}
.iv .val{flex:0 0 auto;min-width:56px;text-align:right;color:var(--ink);
  font-weight:600;font-size:14.5px}
.iv .track{flex:0 0 var(--plot-w);position:relative;width:var(--plot-w);
  height:5px;background:var(--track);border-radius:1px;
  background-image:linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:25% 100%}
.iv .span{position:absolute;top:0;height:5px;border-radius:1px;
  background:rgba(var(--data-rgb),.40)}
.iv .dot{position:absolute;top:-3px;width:3px;height:11px;border-radius:1px;
  background:var(--data)}
.iv .range{flex:0 0 auto;min-width:84px;text-align:right;color:var(--ink-3);
  font-size:11.5px}

/* --- signed contrast: diverging about a zero line ----------------------- */
.dv{display:flex;align-items:center;gap:11px;justify-content:flex-end}
.dv .val{flex:0 0 auto;min-width:62px;text-align:right;font-weight:600;
  font-size:14.5px}
.dv .val.better{color:var(--pole-better-ink)}
.dv .val.worse{color:var(--pole-worse-ink)}
.dv .val.null{color:var(--ink-2)}
.dv .track{flex:0 0 var(--plot-w);position:relative;width:var(--plot-w);
  height:5px;background:var(--track);border-radius:1px}
.dv .zero{position:absolute;left:50%;top:-4px;width:1px;height:13px;
  background:var(--rule-strong)}
.dv .span{position:absolute;top:0;height:5px;border-radius:1px}
.dv .span.better{background:rgba(var(--pole-better-rgb),.40)}
.dv .span.worse{background:rgba(var(--pole-worse-rgb),.40)}
.dv .span.null{background:rgba(var(--pole-null-rgb),.34)}
.dv .dot{position:absolute;top:-3px;width:3px;height:11px;border-radius:1px}
.dv .dot.better{background:var(--pole-better)}
.dv .dot.worse{background:var(--pole-worse)}
.dv .dot.null{background:var(--pole-null)}
.legend{display:flex;gap:22px;flex-wrap:wrap;margin-top:14px;
  font-size:13px;color:var(--ink-2)}
.legend span{display:flex;align-items:center;gap:8px}
.legend i{width:16px;height:5px;border-radius:1px;display:inline-block}
.legend i.better{background:var(--pole-better)}
.legend i.worse{background:var(--pole-worse)}
.legend i.null{background:var(--pole-null)}

/* --- compact route leaderboard ----------------------------------------- */
.route-leaderboard{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:10px;margin:2px 0 40px}
.route-rank{display:grid;grid-template-columns:28px minmax(0,1fr) auto;
  align-items:center;gap:12px;min-height:74px;padding:13px 15px;
  border:1px solid var(--rule);border-radius:6px;background:var(--paper)}
.route-rank.baseline{border-style:dashed}
.route-position{font:600 12px/1 var(--font-mono);color:var(--ink-3);
  text-align:center}
.route-identity{display:flex;align-items:center;gap:11px;min-width:0}
.route-logo{display:inline-flex;align-items:center;justify-content:center;
  width:30px;height:30px;flex:0 0 30px;color:var(--ink)}
.route-logo svg{display:block;width:100%;height:100%}
.route-copy{min-width:0}
.route-name{font:600 14px/1.25 var(--font-sans);color:var(--ink);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.route-detail{margin-top:5px;font:10.5px/1.35 var(--font-mono);
  color:var(--ink-3);white-space:nowrap}
.route-measure{text-align:right;white-space:nowrap}
.route-measure strong{display:block;font:600 18px/1.15 var(--font-mono);
  color:var(--ink)}
.route-measure span{display:block;margin-top:4px;font:10px/1.2 var(--font-sans);
  text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3)}
.provider-cell{display:inline-flex;align-items:center;gap:8px;white-space:nowrap}
.provider-cell .route-logo{width:20px;height:20px;flex-basis:20px}
.contact-actions{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}
.contact-actions a{display:inline-flex;align-items:center;min-height:38px;
  padding:0 14px;border:1px solid var(--rule-strong);border-radius:4px;
  color:var(--ink);font:600 13px/1 var(--font-sans);text-decoration:none}
.contact-actions a:hover{border-color:var(--ink);background:var(--wash)}

/* --- lists and prose ---------------------------------------------------- */
.empty{padding:4px 0 32px;color:var(--ink-2);max-width:62ch}
.empty p{margin:0 0 10px}
code{font-family:var(--font-mono);font-size:.86em}
ul.records{list-style:none;margin:0;padding:0}
ul.records li{padding:16px 0;border-bottom:1px solid var(--rule)}
ul.records li:last-child{border-bottom:0}
ul.records a,ul.records strong{font:500 17px/1.35 var(--font-display);
  text-decoration:none}
ul.records a:hover{text-decoration:underline}
ul.records .sub{color:var(--ink-3);font:11.5px/1.8 var(--font-mono);
  letter-spacing:.04em;margin-top:3px}
.prose{padding:52px 0;max-width:70ch}
.prose h2{margin:0 0 10px;font:400 27px/1.2 var(--font-display);
  letter-spacing:-.012em}
.prose h3{margin:34px 0 8px;font:600 11px/1.4 var(--font-sans);
  text-transform:uppercase;letter-spacing:.1em;color:var(--ink-3)}
.prose p,.prose li{color:var(--ink-2);font-size:15.5px}
.prose li{margin:9px 0}
.prose strong{color:var(--ink)}
footer{color:var(--ink-3);font-size:12.5px;padding:28px 0 60px;
  border-top:1px solid var(--rule)}

@media(max-width:1080px){:root{--plot-w:112px}}
@media(max-width:860px){
  :root{--plot-w:96px}
  .iv .range{display:none}
  .wrap{padding:0 20px}
  .lede{padding:44px 0 30px}
  nav.tabs{gap:18px}
}
@media(max-width:680px){
  :root{--plot-w:70px}
  .top .wrap{min-height:0;padding-top:10px;padding-bottom:0;gap:12px}
  nav.tabs{order:3;width:100%;gap:20px;overflow-x:auto}
  nav.tabs a{padding:10px 0}
  .brand{margin-right:0}
  .lede h1{font-size:31px}
  .lede .deck{font-size:16.5px}
  .board{padding:34px 0 8px}
  .board h2{font-size:22px}
  .route-leaderboard{grid-template-columns:1fr}
  .route-rank{grid-template-columns:24px minmax(0,1fr) auto;
    align-items:start;padding-left:12px;padding-right:12px}
  .route-detail{white-space:normal}
  th,td{padding:11px 10px}
}
"""

_JS = r"""
(function () {
  "use strict";
  // Progressive enhancement only. Every table is already in the document;
  // this re-orders rows, hides them, and switches tabs. Nothing is built here.
  var root = document.documentElement;

  try {
    var saved = localStorage.getItem("obench-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) { /* storage disabled */ }

  document.getElementById("theme").addEventListener("click", function () {
    var now = root.getAttribute("data-theme")
      || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = now === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("obench-theme", next); } catch (e) { /* ignore */ }
  });

  // --- sorting ------------------------------------------------------------
  function sortBy(table, th) {
    var col = th.getAttribute("data-col");
    var numeric = th.getAttribute("data-type") !== "str";
    var was = th.getAttribute("aria-sort");
    var dir = was === "descending" ? "ascending" : "descending";
    var sign = dir === "ascending" ? 1 : -1;

    table.querySelectorAll("thead th").forEach(function (other) {
      other.removeAttribute("aria-sort");
      var arrow = other.querySelector(".arrow");
      if (arrow) arrow.textContent = "↕";
    });
    th.setAttribute("aria-sort", dir);
    var arrow = th.querySelector(".arrow");
    if (arrow) arrow.textContent = dir === "ascending" ? "↑" : "↓";

    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var x = a.getAttribute("data-s" + col);
      var y = b.getAttribute("data-s" + col);
      // Rows with no value for this measure always sink, either direction.
      if (x === "" && y === "") return 0;
      if (x === "") return 1;
      if (y === "") return -1;
      if (numeric) return (parseFloat(x) - parseFloat(y)) * sign;
      return x.localeCompare(y) * sign;
    });
    rows.forEach(function (tr) { tbody.appendChild(tr); });
  }

  document.querySelectorAll("table").forEach(function (table) {
    table.querySelectorAll("thead th.sortable").forEach(function (th) {
      th.addEventListener("click", function () { sortBy(table, th); });
      th.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); sortBy(table, th); }
      });
    });
  });

  // --- filtering ----------------------------------------------------------
  var controls = document.getElementById("controls");
  if (controls) {
    var q = document.getElementById("q");
    var fModel = document.getElementById("f-model");
    var fHarness = document.getElementById("f-harness");
    var fCaveats = document.getElementById("f-caveats");
    var noMatches = document.getElementById("no-matches");
    var boards = document.querySelectorAll("#view-harness .board[data-models]");

    function applyFilters() {
      var text = (q.value || "").trim().toLowerCase();
      var model = fModel ? fModel.value : "";
      var harness = fHarness ? fHarness.value : "";
      var hideCaveats = fCaveats && fCaveats.checked;
      var shown = 0;

      boards.forEach(function (board) {
        if (hideCaveats && board.getAttribute("data-caveats") === "1") {
          board.hidden = true;
          return;
        }
        var visible = 0;
        board.querySelectorAll("tbody tr").forEach(function (tr) {
          var hit = (!text || tr.getAttribute("data-search").indexOf(text) !== -1)
            && (!model || tr.getAttribute("data-model") === model)
            && (!harness || tr.getAttribute("data-harness") === harness);
          tr.hidden = !hit;
          if (hit) visible += 1;
        });
        board.hidden = visible === 0;
        if (visible) shown += 1;
      });
      if (noMatches) noMatches.hidden = shown !== 0;
    }

    [q, fModel, fHarness, fCaveats].forEach(function (el) {
      if (!el) return;
      el.addEventListener(el.tagName === "INPUT" && el.type === "search"
        ? "input" : "change", applyFilters);
    });
  }

  // --- gateway model tabs ------------------------------------------------
  var modelTabs = Array.prototype.slice.call(
    document.querySelectorAll("[data-gateway-model-tab]")
  );
  var modelPanels = Array.prototype.slice.call(
    document.querySelectorAll("[data-gateway-model-panel]")
  );

  function selectGatewayModel(tab, moveFocus) {
    modelTabs.forEach(function (candidate) {
      var selected = candidate === tab;
      candidate.setAttribute("aria-selected", selected ? "true" : "false");
      candidate.setAttribute("tabindex", selected ? "0" : "-1");
    });
    modelPanels.forEach(function (panel) {
      panel.hidden = panel.id !== tab.getAttribute("aria-controls");
    });
    if (moveFocus) tab.focus();
  }

  modelTabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () {
      selectGatewayModel(tab, false);
    });
    tab.addEventListener("keydown", function (ev) {
      var next = null;
      if (ev.key === "ArrowRight") next = (index + 1) % modelTabs.length;
      if (ev.key === "ArrowLeft") {
        next = (index - 1 + modelTabs.length) % modelTabs.length;
      }
      if (ev.key === "Home") next = 0;
      if (ev.key === "End") next = modelTabs.length - 1;
      if (next !== null) {
        ev.preventDefault();
        selectGatewayModel(modelTabs[next], true);
      }
    });
  });
  if (modelTabs.length) selectGatewayModel(modelTabs[0], false);

  // --- tabs ---------------------------------------------------------------
  var VIEWS = [
    "harness", "gateway", "releases", "methodology", "contact"
  ];

  function showView() {
    var hash = (location.hash || "").replace("#", "");
    if (hash === "gateway-probe") {
      hash = "gateway";
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, "", "#gateway");
      }
    }
    var view = "harness";
    if (VIEWS.indexOf(hash) !== -1) {
      view = hash;
    } else if (hash) {
      // A deep link to a section (#community, #packs) should open the view
      // that section lives in rather than silently falling back.
      var target = document.getElementById(hash);
      var host = target && target.closest('main[id^="view-"]');
      if (host) view = host.id.replace("view-", "");
    }
    VIEWS.forEach(function (name) {
      document.getElementById("view-" + name).hidden = name !== view;
    });
    document.querySelectorAll("nav.tabs a").forEach(function (a) {
      if (a.getAttribute("href") === "#" + view) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
  }
  window.addEventListener("hashchange", showView);
  showView();
})();
"""

def _lede(doc, family="harness"):
    """Family-specific title, neutral description, and coverage facts.

    Deliberately draws no conclusion from the data: the boards carry the
    results, and readers do their own reading.
    """
    bundles = doc["harness"]["bundles"]
    if family == "harness":
        harnesses = {a["harness"] for b in bundles for a in b["arms"]}
        models = {a["model"] for b in bundles for a in b["arms"]}
        valid_rows = sum(b.get("countable_rows") or 0 for b in bundles)
        matched_rows = sum(
            (b.get("matched") or {}).get("matched_rows")
            if b.get("table") == "matched"
            else b.get("countable_rows") or 0
            for b in bundles
        )
        facts = [
            f"{len(harnesses)} harnesses",
            f"{len(models)} models",
            f"{doc['harness']['bundle_count']} result-sealed bundles",
            f"{valid_rows:,} valid result rows",
            f"{matched_rows:,} matched result rows",
        ]
        dates = sorted(b["date"] for b in bundles if b.get("date"))
        if dates:
            facts.append(f"updated {dates[-1]}")
        return (
            "Coding-agent harness benchmarks",
            "Compares coding-agent harnesses while holding the model and task "
            "fixed. Each board is one result-sealed bundle.",
            facts,
        )

    if family == "gateway":
        gateway_bundles = doc["gateway"]["bundles"]
        if not gateway_bundles:
            return (
                "AI gateway request benchmarks",
                "Measures AI gateway routes one model request at a "
                "time under separately scheduled cold and warm conditions.",
                ["0 published request bundles"],
            )
        bundle_count = len(gateway_bundles)
        models = {
            bundle.get("model") or bundle.get("title") or bundle.get("id")
            for bundle in gateway_bundles
        }
        route_counts = {
            len(bundle.get("arms") or []) for bundle in gateway_bundles
        }
        facts = [
            f"{bundle_count} published "
            f"{'bundle' if bundle_count == 1 else 'bundles'}",
            f"{len(models)} benchmarked "
            f"{'model' if len(models) == 1 else 'models'}",
        ]
        if len(route_counts) == 1:
            facts.append(f"{route_counts.pop()} routes per bundle")
        facts.append("model-specific denominators below")
        dates = sorted(
            bundle["date"] for bundle in gateway_bundles if bundle.get("date")
        )
        if dates:
            facts.append(f"updated {dates[-1]}")
        return (
            "AI gateway benchmarks",
            "Compares request latency, throughput, and reliability across "
            "AI gateways under separately scheduled cold and warm "
            "conditions. Select a model below; each bundle keeps its own "
            "request counts and matched-block denominators.",
            facts,
        )

    return (
        "OpenBench benchmark results",
        "Digest-verified benchmark releases, methodology, and project "
        "information.",
        ["coding-agent harnesses", "AI gateway routes"],
    )


_METHODOLOGY = """
<section class="prose">
  <h2>What is being measured</h2>
  <p>OpenBench runs two benchmark families. They share a task contract and a
  checker, and no denominators.</p>

  <h3>Harness Bench</h3>
  <p>Varies the coding-agent harness — the CLI that wraps a model in a run loop,
  tool set, and permission policy — while holding the model and task fixed. An
  arm is <code>(harness, model)</code>. A task is solved when its
  <code>checker.sh</code> exits 0; the harness's own claim of success is never
  trusted.</p>

  <h3>Gateway Bench</h3>
  <p>Measures one model request at a time under separately scheduled cold and
  warm transport conditions. It reports request success, route verification,
  transport and stream phase timing, throughput, usage, and per-request cost.
  It is not a coding-agent outcome benchmark. Gateway Bench requests and
  Harness Bench cells are never pooled or compared as one denominator.</p>
  <p>The Gateway Bench leaderboard uses an absolute summary score: 30% cold
  TTFT median, 15% cold TTFT p95, 30% warm TTFT median, 15% warm TTFT p95,
  and 10% warm median output throughput. TTFT starts when the measured request
  is sent; cold DNS, TCP, and TLS setup are reported separately.
  Latency scores linearly from 100 at zero to zero at 20 seconds; throughput
  scores linearly from zero at 5 tok/s to 100 at 200 tok/s. The weighted result
  is multiplied by request success.
  Cost is excluded. The direct-provider arm is an unranked reference, and the detailed
  measurements below the score remain the factual record.</p>

  <h2>Denominators and intervals</h2>
  <ul>
    <li>Harness Bench denominators are countable cells. Infrastructure and
    rate-limit failures are excluded; other failures, including timeouts, stay
    in the denominator.</li>
    <li>Harness Bench uses Wilson 95% intervals over matched
    <code>(task, trial)</code> cells whenever a bundle has two or more arms.</li>
    <li>Gateway Bench displays complete cold and warm block counts separately.
    Availability uses a Wilson 95% interval over every attempted measured
    request, so gateway errors such as HTTP 429 responses and timeouts remain in
    that denominator. Phase summaries use successful, route-verified requests
    and retain metric-specific coverage. Paired deltas use complete
    gateway/direct blocks and bootstrap 95% intervals.</li>
  </ul>

  <h2>Efficiency and cost</h2>
  <ul>
    <li>Median wall time is taken among solved cells only.</li>
    <li>Each Harness Bench arm uses one complete split-token lane across all
    matched result rows, preferring proxy telemetry and otherwise using native
    telemetry. Fresh tokens are uncached input plus output; cache reads and
    cache writes remain separate. Incomplete lanes produce no token metrics
    and report their row coverage instead.</li>
    <li>Each per-solve token figure sums traffic from every matched attempt,
    including failed attempts, then divides by the number of solved cells. It
    measures attempted traffic required per solve, not the average size of
    successful attempts alone.</li>
    <li>Harness <code>$/solve</code> appears only for models with a configured
    price.</li>
    <li>Gateway Bench response headers are the time until HTTP response headers.
    First body byte and semantic TTFT are reported separately; response headers
    are not labeled TTFB.</li>
    <li>Gateway Bench measured cost is the frozen-list request estimate.
    Charged cost is separately reported billing evidence. Each retains its own
    request coverage, as do total, cached-input, and cache-write token
    readings.</li>
    <li>Harness defaults are not clamped.</li>
  </ul>

  <h2>Comparability</h2>
  <ul>
    <li>Cells from different bundles are never blended. Each board is one
    bundle; cross-bundle ranking on different task sets is not supported.</li>
    <li>Every ranked bundle ships <code>results.jsonl</code> plus a provenance
    digest and is re-verified before it appears here. Digests show
    tamper-evidence, not absence of cherry-picking.</li>
    <li>Results cover only the included tasks, trials, model deployments,
    harness versions, and timeout caps.</li>
  </ul>

  <h2>Reproducing a board</h2>
  <p>Every board links its <code>results.jsonl</code>. Re-check a bundle with
  <code>obench verify &lt;bundle&gt;</code> (harness) or
  <code>obench gateway probe verify &lt;bundle&gt;</code> (gateway), and rebuild
  this page with <code>obench site build</code>.</p>
</section>
"""



# --------------------------------------------------------------------------
# Page — rendered here, enhanced in the browser
# --------------------------------------------------------------------------
#
# Every table is written as real HTML at build time. The script below only
# *enhances* what is already on the page: it re-orders rows, hides rows and
# boards, and switches tabs. Nothing is built client-side, so the page is
# complete with JavaScript switched off and there is exactly one renderer to
# keep honest.


def _esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def _attrs(mapping):
    out = []
    for key, value in mapping.items():
        if value is None or value is False:
            continue
        if value is True:
            out.append(f" {key}")
        else:
            out.append(f' {key}="{_esc(value)}"')
    return "".join(out)


def _tag(name, attrs=None, body=""):
    return f"<{name}{_attrs(attrs or {})}>{body}</{name}>"


def _fmt_pct(value, digits=1):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def _fmt_num(value):
    return "—" if value is None else f"{round(value):,}"


def _fmt_secs(value):
    return "—" if value is None else f"{value:.1f}s"


def _fmt_money(value, digits=3):
    return "—" if value is None else f"${value:.{digits}f}"


def _fmt_score(value):
    return "—" if value is None else f"{value:.3f}"


def _signed(value, fmt):
    if value is None:
        return "—"
    return ("+" if value > 0 else "") + fmt(value)


def _safe_href(value):
    """Return ``value`` if it is safe to put in an ``href``, else ``None``.

    Escaping stops attribute breakout but says nothing about the scheme, so a
    manifest carrying ``javascript:...`` would still render a working link.
    Manifests are validated where they are ingested, but ``community.json``,
    ``releases.json``, and ``gateway.json`` are committed files that a pull
    request can edit directly, bypassing that check — so the renderer enforces
    the rule itself rather than trusting its inputs.

    Allowed: absolute ``http(s)`` URLs, same-document fragments, and relative
    paths. Everything else is dropped, and the caller renders plain text.
    """
    if not isinstance(value, str):
        return None
    # Strip characters a browser ignores when resolving the scheme, so
    # "java\tscript:" and "  javascript:" cannot slip past the prefix check.
    collapsed = "".join(ch for ch in value if ch not in "\t\r\n\x00").strip()
    if not collapsed:
        return None
    lowered = collapsed.lower()
    if lowered.startswith(("http://", "https://")):
        return collapsed
    if collapsed.startswith("//"):
        return None          # protocol-relative: inherits whatever the page used
    if collapsed.startswith("#") or collapsed.startswith("/"):
        return collapsed
    # Relative only when no scheme appears before the first path separator.
    head = re.split(r"[/?#]", collapsed, maxsplit=1)[0]
    return None if ":" in head else collapsed


def _link(href, body, **attrs):
    """An anchor when the target is safe, otherwise the text on its own."""
    safe = _safe_href(href)
    if safe is None:
        return body
    return _tag("a", {"href": safe, **attrs}, body)


def _chip(text, cls="", title=None):
    return _tag("span", {"class": ("chip " + cls).strip(), "title": title}, _esc(text))


def _meta_field(label, value):
    return _tag("span", {}, _tag("b", {}, _esc(label) + " ") + _esc(value))


def _sort_key(value, descending):
    """Order strings alphabetically and numbers by magnitude, either way."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower() if not descending else _ReversedText(value.lower())
    return -value if descending else value


class _ReversedText(str):
    """A string that compares backwards, so text can sort descending."""

    def __lt__(self, other):
        return str.__gt__(self, other)

    def __gt__(self, other):
        return str.__lt__(self, other)


def _clamp01(value):
    return max(0.0, min(1.0, value))


def _interval_cell(estimate, low, high, fmt):
    """Value, a bar on the shared 0–100% track, and the numeric range."""
    marks = ""
    if low is not None and high is not None:
        lo, hi = _clamp01(low), _clamp01(high)
        marks += _tag("div", {
            "class": "span",
            "style": f"left:{lo * 100:.4f}%;width:{max(0.8, (hi - lo) * 100):.4f}%",
        })
    if estimate is not None:
        marks += _tag("div", {
            "class": "dot",
            "style": f"left:calc({_clamp01(estimate) * 100:.4f}% - 1px)",
        })
    range_text = "—" if low is None or high is None else f"{fmt(low)}–{fmt(high)}"
    return _tag("div", {"class": "iv"},
                _tag("span", {"class": "val"}, _esc(fmt(estimate)))
                + _tag("div", {"class": "track"}, marks)
                + _tag("span", {"class": "range"}, _esc(range_text)))


def _delta_cell(metric, fmt, higher_is_better, domain):
    """Signed contrast on a zero-centred axis shared by the whole column.

    Direction is carried by the sign, by the pole hue, and by which side of
    zero the bar sits on, so it never rests on colour alone. An interval
    covering zero is drawn neutral: no effect was detected.
    """
    estimate, low, high = metric["estimate"], metric["low"], metric["high"]
    known = low is not None and high is not None
    if known and low <= 0 <= high:
        tone = "null"
    elif estimate is not None and estimate != 0:
        better = estimate > 0 if higher_is_better else estimate < 0
        tone = "better" if better else "worse"
    else:
        tone = "null"

    def place(value):
        return max(0.0, min(100.0, (0.5 + (value / domain) * 0.5) * 100))

    marks = _tag("div", {"class": "zero"})
    if known:
        a, b = place(low), place(high)
        marks += _tag("div", {
            "class": "span " + tone,
            "style": f"left:{min(a, b):.4f}%;width:{max(0.8, abs(b - a)):.4f}%",
        })
    if estimate is not None:
        marks += _tag("div", {
            "class": "dot " + tone,
            "style": f"left:calc({place(estimate):.4f}% - 1px)",
        })
    title = ("95% CI " + _signed(low, fmt) + " to " + _signed(high, fmt)
             if known else "no interval available")
    return _tag("div", {"class": "dv", "title": title},
                _tag("span", {"class": "val " + tone}, _esc(_signed(estimate, fmt)))
                + _tag("div", {"class": "track"}, marks))


def _delta_domain(rows, key):
    """Widest bound in a contrast column, so its rows share one signed scale."""
    widest = 0.0
    for row in rows:
        metric = row.get(key) or {}
        for field in ("estimate", "low", "high"):
            value = metric.get(field)
            if value is not None:
                widest = max(widest, abs(value))
    return widest or 1.0


def _render_table(columns, rows, sorted_by=None, row_attrs=None):
    """One table. ``columns`` entries are dicts:

    ``label``  header text
    ``cell``   row -> cell HTML
    ``key``    row -> sort key (omit for an unsortable column)
    ``type``   ``num`` or ``str`` (how the browser compares the key)
    ``dir``    default direction when the column is first clicked
    ``axis``   optional tick labels drawn under the header
    ``cls``    optional cell class
    ``plot``   the cell draws a plot (used to budget the plot width)
    ``skip_if_empty``  drop the column when no row has a key

    ``sorted_by`` is a column label. When given, the rows are *actually* sorted
    by that column before rendering, so the header's sort indicator agrees with
    what is on screen. When omitted, the caller's own ordering stands and no
    column claims to be sorted.
    """
    columns = [
        col for col in columns
        if not col.get("skip_if_empty")
        or any(col["key"](row) is not None for row in rows)
    ]

    sort_index = None
    if sorted_by is not None:
        sort_index = next(
            (i for i, col in enumerate(columns) if col["label"] == sorted_by), None)
    if sort_index is not None:
        col = columns[sort_index]
        descending = col.get("dir", "desc") == "desc"
        # Stable, so the caller's ordering survives as the tie-break, and rows
        # with nothing to compare sink either way.
        rows = sorted(
            rows,
            key=lambda r: (col["key"](r) is None,
                           _sort_key(col["key"](r), descending)),
        )

    heads = ""
    for index, col in enumerate(columns):
        sortable = "key" in col
        attrs = {
            "scope": "col",
            "class": "sortable" if sortable else None,
            "data-type": col.get("type", "num") if sortable else None,
            "data-col": str(index) if sortable else None,
            "tabindex": "0" if sortable else None,
            "role": "button" if sortable else None,
        }
        if index == sort_index:
            attrs["aria-sort"] = (
                "descending" if col.get("dir", "desc") == "desc" else "ascending")
        body = _esc(col["label"])
        if sortable:
            arrow = "↕"
            if index == sort_index:
                arrow = "↓" if col.get("dir", "desc") == "desc" else "↑"
            body += _tag("span", {"class": "arrow"}, arrow)
        if col.get("axis"):
            body += _tag("div", {"class": "axis"},
                         "".join(_tag("span", {}, _esc(t)) for t in col["axis"]))
        heads += _tag("th", attrs, body)

    body_rows = ""
    for row in rows:
        cells = ""
        keys = {}
        for index, col in enumerate(columns):
            cells += _tag("td", {"class": col.get("cls")}, col["cell"](row))
            if "key" in col:
                value = col["key"](row)
                keys[f"data-s{index}"] = "" if value is None else str(value)
        attrs = dict(row_attrs(row) if row_attrs else {})
        attrs.update(keys)
        body_rows += _tag("tr", attrs, cells)

    # Plot width is a per-table budget: four contrast columns cannot each be as
    # wide as a single solve-rate column. Dense tables also drop the printed
    # interval, which the bar already carries.
    plots = sum(1 for col in columns if col.get("plot"))
    plot_w = {0: 140, 1: 148, 2: 116, 3: 100}.get(plots, 84)
    if len(columns) > 7:
        plot_w = min(plot_w, 116)
    attrs = {"class": "scroll", "style": f"--plot-w:{plot_w}px"}
    if len(columns) > 6 or plots > 2:
        attrs["data-dense"] = "1"
    return _tag("div", attrs, _tag(
        "table", {},
        _tag("thead", {}, _tag("tr", {}, heads))
        + _tag("tbody", {}, body_rows)))


def _harness_board(bundle):
    arms = bundle["arms"]
    title = _esc(bundle["title"])
    if bundle.get("path"):
        title = _link(bundle["path"], title)

    meta = "".join(filter(None, [
        _meta_field("kind", bundle["kind"]),
        _meta_field("date", bundle["date"]) if bundle.get("date") else None,
        _meta_field("denominators",
                    "matched (task, trial)" if bundle["table"] == "matched"
                    else "all countable"),
        _meta_field(
            "matched rows",
            (bundle.get("matched") or {}).get("matched_rows")
            if bundle["table"] == "matched"
            else bundle["countable_rows"],
        ),
        _meta_field(
            "common task/trials",
            (bundle.get("matched") or {}).get("matched_cells_per_group"),
        ) if bundle["table"] == "matched" else None,
        _meta_field("results SHA", (bundle.get("results_sha256") or "")[:12])
        if bundle.get("results_sha256") else None,
        _meta_field("task-set SHA", (bundle.get("task_set_digest") or "")[:12])
        if bundle.get("task_set_digest") else None,
    ]))

    chips = "".join(_chip(m) for m in bundle.get("models") or [])
    if bundle.get("has_caveats"):
        chips += _chip("caveats disclosed", "warn")

    caveats = ""
    if bundle.get("has_caveats"):
        items = "".join(_tag("li", {}, _esc(c)) for c in bundle["caveats"])
        caveats = _tag("details", {"class": "caveats"},
                       _tag("summary", {},
                            f"{len(bundle['caveats'])} caveat(s) from the release page")
                       + _tag("ul", {}, items))

    head = _tag("div", {"class": "head"},
                _tag("h2", {}, title)
                + _tag("div", {"class": "meta"}, meta)
                + (_tag("div", {"class": "chips"}, chips) if chips else "")
                + caveats)

    # The model is a header fact when a board pins one; only worth a column
    # when a board actually compares more than one.
    many_models = len({a["model"] for a in arms}) > 1

    def telemetry_basis(arm):
        source = arm.get("token_telemetry_source")
        bases = arm.get("token_telemetry_bases") or []
        if source is None:
            return _chip("unavailable", "warn")
        return _chip(source) + "".join(_chip(basis) for basis in bases)

    def telemetry_coverage(arm):
        coverage = arm.get("token_telemetry_coverage") or {}
        total = coverage.get("total_rows") or 0
        proxy = coverage.get("proxy_covered_rows", 0)
        native = coverage.get("native_covered_rows", 0)
        return f"proxy {proxy}/{total} · native {native}/{total}"

    columns = [
        {"label": "Harness", "cls": "name", "type": "str", "dir": "asc",
         "cell": lambda a: _esc(a["harness"]), "key": lambda a: a["harness"]},
    ]
    if many_models:
        columns.append(
            {"label": "Model", "type": "str", "dir": "asc",
             "cell": lambda a: _esc(a["model"]), "key": lambda a: a["model"]})
    columns += [
        {"label": "Solve rate · Wilson 95%", "axis": ["0", "50", "100%"], "plot": True,
         "cell": lambda a: _interval_cell(
             a["solve_rate"], (a.get("wilson95") or [None, None])[0],
             (a.get("wilson95") or [None, None])[1], _fmt_pct),
         "key": lambda a: a["solve_rate"]},
        {"label": "Solved",
         "cell": lambda a: f"{a['solved']}/{a['n']}",
         "key": lambda a: (a["solved"] / a["n"]) if a["n"] else None},
        {"label": "Median wall", "dir": "asc", "skip_if_empty": True,
         "cell": lambda a: _fmt_secs(a.get("median_wall_s")),
         "key": lambda a: a.get("median_wall_s")},
        {"label": "Fresh tokens/solve", "dir": "asc",
         "cell": lambda a: _fmt_num(a.get("fresh_tokens_per_solve")),
         "key": lambda a: a.get("fresh_tokens_per_solve")},
        {"label": "Uncached input/solve", "dir": "asc",
         "cell": lambda a: _fmt_num(a.get("tokens_input_uncached_per_solve")),
         "key": lambda a: a.get("tokens_input_uncached_per_solve")},
        {"label": "Output/solve", "dir": "asc",
         "cell": lambda a: _fmt_num(a.get("tokens_output_per_solve")),
         "key": lambda a: a.get("tokens_output_per_solve")},
        {"label": "Cache-read/solve", "dir": "asc",
         "cell": lambda a: _fmt_num(a.get("tokens_cache_read_per_solve")),
         "key": lambda a: a.get("tokens_cache_read_per_solve")},
        {"label": "Cache-write/solve", "dir": "asc",
         "cell": lambda a: _fmt_num(a.get("tokens_cache_write_per_solve")),
         "key": lambda a: a.get("tokens_cache_write_per_solve")},
        {"label": "$/solve", "dir": "asc", "skip_if_empty": True,
         "cell": lambda a: _fmt_money(a.get("cost_per_solve_usd")),
         "key": lambda a: a.get("cost_per_solve_usd")},
        {"label": "Telemetry source / basis",
         "cell": lambda a: _tag("div", {"class": "chips"}, telemetry_basis(a))},
        {"label": "Telemetry coverage",
         "cell": telemetry_coverage},
    ]

    table = _render_table(columns, arms, "Solve rate · Wilson 95%", row_attrs=lambda a: {
        "data-harness": a["harness"],
        "data-model": a["model"],
        "data-search": f"{a['harness']} {a['model']}".lower(),
    })

    links = "".join(filter(None, [
        _tag("span", {}, _link(bundle["results_path"], "results.jsonl"))
        if bundle.get("results_path") else None,
        _tag("span", {}, _link(bundle["path"], "release page"))
        if bundle.get("path") else None,
    ]))
    foot = _tag("div", {"class": "head"}, _tag("div", {"class": "meta"}, links))

    return _tag("section", {
        "class": "board",
        "data-models": ",".join(sorted({a["model"] for a in arms})),
        "data-harnesses": ",".join(sorted({a["harness"] for a in arms})),
        "data-caveats": "1" if bundle.get("has_caveats") else "0",
    }, head + table + foot)


def _gateway_board(bundle):
    title = _esc(bundle["title"])
    if bundle.get("path"):
        title = _link(bundle["path"], title)

    meta = "".join(filter(None, [
        _meta_field("track", bundle.get("track") or "gateway_tax"),
        _meta_field("harness", bundle["harness"]) if bundle.get("harness") else None,
        _meta_field("date", bundle["date"]) if bundle.get("date") else None,
        _meta_field("blocks", f"{bundle['blocks_included']}/{bundle['blocks_observed']}"),
        _meta_field(
            "cap-affected blocks",
            f"{bundle.get('blocks_max_calls_affected', 0)}/{bundle['blocks_included']}",
        ),
        _meta_field("tasks", bundle["tasks_included"]),
        _meta_field("lane", bundle["execution_lane"]) if bundle.get("execution_lane") else None,
        _meta_field("model match", bundle["model_match"])
        if bundle.get("model_match") else None,
        _meta_field("provider prompt", bundle["provider_prompt_mode"])
        if bundle.get("provider_prompt_mode") else None,
        _meta_field("experiment", (bundle.get("experiment_digest") or "")[:12])
        if bundle.get("experiment_digest") else None,
    ]))
    excluded = bundle.get("blocks_excluded") or {}
    chips = "".join(_chip(f"excluded: {reason} × {count}", "warn")
                    for reason, count in sorted(excluded.items()))
    head = _tag("div", {"class": "head"},
                _tag("h2", {}, title)
                + _tag("div", {"class": "meta"}, meta)
                + (_tag("div", {"class": "chips"}, chips) if chips else ""))

    def metric(arm, name):
        return arm.get(name) or {}

    def coverage_text(coverage, *, noun=None):
        coverage = coverage or {}
        covered = coverage.get("covered")
        total = coverage.get("total")
        if covered is None or total is None:
            return "—"
        value = f"{covered}/{total}"
        return f"{value} {noun}" if noun else value

    def grouped_coverage(row, groups, coverage_name):
        parts = []
        for group_label, members in groups:
            values = [
                coverage_text(metric(row, name).get(coverage_name))
                for name, _short_label in members
            ]
            if len(set(values)) == 1:
                parts.append(f"{group_label} {values[0]}")
            else:
                parts.extend(
                    f"{short_label} {value}"
                    for (_name, short_label), value in zip(members, values)
                )
        return " · ".join(parts)

    def route_distribution(arm):
        routes = arm.get("route_distribution") or {}
        if not routes:
            return "—"
        parts = []
        for name, evidence in sorted(routes.items()):
            share = evidence.get("share") if isinstance(evidence, dict) else None
            task_coverage = (
                evidence.get("task_coverage") if isinstance(evidence, dict) else None
            )
            details = []
            if share is not None:
                details.append(_fmt_pct(share, 0))
            if task_coverage:
                details.append(
                    "tasks " + coverage_text(task_coverage)
                )
            parts.append(
                _esc(name)
                + (
                    " " + _tag("span", {"class": "sub"}, " · ".join(details))
                    if details else ""
                )
            )
        return "<br>".join(parts)

    def outcome_coverage(arm):
        cost = (arm.get("cost") or {}).get("coverage") or {}
        outcome = grouped_coverage(
            arm,
            (("outcome", (
                ("solve_rate", "solve"),
                ("mean_checker_score", "score"),
                ("availability", "availability"),
                ("latency_s", "latency"),
            )),),
            "cell_coverage",
        )
        parts = [outcome] if outcome else []
        if cost:
            parts.append(
                "cost "
                + f"{cost.get('covered_calls', 0)}/{cost.get('total_calls', 0)} calls"
            )
        return " · ".join(parts) or "—"

    columns = [
        {"label": "Route", "cls": "name", "type": "str", "dir": "asc",
         "cell": lambda a: _gateway_route_cell(a["arm_id"]),
         "key": lambda a: a["arm_id"]},
        {"label": "Role", "type": "str", "dir": "asc",
         "cell": lambda a: _tag("div", {"class": "chips"},
                                _chip(a.get("role") or "—",
                                      "role-direct" if a.get("role") == "direct" else "")
                                + (_chip(a["requested_provider"])
                                   if a.get("requested_provider") else "")),
         "key": lambda a: a.get("role") or ""},
        {"label": "Solve rate · 95% CI", "axis": ["0", "50", "100%"], "plot": True,
         "cell": lambda a: _interval_cell(
             a["solve_rate"]["estimate"], a["solve_rate"]["low"],
             a["solve_rate"]["high"], _fmt_pct),
         "key": lambda a: a["solve_rate"]["estimate"]},
        {"label": "Mean score",
         "cell": lambda a: _fmt_score(a["mean_checker_score"]["estimate"]),
         "key": lambda a: a["mean_checker_score"]["estimate"]},
        {"label": "Availability",
         "cell": lambda a: _fmt_pct(a["availability"]["estimate"]),
         "key": lambda a: a["availability"]["estimate"]},
        {"label": (
            f"{bundle['budget_max_calls']}-call cap"
            if bundle.get("budget_max_calls") is not None
            else "Call cap"
         ),
         "cell": lambda a: (
             f"{a['max_calls']['cells']}/{a['max_calls']['total_cells']} "
             f"({_fmt_pct(a['max_calls']['ratio'])})"
         ),
         "key": lambda a: a["max_calls"]["ratio"]},
        {"label": "Median E2E cell latency", "dir": "asc",
         "cell": lambda a: _fmt_secs(a["latency_s"]["estimate"]),
         "key": lambda a: a["latency_s"]["estimate"]},
        {"label": "List-est. $/solve", "dir": "asc", "skip_if_empty": True,
         "cell": lambda a: _fmt_money((a.get("cost") or {}).get("cost_per_solve_usd"), 4),
         "key": lambda a: (a.get("cost") or {}).get("cost_per_solve_usd")},
        {"label": "Outcome / cost coverage",
         "cell": outcome_coverage},
    ]
    parts = head + _render_table(columns, bundle["arms"])

    def telemetry_coverage(arm):
        groups = (
            ("timing", (
                ("ttfb_s", "TTFB"),
                ("semantic_ttft_s", "TTFT"),
            )),
            ("throughput", (("throughput_tokens_per_s", "throughput"),)),
            ("tokens", (
                ("mean_input_tokens_per_call", "input"),
                ("mean_output_tokens_per_call", "output"),
                ("mean_total_tokens_per_call", "total"),
            )),
            ("cache", (
                ("cache_hit_call_rate", "cache hits"),
                ("cached_input_fraction", "cached fraction"),
                ("mean_cached_input_tokens_per_call", "cache read"),
            )),
            ("cache write", (
                ("mean_cache_write_input_tokens_per_call", "cache write"),
            )),
        )
        return grouped_coverage(arm, groups, "call_coverage")

    telemetry_columns = [
        {"label": "Route", "cls": "name", "type": "str", "dir": "asc",
         "cell": lambda a: _gateway_route_cell(a["arm_id"]),
         "key": lambda a: a["arm_id"]},
        {"label": "Served route",
         "cell": route_distribution},
        {"label": "TTFB", "dir": "asc",
         "cell": lambda a: _fmt_secs(metric(a, "ttfb_s").get("estimate")),
         "key": lambda a: metric(a, "ttfb_s").get("estimate")},
        {"label": "Semantic TTFT", "dir": "asc",
         "cell": lambda a: _fmt_secs(metric(a, "semantic_ttft_s").get("estimate")),
         "key": lambda a: metric(a, "semantic_ttft_s").get("estimate")},
        {"label": "Output tok/s", "dir": "desc",
         "cell": lambda a: (
             "—" if metric(a, "throughput_tokens_per_s").get("estimate") is None
             else f"{metric(a, 'throughput_tokens_per_s')['estimate']:.1f}"
         ),
         "key": lambda a: metric(a, "throughput_tokens_per_s").get("estimate")},
        {"label": "Input / output / total tok/call",
         "cell": lambda a: (
             f"{_fmt_num(metric(a, 'mean_input_tokens_per_call').get('estimate'))}"
             f" / {_fmt_num(metric(a, 'mean_output_tokens_per_call').get('estimate'))}"
             f" / {_fmt_num(metric(a, 'mean_total_tokens_per_call').get('estimate'))}"
         )},
        {"label": "Cache-hit calls",
         "cell": lambda a: _fmt_pct(metric(a, "cache_hit_call_rate").get("estimate"))},
        {"label": "Cached input",
         "cell": lambda a: _fmt_pct(metric(a, "cached_input_fraction").get("estimate"))},
        {"label": "Cache read / write tok/call",
         "cell": lambda a: (
             f"{_fmt_num(metric(a, 'mean_cached_input_tokens_per_call').get('estimate'))}"
             f" / "
             f"{_fmt_num(metric(a, 'mean_cache_write_input_tokens_per_call').get('estimate'))}"
         )},
        {"label": "Telemetry coverage",
         "cell": telemetry_coverage},
    ]
    parts += _tag(
        "div",
        {"class": "head"},
        _tag("h2", {}, "Serving telemetry")
        + _tag(
            "p",
            {},
            "Per-call proxy telemetry. The median E2E value above is per "
            "completed or timeout-capped benchmark cell, not request latency. "
            "Cache fields are provider-reported prompt-prefix behavior under "
            f"{_tag('code', {}, _esc(bundle.get('provider_prompt_mode') or 'unknown'))}; "
            "gateway response caching was disabled.",
        ),
    )
    parts += _render_table(telemetry_columns, bundle["arms"])

    contrasts = bundle.get("contrasts") or []
    if contrasts:
        def delta_column(
            label, key, fmt, higher_is_better, direction="desc", *,
            skip_if_empty=False,
        ):
            domain = _delta_domain(contrasts, key)
            return {
                "label": label, "dir": direction, "plot": True,
                "cell": lambda r: _delta_cell(
                    r.get(key) or {}, fmt, higher_is_better, domain
                ),
                "key": lambda r: (r.get(key) or {}).get("estimate"),
                "skip_if_empty": skip_if_empty,
            }

        tax_columns = [
            {"label": "Gateway arm", "cls": "name", "type": "str", "dir": "asc",
             "cell": lambda r: _gateway_route_cell(r["arm_id"]),
             "key": lambda r: r["arm_id"]},
            {"label": "vs direct",
             "cell": lambda r: _gateway_route_cell(r["direct_arm"])},
            delta_column("Δ solve rate", "solve_rate", _fmt_pct, True),
            delta_column("Δ mean score", "mean_checker_score", _fmt_score, True),
            delta_column("Δ availability", "availability", _fmt_pct, True),
            delta_column("Δ median E2E cell latency", "latency_s",
                         lambda v: f"{v:.2f}s", False, "asc"),
        ]
        legend = "".join(
            _tag("span", {}, _tag("i", {"class": tone}, "") + _esc(text))
            for tone, text in (
                ("better", "Gateway better than direct"),
                ("worse", "Gateway worse than direct"),
                ("null", "Interval spans zero — no detected effect"),
            ))
        parts += _tag("div", {"class": "head"},
                      _tag("h2", {}, "Gateway tax")
                      + _tag("p", {},
                             "Paired difference from the direct control arm, "
                             "bootstrap 95% intervals.")
                      + _tag("div", {"class": "legend"}, legend))
        parts += _render_table(tax_columns, contrasts)

        telemetry_tax_columns = [
            {"label": "Gateway arm", "cls": "name", "type": "str", "dir": "asc",
             "cell": lambda r: _gateway_route_cell(r["arm_id"]),
             "key": lambda r: r["arm_id"]},
            {"label": "vs direct",
             "cell": lambda r: _gateway_route_cell(r["direct_arm"])},
            delta_column(
                "Δ TTFB", "ttfb_s", lambda v: f"{v:.2f}s", False, "asc",
                skip_if_empty=True,
            ),
            delta_column(
                "Δ semantic TTFT", "semantic_ttft_s",
                lambda v: f"{v:.2f}s", False, "asc",
                skip_if_empty=True,
            ),
            delta_column(
                "Δ output tok/s", "throughput_tokens_per_s",
                lambda v: f"{v:.1f}", True, skip_if_empty=True,
            ),
            delta_column(
                "Δ input tok/call", "mean_input_tokens_per_call",
                lambda v: f"{v:.0f}", False, "asc",
                skip_if_empty=True,
            ),
            delta_column(
                "Δ output tok/call", "mean_output_tokens_per_call",
                lambda v: f"{v:.0f}", False, "asc",
                skip_if_empty=True,
            ),
            delta_column(
                "Δ list cost / attempted cell", "attempted_cost_usd",
                lambda v: f"${v:.4f}", False, "asc",
                skip_if_empty=True,
            ),
            {"label": "Paired block coverage",
             "cell": lambda r: grouped_coverage(
                 r,
                 (
                     ("timing", (
                         ("ttfb_s", "TTFB"),
                         ("semantic_ttft_s", "TTFT"),
                     )),
                     ("throughput", (
                         ("throughput_tokens_per_s", "throughput"),
                     )),
                     ("tokens", (
                         ("mean_input_tokens_per_call", "input"),
                         ("mean_output_tokens_per_call", "output"),
                     )),
                     ("cost", (("attempted_cost_usd", "cost"),)),
                 ),
                 "paired_block_coverage",
             )},
        ]
        parts += _tag(
            "div",
            {"class": "head"},
            _tag("h2", {}, "Serving telemetry tax")
            + _tag(
                "p",
                {},
                "Paired gateway-minus-direct per-call telemetry. Cost is the "
                "frozen-list estimate per attempted cell, not cost per solve. "
                "Provider cache-accounting deltas are intentionally omitted.",
            ),
        )
        parts += _render_table(telemetry_tax_columns, contrasts)

    return _tag("section", {"class": "board", "data-caveats": "0"}, parts)


def _gateway_route_key(arm_id):
    arm_id = arm_id or ""
    for provider in ("cloudflare", "concentrate", "openrouter", "vercel"):
        if arm_id.startswith(provider + "-"):
            return provider + "-openai"
    for key in (
        "direct-openai",
        "direct-moonshot",
    ):
        if arm_id == key or arm_id.startswith(key + "-"):
            return key
    return arm_id


def _gateway_probe_route_name(arm_id):
    return {
        "cloudflare-openai": "Cloudflare",
        "concentrate-openai": "Concentrate",
        "direct-openai": "Direct OpenAI",
        "direct-moonshot": "Direct Moonshot",
        "openrouter-openai": "OpenRouter",
        "vercel-openai": "Vercel",
    }.get(_gateway_route_key(arm_id), arm_id)


def _gateway_probe_logo(arm_id):
    """Return a compact inline mark without adding external page resources."""
    arm_id = _gateway_route_key(arm_id)
    attrs = {
        "class": "route-logo",
        "aria-hidden": "true",
    }
    svg_attrs = {
        "xmlns": "http://www.w3.org/2000/svg",
        "focusable": "false",
    }
    if arm_id == "cloudflare-openai":
        svg_attrs["viewBox"] = "0 0 177 80"
        paths = (
            '<path fill="#F6821F" d="M120.36 78.957l.894-3.141c1.084-3.717.'
            '68-7.178-1.128-9.698-1.658-2.329-4.423-3.696-7.783-3.867L48.'
            '761 61.44c-.425-.022-.787-.214-1-.535-.212-.32-.276-.748-.127'
            '-1.153.213-.62.83-1.111 1.467-1.133l64.157-.811c7.613-.342 '
            '15.842-6.558 18.735-14.121l3.657-9.613c.106-.257.149-.534.149'
            '-.812 0-.15-.021-.299-.043-.449C131.631 14.035 114.938 0 '
            '95.012 0 76.64 0 61.031 11.92 55.438 28.477c-3.615-2.714-8.'
            '23-4.166-13.206-3.675-8.824.876-15.906 8.011-16.778 16.877-.'
            '234 2.307-.042 4.507.489 6.601C11.547 48.707 0 60.563 0 75.'
            '111c0 1.325.106 2.606.276 3.888.085.62.617 1.069 1.234 1.'
            '069l117.362.021h.042c.66-.021 1.255-.47 1.446-1.132Z"/>'
            '<path fill="#FBAD41" d="M141.541 34.782c-.595 0-1.17.022-1.765.'
            '043-.106 0-.191.021-.276.064-.298.107-.553.363-.638.684l-2.'
            '509 8.673c-1.085 3.717-.681 7.178 1.127 9.698 1.658 2.329 4.'
            '423 3.696 7.783 3.867l13.546.812c.403.021.744.214.956.534.234.'
            '321.277.748.149 1.154-.212.619-.829 1.111-1.467 1.132l-14.078'
            '.812c-7.634.363-15.885 6.558-18.777 14.121l-1.02 2.67c-.192.'
            '492.17 1.004.659 1.026h48.506c.574 0 1.084-.385 1.254-.94.851'
            '-3.013 1.298-6.174 1.298-9.464 0-19.248-15.567-34.886-34.748'
            '-34.886Z"/>'
        )
        svg = _tag("svg", svg_attrs, paths)
    elif arm_id == "openrouter-openai":
        svg_attrs["viewBox"] = "0 0 362 259"
        svg = _tag(
            "svg",
            svg_attrs,
            '<path fill="#7624F4" d="M284.128 0c42.797 0 77.489 34.693 '
            '77.489 77.489s-34.692 77.49-77.489 77.49l76.861 76.862c9.764 '
            '9.763 2.849 26.457-10.957 26.457H129.149C57.822 258.298 0 '
            '200.476 0 129.149S57.822 0 129.149 0h154.979ZM129.149 51.66c'
            '-42.796 0-77.489 34.693-77.489 77.489s34.693 77.489 77.489 '
            '77.489 77.489-34.693 77.489-77.489S171.945 51.66 129.149 51.66Z"/>',
        )
    elif arm_id == "vercel-openai":
        svg_attrs["viewBox"] = "0 0 92 80"
        svg = _tag(
            "svg",
            svg_attrs,
            '<path fill="currentColor" d="M91.575 80 45.788 0 0 80h91.575Z"/>',
        )
    elif arm_id == "direct-openai":
        svg_attrs["viewBox"] = "0 0 20 20"
        svg = _tag(
            "svg",
            svg_attrs,
            '<path fill="currentColor" d="M11.248 18.25c-.55 0-1.073-.105-1.568'
            '-.314a4.3 4.3 0 0 1-1.32-.874 4 4 0 0 1-1.304.214 4 4 0 0 '
            '1-2.046-.544 4.27 4.27 0 0 1-1.518-1.485 4 4 0 0 1-.56-2.095'
            'c0-.32.044-.667.131-1.04A4.4 4.4 0 0 1 2.04 10.71a4.07 4.07 '
            '0 0 1 .017-3.4 4.2 4.2 0 0 1 1.056-1.418 3.8 3.8 0 0 1 1.6'
            '-.842 3.9 3.9 0 0 1 .76-1.683q.593-.759 1.451-1.188a4.04 4.04'
            ' 0 0 1 1.832-.429q.825 0 1.567.313.742.314 1.32.875a4 4 0 0'
            ' 1 1.304-.215q1.106 0 2.046.545a4.14 4.14 0 0 1 1.501 1.485'
            'q.578.941.578 2.095 0 .48-.132 1.04.66.61 1.023 1.419.363.792'
            '.363 1.666 0 .892-.38 1.717a4.3 4.3 0 0 1-1.072 1.435 3.8 '
            '3.8 0 0 1-1.584.825 3.8 3.8 0 0 1-.775 1.683 4.06 4.06 0 0 '
            '1-1.436 1.188 4.04 4.04 0 0 1-1.832.429m-4.076-2.062q.825 0 '
            '1.435-.347l3.103-1.782a.36.36 0 0 0 .164-.313v-1.42L7.881 '
            '14.62a.67.67 0 0 1-.726 0l-3.118-1.798v.313q0 .841.396 1.551.'
            '413.693 1.139 1.089a3.2 3.2 0 0 0 1.617.412m.165-2.69q.099.05'
            '.181.05.083 0 .165-.05l1.238-.71-3.977-2.31a.7.7 0 0 1-.363'
            '-.643v-3.58q-.825.362-1.32 1.122a2.9 2.9 0 0 0-.495 1.65q0 .'
            '809.413 1.55.412.743 1.072 1.123zm3.91 3.663q.875 0 1.585-.'
            '396a2.96 2.96 0 0 0 1.534-2.64v-3.564a.32.32 0 0 0-.165-.297'
            'l-1.254-.726v4.604a.7.7 0 0 1-.363.643l-3.119 1.799a3 3 0 0 '
            '0 1.783.577m.627-6.039V8.878L10.01 7.822 8.129 8.878v2.244l1.'
            '881 1.056zM7.057 5.859a.7.7 0 0 1 .363-.644l3.119-1.798a3 3 0'
            ' 0 0-1.782-.578q-.874 0-1.584.396A2.96 2.96 0 0 0 6.05 4.324'
            'a3.07 3.07 0 0 0-.396 1.551v3.547q0 .199.165.314l1.237.726zm'
            '8.383 7.887q.825-.364 1.303-1.123.495-.758.495-1.65a3.15 3.15 '
            '0 0 0-.412-1.55q-.413-.743-1.073-1.123l-3.086-1.782q-.099-.'
            '065-.181-.049a.3.3 0 0 0-.165.05l-1.238.692 3.993 2.327q.165.'
            '099.264.264.1.165.1.363zm-3.317-8.382a.63.63 0 0 1 .726 0l3.'
            '135 1.831v-.297q0-.792-.396-1.501a2.86 2.86 0 0 0-1.105-1.155'
            'q-.71-.43-1.65-.43-.825 0-1.436.347L8.294 5.941a.36.36 0 0 0'
            '-.165.314v1.418z"/>',
        )
    elif arm_id == "concentrate-openai":
        svg_attrs["viewBox"] = "0 0 80 80"
        dots = []
        for row in range(8):
            for column in range(8):
                strength = (column / 7) * (1 - (row / 7))
                radius = 0.35 + 3.97 * (strength ** 0.62)
                dots.append(
                    f'<circle cx="{6.4 + 9.6 * column:g}" '
                    f'cy="{6.4 + 9.6 * row:g}" r="{radius:.3f}" '
                    'fill="currentColor"/>'
                )
        svg = _tag("svg", svg_attrs, "".join(dots))
    else:
        initial = (_gateway_probe_route_name(arm_id) or "?")[:1].upper()
        svg_attrs["viewBox"] = "0 0 32 32"
        svg = _tag(
            "svg",
            svg_attrs,
            '<circle cx="16" cy="16" r="15" fill="none" '
            'stroke="currentColor"/>'
            + _tag(
                "text",
                {
                    "x": "16",
                    "y": "21",
                    "text-anchor": "middle",
                    "font-size": "15",
                    "fill": "currentColor",
                },
                _esc(initial),
            ),
        )
    return _tag("span", attrs, svg)


def _gateway_route_cell(arm_id):
    return _tag(
        "span",
        {"class": "provider-cell"},
        _gateway_probe_logo(arm_id)
        + _tag("span", {}, _esc(_gateway_probe_route_name(arm_id))),
    )


_GATEWAY_COMPOSITE_WEIGHTS = {
    ("cold", "p50"): 0.30,
    ("cold", "p95"): 0.15,
    ("warm", "p50"): 0.30,
    ("warm", "p95"): 0.15,
}


def _gateway_probe_composite_scores(bundle):
    """Return absolute, cost-free route scores on fixed 0-100 scales."""
    arms = {arm["arm_id"]: arm for arm in bundle.get("arms") or []}
    baseline_id = bundle.get("baseline_arm_id")
    if baseline_id not in arms:
        return []

    def metric(arm, condition, name, percentile):
        condition_data = (arm.get("conditions") or {}).get(condition) or {}
        return (
            ((condition_data.get("metrics") or {}).get(name) or {})
            .get(percentile)
        )

    def metric_complete(arm, condition, name):
        condition_data = (arm.get("conditions") or {}).get(condition) or {}
        denominators = condition_data.get("denominators") or {}
        summary = (condition_data.get("metrics") or {}).get(name) or {}
        coverage = summary.get("coverage") or {}
        scheduled = denominators.get("scheduled")
        attempted = denominators.get("attempted")
        success = denominators.get("success")
        verified = denominators.get("route_verified")
        covered = coverage.get("covered")
        return (
            scheduled is not None
            and attempted == scheduled
            and success == verified
            and covered == success
        )

    def availability(arm):
        successes = 0
        attempted = 0
        for condition in ("cold", "warm"):
            value = (
                ((arm.get("conditions") or {}).get(condition) or {})
                .get("availability") or {}
            )
            if value.get("successes") is None or value.get("attempted") is None:
                return None
            successes += value["successes"]
            attempted += value["attempted"]
        return successes / attempted if attempted else None

    def score_latency(value):
        return max(0.0, min(100.0, 100.0 * (1.0 - value / 20.0)))

    def score_throughput(value):
        return max(0.0, min(100.0, (value - 5.0) / 195.0 * 100.0))

    rows = []
    for arm in arms.values():
        weighted_score = 0.0
        complete = True
        for (condition, percentile), weight in _GATEWAY_COMPOSITE_WEIGHTS.items():
            metric_name = "request_to_semantic_ttft_s"
            value = metric(
                arm, condition, metric_name, percentile
            )
            if value is None or not metric_complete(
                arm, condition, metric_name
            ):
                complete = False
                break
            weighted_score += weight * score_latency(value)

        throughput = metric(
            arm, "warm", "throughput_tokens_per_s", "p50"
        )
        route_availability = availability(arm)
        if (
            not complete
            or throughput is None
            or not metric_complete(
                arm, "warm", "throughput_tokens_per_s"
            )
            or route_availability is None
        ):
            continue
        weighted_score += 0.10 * score_throughput(throughput)
        score = weighted_score * route_availability
        rows.append({
            "arm_id": arm["arm_id"],
            "baseline": arm["arm_id"] == baseline_id,
            "score": score,
            "availability": route_availability,
        })
    return rows


def _gateway_probe_leaderboard(bundle):
    rows = _gateway_probe_composite_scores(bundle)
    if not rows:
        return ""
    baseline_name = _gateway_probe_route_name(bundle.get("baseline_arm_id"))

    gateway_rows = sorted(
        (row for row in rows if not row["baseline"]),
        key=lambda item: (-item["score"], item["arm_id"]),
    )
    entries = []
    for position, row in enumerate(gateway_rows, 1):
        entries.append(_tag(
            "div",
            {"class": "route-rank"},
            _tag("div", {"class": "route-position"}, str(position))
            + _tag(
                "div",
                {"class": "route-identity"},
                _gateway_probe_logo(row["arm_id"])
                + _tag(
                    "div",
                    {"class": "route-copy"},
                    _tag(
                        "div",
                        {"class": "route-name"},
                        _esc(_gateway_probe_route_name(row["arm_id"])),
                    )
                    + _tag(
                        "div",
                        {"class": "route-detail"},
                        f"{row['availability'] * 100:.0f}% request success",
                    ),
                ),
            )
            + _tag(
                "div",
                {"class": "route-measure"},
                _tag("strong", {}, f"{row['score']:.1f}")
                + _tag("span", {}, "composite"),
            ),
        ))

    return (
        _tag(
            "div",
            {"class": "head"},
            _tag("h2", {}, "Gateway leaderboard")
            + _tag(
                "p",
                {},
                "OpenBench Composite: absolute TTFT latency, output throughput, "
                "and request success on a 0–100 scale. Higher is better; cost "
                "is excluded and "
                + _esc(baseline_name)
                + " is an unranked reference.",
            ),
        )
        + _tag("div", {"class": "route-leaderboard"}, "".join(entries))
    )


def _gateway_probe_board(bundle):
    title = _esc(bundle["title"])
    if bundle.get("path"):
        title = _link(bundle["path"], title)

    run_note = _gateway_run_note(bundle.get("run_note"))
    scheduled = bundle.get("scheduled_blocks_per_condition")
    complete = bundle.get("complete_blocks") or {}
    cold_blocks = complete.get("cold", 0)
    warm_blocks = complete.get("warm", 0)
    is_spike = scheduled == 1
    baseline_name = _gateway_probe_route_name(bundle.get("baseline_arm_id"))
    meta = "".join(filter(None, [
        _meta_field("date", bundle["date"]) if bundle.get("date") else None,
        _meta_field("model match", bundle["model_match"])
        if bundle.get("model_match") else None,
        _meta_field("cold blocks", f"{complete.get('cold', 0)}/{scheduled}"),
        _meta_field("warm blocks", f"{complete.get('warm', 0)}/{scheduled}"),
        _meta_field("requests", bundle.get("result_count")),
        _meta_field(
            "verified commit",
            (bundle.get("verified_with_commit") or "")[:12],
        ) if bundle.get("verified_with_commit") else None,
        _meta_field(
            "experiment",
            (bundle.get("experiment_digest") or "")[:12],
        ) if bundle.get("experiment_digest") else None,
    ]))
    head = _tag(
        "div",
        {"class": "head"},
        _tag("h2", {}, title)
        + _tag("div", {"class": "meta"}, meta)
        + _tag(
            "p",
            {
                "class": (
                    "evidence-depth is-spike" if is_spike else "evidence-depth"
                )
            },
            (
                f"Evidence depth: {cold_blocks} cold + {warm_blocks} warm "
                "matched blocks per route."
                + (
                    " This is a spike denominator and is not maturity-equivalent "
                    "to bundles with larger matched-block denominators."
                    if is_spike else ""
                )
            ),
        )
        + (
            _tag(
                "p",
                {},
                _tag("b", {}, "Run note: ") + _esc(run_note),
            )
            if run_note else ""
        ),
    )

    completion_integrity = bundle.get("completion_integrity")
    completion_section = ""
    if completion_integrity:
        completion_rows = []
        integrity_arms = completion_integrity.get("arms") or {}
        for arm in bundle["arms"]:
            arm_id = arm["arm_id"]
            conditions = integrity_arms.get(arm_id) or {}
            for condition in ("cold", "warm"):
                if condition not in conditions:
                    continue
                completion_rows.append({
                    "arm_id": arm_id,
                    "condition": condition,
                    "item": conditions[condition],
                })

        def completion_count(row, name):
            value = (row["item"].get("measured") or {}).get(name)
            return "—" if value is None else str(value)

        def primer_coverage(row):
            if row["condition"] != "warm":
                return "—"
            primer = row["item"].get("warm_primer") or {}
            natural_stop = primer.get("natural_stop")
            total = primer.get("total")
            if natural_stop is None or total is None:
                return "—"
            return f"{natural_stop}/{total}"

        completion_columns = [
            {
                "label": "Route",
                "cls": "name",
                "type": "str",
                "dir": "asc",
                "cell": lambda row: _gateway_route_cell(row["arm_id"]),
                "key": lambda row: _gateway_probe_route_name(row["arm_id"]),
            },
            {
                "label": "Condition",
                "cell": lambda row: _esc(row["condition"].title()),
            },
            {
                "label": "Measured natural-stop",
                "cell": lambda row: completion_count(row, "natural_stop"),
            },
            {
                "label": "Measured length",
                "cell": lambda row: completion_count(row, "length"),
            },
            {
                "label": "Measured missing",
                "cell": lambda row: completion_count(row, "missing"),
            },
            {
                "label": "Measured other",
                "cell": lambda row: completion_count(row, "other"),
            },
            {
                "label": "Warm-primer natural-stop",
                "cell": primer_coverage,
            },
        ]
        completion_section = (
            _tag(
                "div",
                {"class": "head"},
                _tag("h2", {}, "Completion integrity")
                + _tag(
                    "p",
                    {},
                    "Provider-reported completion reasons. Natural stop means "
                    "the response ended normally; length means the provider "
                    "reported a length-based termination, commonly an output "
                    "or context limit; missing means no finish reason was "
                    "reported; other covers any remaining explicit reason. "
                    "Warm-primer natural stop shows whether the separate "
                    "connection-warming request ended normally.",
                ),
            )
            + _render_table(completion_columns, completion_rows)
        )

    def summary(item, name):
        return (item.get("metrics") or {}).get(name) or {}

    def coverage(value):
        value = value or {}
        covered = value.get("covered")
        total = value.get("total")
        if covered is None or total is None:
            return "—"
        return f"{covered}/{total}"

    def coverage_detail(value, *, compact=False):
        value = value or {}
        covered = value.get("covered")
        total = value.get("total")
        if covered is not None and total is not None and covered == total:
            return ""
        rendered = coverage(value)
        if compact:
            return " " + _tag("span", {"class": "sub"}, f"({rendered})")
        return _tag("div", {"class": "sub"}, "coverage " + rendered)

    def percentile_cell(item, name, fmt):
        value = summary(item, name)
        return (
            f"{fmt(value.get('p50'))} / {fmt(value.get('p95'))}"
            + coverage_detail(value.get("coverage"))
        )

    def compact_percentile(item, name, fmt):
        value = summary(item, name)
        return (
            f"{fmt(value.get('p50'))} / {fmt(value.get('p95'))}"
            + coverage_detail(
                value.get("coverage"),
                compact=True,
            )
        )

    def seconds(value):
        return "—" if value is None else f"{value:.3f}s"

    def decimal(value):
        return "—" if value is None else f"{value:.1f}"

    def tokens_cell(row):
        item = row["item"]
        return "<br>".join((
            "total " + compact_percentile(item, "total_tokens", decimal),
            "cached " + compact_percentile(item, "cached_input_tokens", decimal),
            "cache write "
            + compact_percentile(
                item, "cache_write_input_tokens", decimal
            ),
        ))

    request_columns = [
        {"label": "Route", "cls": "name", "type": "str", "dir": "asc",
         "cell": lambda row: _gateway_route_cell(row["arm_id"]),
         "key": lambda row: _gateway_probe_route_name(row["arm_id"])},
        {"label": "TTFT p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(
             row["item"],
             "request_to_semantic_ttft_s",
             seconds,
         ),
         "key": lambda row: summary(
             row["item"],
             "request_to_semantic_ttft_s",
         ).get("p50")},
        {"label": "Stream total p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(
             row["item"], "request_stream_total_s", seconds
         ),
         "key": lambda row: summary(
             row["item"], "request_stream_total_s"
         ).get("p50")},
        {"label": "Response headers p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(
             row["item"], "request_to_response_headers_s", seconds
         ),
         "key": lambda row: summary(
             row["item"], "request_to_response_headers_s"
         ).get("p50")},
        {"label": "First body byte p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(
             row["item"], "request_to_first_body_byte_s", seconds
         ),
         "key": lambda row: summary(
             row["item"], "request_to_first_body_byte_s"
         ).get("p50")},
        {"label": "Throughput tok/s p50 / p95", "dir": "desc",
         "cell": lambda row: percentile_cell(
             row["item"], "throughput_tokens_per_s", decimal
         ),
         "key": lambda row: summary(
             row["item"], "throughput_tokens_per_s"
         ).get("p50")},
        {"label": "Total / cached / cache-write tokens p50 / p95",
         "cell": tokens_cell},
    ]

    parts = head + _gateway_probe_leaderboard(bundle)
    for condition in ("cold", "warm"):
        rows = [
            {
                "arm_id": arm["arm_id"],
                "role": arm.get("role"),
                "condition": condition,
                "item": (arm.get("conditions") or {}).get(condition) or {},
            }
            for arm in bundle["arms"]
        ]
        parts += _tag(
            "div",
            {"class": "head"},
            _tag("h2", {}, f"{condition.title()} requests")
            + _tag(
                "p",
                {},
                f"Complete blocks: {complete.get(condition, 0)}/{scheduled}. "
                "TTFT begins when the measured request is sent; connection "
                "setup is reported separately.",
            ),
        )
        parts += _render_table(request_columns, rows)

    setup_columns = [
        {"label": "Route", "cls": "name", "type": "str", "dir": "asc",
         "cell": lambda row: _gateway_route_cell(row["arm_id"]),
         "key": lambda row: _gateway_probe_route_name(row["arm_id"])},
        {"label": "DNS p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(row["item"], "setup_dns_s", seconds),
         "key": lambda row: summary(row["item"], "setup_dns_s").get("p50")},
        {"label": "TCP p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(row["item"], "setup_tcp_s", seconds),
         "key": lambda row: summary(row["item"], "setup_tcp_s").get("p50")},
        {"label": "TLS p50 / p95", "dir": "asc",
         "cell": lambda row: percentile_cell(row["item"], "setup_tls_s", seconds),
         "key": lambda row: summary(row["item"], "setup_tls_s").get("p50")},
    ]
    setup_rows = [
        {
            "arm_id": arm["arm_id"],
            "item": (arm.get("conditions") or {}).get("cold") or {},
        }
        for arm in bundle["arms"]
    ]
    parts += _tag(
        "div",
        {"class": "head"},
        _tag("h2", {}, "Cold setup")
        + _tag("p", {}, "Connection setup phases for cold requests only."),
    )
    parts += _render_table(setup_columns, setup_rows)

    contrast_rows = []
    for contrast in bundle.get("contrasts") or []:
        for condition in ("cold", "warm"):
            metrics = (contrast.get("conditions") or {}).get(condition) or {}
            contrast_rows.append({
                "arm_id": contrast["arm_id"],
                "direct_arm": contrast.get("direct_arm"),
                "condition": condition,
                "headers": metrics.get("request_to_response_headers_s") or {},
                "semantic_ttft": metrics.get(
                    "request_to_semantic_ttft_s"
                ) or {},
            })

    def contrast_metric(value):
        interval = value.get("interval") or {}
        return {
            "estimate": value.get("median_gateway_minus_direct"),
            "low": interval.get("low"),
            "high": interval.get("high"),
        }

    if contrast_rows:
        headers_domain = max(
            _delta_domain(
                [{"value": contrast_metric(row["headers"])}
                 for row in contrast_rows],
                "value",
            ),
            1e-9,
        )
        semantic_domain = max(
            _delta_domain(
                [{"value": contrast_metric(row["semantic_ttft"])}
                 for row in contrast_rows],
                "value",
            ),
            1e-9,
        )

        def contrast_cell(row, name, domain):
            value = row[name]
            metric_value = contrast_metric(value)
            interval_text = ""
            if (
                metric_value["low"] is not None
                and metric_value["high"] is not None
            ):
                interval_text = (
                    "95% CI "
                    + _signed(
                        metric_value["low"], lambda item: f"{item:.3f}s"
                    )
                    + " to "
                    + _signed(
                        metric_value["high"], lambda item: f"{item:.3f}s"
                    )
                    + " · "
                )
            return (
                _delta_cell(
                    metric_value,
                    lambda item: f"{item:.3f}s",
                    False,
                    domain,
                )
                + _tag(
                    "div",
                    {"class": "sub"},
                    interval_text
                    + "paired "
                    + coverage(value.get("coverage")),
                )
            )

        contrast_columns = [
            {"label": "Gateway route", "cls": "name", "type": "str", "dir": "asc",
             "cell": lambda row: _gateway_route_cell(row["arm_id"]),
             "key": lambda row: _gateway_probe_route_name(row["arm_id"])},
            {"label": "Condition", "type": "str", "dir": "asc",
             "cell": lambda row: _esc(row["condition"]),
             "key": lambda row: row["condition"]},
            {"label": "Δ response headers", "plot": True,
             "cell": lambda row: contrast_cell(row, "headers", headers_domain),
             "key": lambda row: row["headers"].get(
                 "median_gateway_minus_direct"
             )},
            {"label": "Δ TTFT", "plot": True,
             "cell": lambda row: contrast_cell(
                 row, "semantic_ttft", semantic_domain
             ),
             "key": lambda row: row["semantic_ttft"].get(
                 "median_gateway_minus_direct"
             )},
        ]
        parts += _tag(
            "div",
            {"class": "head"},
            _tag("h2", {}, "Paired request deltas")
            + _tag(
                "p",
                {},
                "Every delta is gateway minus "
                + _esc(baseline_name)
                + ". These are latency "
                "metrics, so positive means slower/worse and negative means "
                "faster/better. Medians use complete paired blocks with "
                "bootstrap 95% intervals.",
            ),
        )
        parts += _render_table(contrast_columns, contrast_rows)

    parts += completion_section

    links = "".join(filter(None, [
        _tag("span", {}, _link(bundle["results_path"], "results.jsonl"))
        if bundle.get("results_path") else None,
        _tag("span", {}, _link(bundle["path"], "release page"))
        if bundle.get("path") else None,
    ]))
    if links:
        parts += _tag(
            "div",
            {"class": "head"},
            _tag("div", {"class": "meta"}, links),
        )
    return _tag("section", {"class": "board", "data-caveats": "0"}, parts)


def _skipped_board(title, blurb, entries):
    items = "".join(
        _tag("li", {},
             _tag("strong", {}, _esc(e["id"]))
             + _tag("div", {"class": "sub"}, _esc(e["reason"])))
        for e in entries)
    return _tag("section", {"class": "board", "data-caveats": "0"},
                _tag("div", {"class": "head"},
                     _tag("h2", {}, _esc(title)) + _tag("p", {}, _esc(blurb)))
                + _tag("ul", {"class": "records"}, items))


def _records_section(title, blurb, items, anchor=None):
    if not items:
        return ""
    return _tag("section", {"class": "board", "id": anchor},
                _tag("div", {"class": "head"},
                     _tag("h2", {}, _esc(title)) + _tag("p", {}, _esc(blurb)))
                + _tag("div", {"class": "head"},
                       _tag("ul", {"class": "records"}, "".join(items))))


def _record(name_html, meta_parts, sub=None, extra=""):
    body = name_html
    meta = "  ·  ".join(str(p) for p in meta_parts if p)
    if meta:
        body += _tag("div", {"class": "sub"}, _esc(meta))
    if sub:
        body += _tag("div", {"class": "sub"}, _esc(sub))
    return _tag("li", {}, body + extra)


def _linked_title(entry):
    name = _esc(entry.get("title") or entry.get("id") or "")
    if entry.get("path"):
        linked = _link(entry["path"], name)
        if linked != name:
            return linked
    return _tag("strong", {}, name)


def _releases_section(entries):
    return _records_section(
        "Releases", "First-party bundles.",
        [_record(_linked_title(e),
                 [e.get("date"), ", ".join(e.get("models") or [])])
         for e in entries])


def _community_section(entries):
    items = []
    for entry in entries:
        extra = ""
        if entry.get("link"):
            extra = _tag("div", {"class": "sub"},
                         _link(entry["link"], "source",
                               rel="nofollow noopener"))
        items.append(_record(
            _linked_title(entry),
            [entry.get("date"),
             "@" + entry["submitter"] if entry.get("submitter") else None],
            sub=entry.get("claim") or entry.get("description"),
            extra=extra))
    return _records_section(
        "Community",
        "Third-party bundles, re-verified by CI. Digests show tamper-evidence, "
        "not absence of cherry-picking.",
        items, anchor="community")


def _packs_section(entries):
    items = []
    for entry in entries:
        name = entry.get("id") or ""
        if entry.get("latest"):
            name = f"{name}@{entry['latest']}"
        head = _tag("strong", {}, _esc(name))
        if entry.get("kind"):
            head += " " + _chip(entry["kind"])
        items.append(_record(
            head,
            [entry.get("license"), entry.get("source"),
             (entry.get("content_sha256") or "")[:12] or None],
            sub=entry.get("description")))
    return _records_section(
        "Packs",
        "Versioned task and harness packs.",
        items, anchor="packs")


def _controls(doc):
    models, harnesses = set(), set()
    for bundle in doc["harness"]["bundles"]:
        for arm in bundle["arms"]:
            models.add(arm["model"])
            harnesses.add(arm["harness"])

    def select(control_id, label, values):
        options = _tag("option", {"value": ""}, _esc(label))
        options += "".join(_tag("option", {"value": v}, _esc(v)) for v in sorted(values))
        return _tag("select", {"id": control_id, "aria-label": label}, options)

    body = _tag("input", {
        "type": "search", "id": "q", "placeholder": "Filter by harness or model…",
        "aria-label": "Filter by harness or model",
    })
    if models:
        body += select("f-model", "All models", models)
    if harnesses:
        body += select("f-harness", "All harnesses", harnesses)
    body += _tag("label", {},
                 _tag("input", {"type": "checkbox", "id": "f-caveats"})
                 + "Hide boards with disclosed caveats")
    return _tag("div", {"class": "controls", "id": "controls"}, body)


def _harness_view(doc):
    family = doc["harness"]
    body = _controls(doc)
    body += _tag("p", {"class": "note"}, _esc(family["note"]))
    body += "".join(_harness_board(b) for b in family["bundles"])
    body += _tag("section", {"class": "board", "id": "no-matches", "hidden": True},
                 _tag("div", {"class": "empty"},
                      _tag("p", {}, "No boards match the current filters.")))
    if family.get("skipped"):
        body += _skipped_board(
            f"Not ranked ({len(family['skipped'])})",
            "No result-sealed results.jsonl.",
            family["skipped"])
    return body


def _gateway_probe_view(doc):
    family = doc["gateway"]
    body = _tag("p", {"class": "note"}, _esc(family["note"]))
    if not family["bundles"]:
        body += _tag("section", {"class": "board"}, _tag(
            "div", {"class": "empty"},
            _tag("p", {}, "No verified Gateway Bench bundles published yet.")))
    else:
        tabs = []
        panels = []
        for index, bundle in enumerate(family["bundles"]):
            suffix = re.sub(r"[^a-z0-9_-]+", "-", bundle["id"].lower()).strip("-")
            tab_id = f"gateway-model-tab-{suffix}"
            panel_id = f"gateway-model-panel-{suffix}"
            complete = bundle.get("complete_blocks") or {}
            tabs.append(_tag(
                "button",
                {
                    "type": "button",
                    "role": "tab",
                    "id": tab_id,
                    "aria-controls": panel_id,
                    "aria-selected": "true" if index == 0 else "false",
                    "tabindex": "0" if index == 0 else "-1",
                    "data-gateway-model-tab": "",
                },
                _esc(bundle.get("model") or bundle["title"])
                + _tag(
                    "span",
                    {"class": "tab-depth"},
                    _esc(
                        f"{complete.get('cold', 0)} cold + "
                        f"{complete.get('warm', 0)} warm matched blocks"
                    ),
                ),
            ))
            panels.append(_tag(
                "section",
                {
                    "role": "tabpanel",
                    "id": panel_id,
                    "aria-labelledby": tab_id,
                    "data-gateway-model-panel": "",
                },
                _gateway_probe_board(bundle),
            ))
        body += _tag(
            "div",
            {
                "class": "gateway-model-tabs",
                "role": "tablist",
                "aria-label": "Benchmark model",
            },
            "".join(tabs),
        )
        body += "".join(panels)
    if family.get("skipped"):
        body += _skipped_board(
            f"Not published ({len(family['skipped'])})",
            "Did not pass public Gateway Bench verification.",
            family["skipped"],
        )
    return body


def _releases_view(doc):
    return (_releases_section(doc["releases"])
            + _community_section(doc["community"])
            + _packs_section(doc["packs"]))


def _contact_view():
    return _tag(
        "section",
        {"class": "prose"},
        _tag("h2", {}, "Contact")
        + _tag(
            "p",
            {},
            "Want to add a gateway or harness, submit results, report a "
            "problem, or share an idea? Reach Matthew through either channel.",
        )
        + _tag(
            "div",
            {"class": "contact-actions"},
            _link(
                "https://github.com/minghinmatthewlam/openbench/issues/new",
                "Open a GitHub issue",
            )
            + _link("https://x.com/mattlam_", "Message @mattlam_ on X"),
        ),
    )


def render_board_html(doc):
    """The whole page: content rendered here, behaviour layered on top."""
    def render_lede(family):
        title, deck, facts = _lede(doc, family)
        return _tag(
            "div",
            {"class": "lede", "data-lede": family},
            _tag("h1", {}, _esc(title))
            + _tag("p", {"class": "deck"}, _esc(deck))
            + _tag(
                "div",
                {"class": "dateline"},
                "".join(_tag("span", {}, _esc(f)) for f in facts),
            ),
        )

    tabs = "".join(
        _tag("a", {"href": "#" + slug}, _esc(label))
        for slug, label in (
            ("harness", "Harness Bench"),
            ("gateway", "Gateway Bench"),
            ("releases", "Releases"),
            ("methodology", "Methodology"),
            ("contact", "Contact"),
        )
    )
    theme_button = _tag("button", {
        "class": "theme", "id": "theme", "type": "button",
        "aria-label": "Toggle colour theme", "title": "Toggle colour theme",
    }, '<svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor">'
       '<path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm0 1.5V14.5a6.5 6.5 0 0 1 0-13z"/>'
       "</svg>")

    masthead = _tag("header", {"class": "top"}, _tag(
        "div", {"class": "wrap"},
        _tag("div", {"class": "brand"},
             _tag("span", {"class": "name"}, "OpenBench")
             + _tag("span", {"class": "what"}, "leaderboards"))
        + _tag("nav", {"class": "tabs"}, tabs)
        + theme_button))

    # Every view ships expanded. The script collapses them into tabs; without
    # it the nav degrades to jump links over one continuous page.
    views = (
        _tag(
            "main",
            {"id": "view-harness"},
            render_lede("harness") + _harness_view(doc),
        )
        + _tag(
            "main",
            {"id": "view-gateway"},
            render_lede("gateway") + _gateway_probe_view(doc),
        )
        + _tag(
            "main",
            {"id": "view-releases"},
            render_lede("general") + _releases_view(doc),
        )
        + _tag(
            "main",
            {"id": "view-methodology"},
            render_lede("general") + _METHODOLOGY,
        )
        + _tag(
            "main",
            {"id": "view-contact"},
            render_lede("general") + _contact_view(),
        )
    )

    footer = _tag("footer", {},
                  "Generated by " + _tag("code", {}, "obench site")
                  + " &middot; static, self-contained, no third-party assets "
                  "&middot; " + _tag("a", {"href": "board.json"}, "board.json"))

    site_metadata = doc.get("site_metadata") or {}
    canonical_url = _safe_href(site_metadata.get("canonical_url"))
    social_image_url = _safe_href(site_metadata.get("social_image_url"))
    social_metadata = ""
    if canonical_url and canonical_url.startswith(("http://", "https://")):
        social_metadata += (
            f'<link rel="canonical" href="{_esc(canonical_url)}">'
            f'<meta property="og:url" content="{_esc(canonical_url)}">'
        )
    if social_image_url and social_image_url.startswith(("http://", "https://")):
        social_metadata += (
            f'<meta property="og:image" content="{_esc(social_image_url)}">'
            '<meta property="og:image:width" content="1200">'
            '<meta property="og:image:height" content="630">'
            f'<meta name="twitter:image" content="{_esc(social_image_url)}">'
        )
    social_metadata += (
        '<meta property="og:type" content="website">'
        '<meta property="og:title" content="OpenBench gateway benchmarks">'
        '<meta property="og:description" content="Digest-verified AI '
        'gateway benchmark results.">'
        '<meta name="twitter:card" content="summary_large_image">'
    )

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>OpenBench leaderboards</title>"
        '<meta name="description" content="Harness and serving-route leaderboards '
        'for OpenBench, built from digest-verified result bundles.">'
        + social_metadata
        + f"<style>{_CSS}</style></head><body>"
        + masthead
        + '<div class="wrap">'
        + views
        + footer
        + "</div>"
        + f"<script>{_JS}</script>"
        + "</body></html>\n"
    )


def write_board(
    site_dir,
    community_dir=None,
    gateway_dirs=None,
    gateway_probe_dirs=None,
):
    """Build and write ``index.html`` + ``board.json`` under ``site_dir``."""
    site_dir = os.path.abspath(site_dir)
    doc = build_board(
        site_dir,
        community_dir=community_dir,
        gateway_dirs=gateway_dirs,
        gateway_probe_dirs=gateway_probe_dirs,
    )
    json_path = os.path.join(site_dir, "board.json")
    html_path = os.path.join(site_dir, "index.html")
    outputs = (
        (json_path, json.dumps(
            doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n"),
        (html_path, render_board_html(doc)),
    )
    originals = {}
    pending = []
    for path, _ in outputs:
        try:
            with open(path, "rb") as fh:
                originals[path] = fh.read()
        except FileNotFoundError:
            originals[path] = None
    try:
        for path, content in outputs:
            pending.append((report_page._temporary_text(path, content), path))
        for temporary, path in pending:
            os.replace(temporary, path)
    except BaseException:
        for path, _ in reversed(outputs):
            original = originals[path]
            if original is None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            else:
                temporary = report_page._temporary_text(
                    path, original.decode("utf-8"))
                os.replace(temporary, path)
        raise
    finally:
        for temporary, _ in pending:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {
        "json_path": json_path,
        "html_path": html_path,
        "harness_bundles": doc["harness"]["bundle_count"],
        "gateway_bundles": doc["gateway"]["bundle_count"],
        "skipped": (
            len(doc["harness"]["skipped"])
            + len(doc["gateway"]["skipped"])
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="obench site",
        description="Build the unified static site (harness + gateway).",
    )
    sub = parser.add_subparsers(dest="command")
    build = sub.add_parser("build", help="write index.html + board.json")
    build.add_argument(
        "--site-dir",
        default=leaderboard._default_site_dir(),
        help="GitHub Pages root (default: docs/)",
    )
    build.add_argument(
        "--community-dir",
        default=None,
        help="optional data/community root to include (default: auto when present)",
    )
    build.add_argument(
        "--no-community-dir",
        action="store_true",
        help="do not scan data/community",
    )
    build.add_argument(
        "--gateway-dir",
        action="append",
        default=None,
        help=argparse.SUPPRESS,
    )
    build.add_argument(
        "--gateway-probe-dir",
        action="append",
        default=None,
        help=(
            "Gateway Bench public bundle root "
            "(repeatable; default: <site-dir>/gateway-probe)"
        ),
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "build":
        community_dir = args.community_dir
        if community_dir is None and not args.no_community_dir:
            community_dir = leaderboard._default_community_dir()
        if args.no_community_dir:
            community_dir = None
        info = write_board(
            args.site_dir,
            community_dir=community_dir,
            gateway_dirs=args.gateway_dir,
            gateway_probe_dirs=args.gateway_probe_dir,
        )
        print(f"index.html  {info['html_path']}")
        print(f"board.json  {info['json_path']}")
        print(
            f"harness_bundles={info['harness_bundles']} "
            f"gateway_bundles={info['gateway_bundles']} "
            f"skipped={info['skipped']}"
        )
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
