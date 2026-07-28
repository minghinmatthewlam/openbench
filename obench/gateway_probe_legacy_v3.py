"""Admission for the single immutable public Gateway Bench schema-v3 archive."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import gateway_probe_publish, gateway_probe_results, gateway_spec
from .gateway_probe_models import GatewayProbeRunError


_VERIFIED_COMMIT = "6d1de84d6c96430c47b50a51bd802a986db5c2ba"
_APPROVED_BUNDLE_ID = "2026-07-27-gpt4o-mini-managed-30"
_APPROVED_MANIFEST_SHA256 = (
    "5eb6c9c4b03404dda035cd116e0bec55a08bf5beaea0d3ebbc2e7fa233d4f487"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GatewayProbeRunError(f"invalid legacy public {label}") from exc


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise GatewayProbeRunError(
                        "legacy public results contain a blank line"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise GatewayProbeRunError(
                        f"legacy public result line {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GatewayProbeRunError("invalid legacy public results") from exc
    return rows


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify the pinned v3 bundle without weakening the current v4 verifier."""
    bundle_input = Path(bundle_dir)
    if bundle_input.is_symlink():
        raise GatewayProbeRunError(
            "legacy public Gateway Bench bundle must not be a symlink"
        )
    directory = bundle_input.resolve()
    if not directory.is_dir():
        raise GatewayProbeRunError("legacy public Gateway Bench bundle is missing")
    if directory.name != _APPROVED_BUNDLE_ID:
        raise GatewayProbeRunError("legacy public bundle identity is not approved")
    names = {path.name for path in directory.iterdir()}
    if names != gateway_probe_publish._PUBLIC_DIRECTORY_FILES:
        raise GatewayProbeRunError(
            "legacy public Gateway Bench bundle has unexpected files"
        )

    manifest_path = directory / "manifest.json"
    if _sha256(manifest_path) != _APPROVED_MANIFEST_SHA256:
        raise GatewayProbeRunError("legacy public manifest pin mismatch")
    manifest = _load_json(manifest_path, "manifest")
    verification = manifest.get("verification") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != gateway_probe_publish.PUBLIC_SCHEMA_VERSION
        or manifest.get("bundle_kind")
        != gateway_probe_publish.PUBLIC_BUNDLE_KIND
        or manifest.get("benchmark") != gateway_probe_results.BENCHMARK
        or manifest.get("result_schema_version") != 3
        or verification != {"verified_with_commit": _VERIFIED_COMMIT}
        or set(manifest.get("files") or {}) != set(gateway_probe_publish.PUBLIC_FILES)
    ):
        raise GatewayProbeRunError("legacy public manifest does not match schema v3")

    for name in gateway_probe_publish.PUBLIC_FILES:
        file_record = manifest["files"].get(name)
        if (
            not isinstance(file_record, dict)
            or set(file_record) != {"sha256"}
            or _SHA256_RE.fullmatch(file_record.get("sha256", "")) is None
            or _sha256(directory / name) != file_record["sha256"]
        ):
            raise GatewayProbeRunError(
                f"legacy public artifact digest mismatch: {name}"
            )

    experiment = _load_json(directory / "experiment.json", "experiment")
    report = _load_json(directory / "report.json", "report")
    prices = _load_json(directory / "prices.json", "prices")
    schedule = _load_json(directory / "schedule.json", "schedule")
    rows = _load_rows(directory / "results.jsonl")
    report_text = (directory / "report.md").read_text(encoding="utf-8")
    for label, value in (
        ("experiment", experiment),
        ("report", report),
        ("prices", prices),
        ("schedule", schedule),
        ("results", rows),
        ("report text", report_text),
    ):
        gateway_probe_publish._assert_public_safe(value, f"legacy public {label}")

    if (
        experiment.get("schema_version") != 1
        or experiment.get("benchmark") != gateway_probe_results.BENCHMARK
        or experiment.get("experiment_id") != manifest.get("experiment_id")
        or experiment.get("experiment_digest") != manifest.get("experiment_digest")
        or report.get("schema_version") != manifest.get("report_schema_version")
        or report.get("experiment_id") != manifest.get("experiment_id")
        or report.get("experiment_digest") != manifest.get("experiment_digest")
        or report.get("schedule_digest") != manifest.get("schedule_digest")
        or report.get("price_digest") != manifest.get("price_digest")
        or report.get("complete_blocks") != manifest.get("complete_blocks")
        or report.get("scheduled_blocks_per_condition")
        != manifest.get("scheduled_blocks_per_condition")
        or gateway_spec.canonical_digest(schedule.get("blocks"))
        != manifest.get("schedule_digest")
        or gateway_spec.canonical_digest(
            gateway_probe_publish._project_prices(prices)
        )
        != manifest.get("price_digest")
        or len(rows) != manifest.get("result_count")
    ):
        raise GatewayProbeRunError(
            "legacy public evidence does not match its pinned manifest"
        )

    seen_cells = set()
    for row in rows:
        identity = row.get("identity")
        row_experiment = identity.get("experiment") if isinstance(identity, dict) else None
        comparison = identity.get("comparison") if isinstance(identity, dict) else None
        cell_id = row.get("cell_id")
        if (
            row.get("schema_version") != 3
            or row.get("benchmark") != gateway_probe_results.BENCHMARK
            or not isinstance(cell_id, str)
            or not cell_id.startswith("gateway-probe-cell-v3-")
            or cell_id in seen_cells
            or row_experiment
            != {
                "id": manifest["experiment_id"],
                "digest": manifest["experiment_digest"],
            }
            or comparison
            != {
                "schedule_digest": manifest["schedule_digest"],
                "price_digest": manifest["price_digest"],
            }
        ):
            raise GatewayProbeRunError(
                "legacy public results do not match frozen schema v3"
            )
        seen_cells.add(cell_id)
    return manifest
