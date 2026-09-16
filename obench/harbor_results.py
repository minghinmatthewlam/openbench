#!/usr/bin/env python3
"""Fail-closed Harbor 0.20.0 job-result ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .atif import validate_trajectory
from .harbor_metering import (
    HarborMeteringError,
    UsageCounters,
    apply_to_imported_row,
    verify_evidence_dir,
)
from .harbor_job import (
    COMPARISON_PLAN_SCHEMA_VERSION,
    canonical_agent_config_sha256,
    canonical_comparison_plan_bytes,
    canonical_harbor_job_config_sha256,
)
from .harbor_oauth import AGENT_IMPORT_PATH
from .harbor_profiles import (
    CODEX_PROFILE_IMPORT,
    CURSOR_PROFILE_IMPORT,
    DEVIN_PROFILE_IMPORT,
    OPENCODE_PROFILE_IMPORT,
    PI_PROFILE_IMPORT,
    HarborProfileError,
    canonical_openbench_model,
)
from .run import (
    ROW_FIELDS,
    ResultsLogError,
    make_run_id,
    results_file_lock,
)
from .usage_evidence import harbor_usage_policy

HARBOR_VERSION = "0.20.0"
HARBOR_GIT_COMMIT = "72bc40b1e58b47a9cc6e0f14c29aced3a9e53767"
JOB_LOCK_SCHEMA_VERSION = 3
TRIAL_LOCK_SCHEMA_VERSION = 2
FINAL_WORKSPACE_DESTINATION = "workspace"
FINAL_WORKSPACE_SOURCE = "/app"
FINAL_WORKSPACE_MANIFEST_DESTINATION = "artifacts/workspace"
VERIFIER_EVIDENCE_SCHEMA_VERSION = "openbench-verifier-evidence-v2"
MAX_JSON_BYTES = 32 * 1024 * 1024
HARBOR_AGENT_SEMANTIC_NAME_ALIASES = {
    AGENT_IMPORT_PATH: "codex",
    CODEX_PROFILE_IMPORT: "codex",
    PI_PROFILE_IMPORT: "pi",
    OPENCODE_PROFILE_IMPORT: "opencode",
    CURSOR_PROFILE_IMPORT: "cursor",
    DEVIN_PROFILE_IMPORT: "devin",
}
HARBOR_STOCK_AGENT_CONFIG_NAMES = frozenset(
    {
        "codex",
        "pi",
        "opencode",
        "cursor",
        "devin",
        *HARBOR_AGENT_SEMANTIC_NAME_ALIASES,
    }
)
HARBOR_PROXY_REQUIRED_AGENTS = frozenset({
    CODEX_PROFILE_IMPORT,
    PI_PROFILE_IMPORT,
})
HARBOR_TIMEOUT_EXCEPTIONS = frozenset({
    "AgentSetupTimeoutError",
    "AgentTimeoutError",
    "EnvironmentStartTimeoutError",
    "TimeoutError",
    "VerifierTimeoutError",
})
HARBOR_RATE_LIMIT_EXCEPTIONS = frozenset({"ApiUsageLimitError"})
HARBOR_AUTH_EXCEPTIONS = frozenset({"AgentAuthenticationError"})
HARBOR_SAFETY_EXCEPTIONS = frozenset({"AgentSafetyRefusalError"})


class HarborResultsError(ValueError):
    """Raised when Harbor evidence cannot support normalized result rows."""


def _fail(location: str, message: str) -> HarborResultsError:
    return HarborResultsError(f"{location}: {message}")


def expected_harbor_agent_semantic_name(config_name: str) -> str:
    """Resolve a pinned Harbor config identity to its reported agent name."""
    return HARBOR_AGENT_SEMANTIC_NAME_ALIASES.get(config_name, config_name)


def harbor_exception_semantics(exception_type: str) -> tuple[str, str]:
    """Map a Harbor terminal exception to OpenBench class and stable reason."""
    if exception_type in HARBOR_TIMEOUT_EXCEPTIONS:
        return "timeout", f"harbor_timeout:{exception_type}"
    if exception_type in HARBOR_RATE_LIMIT_EXCEPTIONS:
        return "rate_limited", f"harbor_rate_limit:{exception_type}"
    if exception_type in HARBOR_AUTH_EXCEPTIONS:
        return "infra", f"harbor_auth:{exception_type}"
    if exception_type in HARBOR_SAFETY_EXCEPTIONS:
        return "infra", f"harbor_safety:{exception_type}"
    return "infra", f"harbor_infrastructure:{exception_type}"


def _agent_config_identity(agent: dict[str, Any], location: str) -> str:
    """Return Harbor's immutable built-in name or custom-agent import path."""

    name = agent.get("name")
    import_path = agent.get("import_path")
    if isinstance(name, str) and name:
        return name
    if name is not None:
        raise _fail(f"{location}.name", "expected a non-empty string or null")
    return _string(import_path, f"{location}.import_path")


def _agent_config_match_key(agent: dict[str, Any]) -> str:
    """Normalize Harbor's omitted AgentConfig defaults for exact matching."""

    normalized = _normalized_mapping(
        agent,
        _AGENT_DEFAULTS,
        exclude={"skills"},
    )
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _require_regular_file(path: Path, location: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise _fail(location, f"required regular file is missing: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _fail(location, f"cannot stat {path}: {exc}") from exc
    if size <= 0:
        raise _fail(location, f"required file is empty: {path}")
    if size > MAX_JSON_BYTES:
        raise _fail(location, f"file exceeds {MAX_JSON_BYTES} bytes: {path}")


def _read_json(path: Path, location: str) -> Any:
    _require_regular_file(path, location)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _fail(location, f"invalid JSON in {path}: {exc}") from exc
    return value


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(location, "expected a JSON object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail(location, "expected a JSON array")
    return value


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(location, "expected a non-empty string")
    return value


def _integer(value: Any, location: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _fail(location, f"expected an integer >= {minimum}")
    return value


def _number(
    value: Any,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise _fail(location, "expected a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise _fail(location, f"expected a number >= {minimum}")
    if maximum is not None and result > maximum:
        raise _fail(location, f"expected a number <= {maximum}")
    return result


def _timestamp(value: Any, location: str) -> datetime:
    text = _string(value, location)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail(location, "expected an ISO 8601 datetime") from exc
    if parsed.tzinfo is None:
        raise _fail(location, "datetime must include a UTC offset")
    return parsed


def _optional_timestamp(value: Any, location: str) -> datetime | None:
    return None if value is None else _timestamp(value, location)


def _job_timestamp(value: Any, location: str) -> datetime:
    text = _string(value, location)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail(location, "expected an ISO 8601 datetime") from exc


def _optional_job_timestamp(value: Any, location: str) -> datetime | None:
    return None if value is None else _job_timestamp(value, location)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256_file(path: Path | None) -> str | None:
    return None if path is None else _sha256_file(path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_digest(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise _fail("final workspace", f"expected a real directory: {root}")
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    if not entries:
        raise _fail("final workspace", f"directory is empty: {root}")
    digest = hashlib.sha256()
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise _fail("final workspace", f"symlinks are not accepted: {path}")
        if path.is_dir():
            digest.update(f"d\0{relative}\0".encode())
        elif path.is_file():
            file_digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    file_digest.update(chunk)
            digest.update(
                f"f\0{relative}\0{size}\0{file_digest.hexdigest()}\0".encode()
            )
        else:
            raise _fail("final workspace", f"unsupported filesystem entry: {path}")
    return digest.hexdigest()


def _canonical_lock(lock: dict[str, Any]) -> str:
    return json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_digest(value: Any, location: str) -> str:
    digest = _string(value, location)
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(char not in "0123456789abcdef" for char in digest[7:])
    ):
        raise _fail(location, "expected sha256:<64 lowercase hex characters>")
    return digest


def _validate_openbench_task_content_digest(
    value: Any, location: str
) -> dict[str, Any]:
    digest = _object(value, location)
    if set(digest) != {"scheme", "sha256"}:
        raise _fail(location, "expected exactly 'scheme' and 'sha256'")
    if digest.get("scheme") != 2 or isinstance(digest.get("scheme"), bool):
        raise _fail(f"{location}.scheme", "expected OpenBench digest scheme 2")
    sha256 = digest.get("sha256")
    if (
        not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise _fail(
            f"{location}.sha256",
            "expected 64 lowercase hex characters",
        )
    return {"scheme": 2, "sha256": sha256}


def _validate_openbench_harbor_export(
    value: Any, location: str
) -> dict[str, Any]:
    config = _object(value, location)
    if set(config) != {"schema_version", "base_image", "network_mode"}:
        raise _fail(
            location,
            "expected exactly 'schema_version', 'base_image', and 'network_mode'",
        )
    if config.get("schema_version") != 1 or isinstance(
        config.get("schema_version"), bool
    ):
        raise _fail(f"{location}.schema_version", "expected 1")
    base_image = _string(config.get("base_image"), f"{location}.base_image")
    network_mode = _string(
        config.get("network_mode"),
        f"{location}.network_mode",
    )
    return {
        "schema_version": 1,
        "base_image": base_image,
        "network_mode": network_mode,
    }


def _validate_job_lock(job_lock: Any) -> tuple[dict[str, Any], Counter[str]]:
    lock = _object(job_lock, "job lock")
    if lock.get("schema_version") != JOB_LOCK_SCHEMA_VERSION:
        raise _fail(
            "job lock.schema_version",
            f"expected {JOB_LOCK_SCHEMA_VERSION} for Harbor {HARBOR_VERSION}",
        )
    _timestamp(lock.get("created_at"), "job lock.created_at")
    _integer(lock.get("n_concurrent_trials"), "job lock.n_concurrent_trials", minimum=1)
    _object(lock.get("retry"), "job lock.retry")

    harbor = _object(lock.get("harbor"), "job lock.harbor")
    if harbor.get("version") != HARBOR_VERSION:
        raise _fail(
            "job lock.harbor.version",
            f"expected resolved Harbor version {HARBOR_VERSION!r}",
        )
    if harbor.get("git_commit_hash") != HARBOR_GIT_COMMIT:
        raise _fail(
            "job lock.harbor.git_commit_hash",
            f"expected resolved Harbor commit {HARBOR_GIT_COMMIT}",
        )
    if not isinstance(harbor.get("is_editable"), bool):
        raise _fail("job lock.harbor.is_editable", "expected a boolean")
    if harbor["is_editable"]:
        raise _fail(
            "job lock.harbor.is_editable",
            "editable Harbor installations are not publication-grade",
        )

    trials = _array(lock.get("trials"), "job lock.trials")
    if not trials:
        raise _fail("job lock.trials", "must contain at least one resolved trial")
    canonical: Counter[str] = Counter()
    for index, trial_lock in enumerate(trials):
        validated = _validate_trial_lock(
            trial_lock, f"job lock.trials[{index}]"
        )
        canonical[_canonical_lock(validated)] += 1
    return lock, canonical


def _validate_trial_lock(value: Any, location: str) -> dict[str, Any]:
    lock = _object(value, location)
    if lock.get("schema_version") != TRIAL_LOCK_SCHEMA_VERSION:
        raise _fail(
            f"{location}.schema_version",
            f"expected {TRIAL_LOCK_SCHEMA_VERSION}",
        )
    task = _object(lock.get("task"), f"{location}.task")
    _string(task.get("name"), f"{location}.task.name")
    if task.get("type") not in {"local", "git", "package"}:
        raise _fail(f"{location}.task.type", "expected local, git, or package")
    _validate_digest(task.get("digest"), f"{location}.task.digest")
    agent = _object(lock.get("agent"), f"{location}.agent")
    _agent_config_identity(agent, f"{location}.agent")
    _string(agent.get("model_name"), f"{location}.agent.model_name")
    _object(lock.get("environment"), f"{location}.environment")
    _object(lock.get("verifier"), f"{location}.verifier")
    return lock


_AGENT_DEFAULTS = {
    "name": None,
    "import_path": None,
    "n_concurrent": None,
    "concurrency_group": None,
    "override_timeout_sec": None,
    "override_setup_timeout_sec": None,
    "max_timeout_sec": None,
    "resume_trajectory": False,
    "load_trajectory": None,
    "extra_allowed_hosts": [],
    "include_logs": [],
    "exclude_logs": [],
    "kwargs": {},
    "env": {},
    "mcp_servers": [],
}
_ENVIRONMENT_DEFAULTS = {
    # Pinned Harbor omits this null selector from custom-environment locks,
    # but includes it in TrialResult.config. Explicit selectors still compare.
    "type": None,
    "import_path": None,
    "force_build": False,
    "delete": True,
    "cpu_enforcement_policy": "auto",
    "memory_enforcement_policy": "auto",
    "override_cpus": None,
    "override_memory_mb": None,
    "override_storage_mb": None,
    "override_gpus": None,
    "override_tpu": None,
    "mounts": None,
    "env": {},
    "kwargs": {},
    "extra_allowed_hosts": [],
}
_VERIFIER_DEFAULTS = {
    "override_timeout_sec": None,
    "max_timeout_sec": None,
    "include_logs": [],
    "exclude_logs": [],
    "env": {},
    "import_path": None,
    "kwargs": {},
    "disable": False,
}


def _normalized_mapping(
    value: dict[str, Any],
    defaults: dict[str, Any],
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    normalized = {
        key: json.loads(json.dumps(default))
        for key, default in defaults.items()
    }
    normalized.update(value)
    for key in exclude or set():
        normalized.pop(key, None)
    return normalized


def _validate_lock_backed_config(
    result: dict[str, Any],
    trial_lock: dict[str, Any],
    *,
    expected_job_id: str,
    location: str,
) -> None:
    config = _object(result.get("config"), f"{location}.result.config")
    scalar_defaults = {
        "install_only": False,
        "timeout_multiplier": 1.0,
        "agent_timeout_multiplier": None,
        "verifier_timeout_multiplier": None,
        "agent_setup_timeout_multiplier": None,
        "environment_build_timeout_multiplier": None,
    }
    for field, default in scalar_defaults.items():
        if config.get(field, default) != trial_lock.get(field, default):
            raise _fail(
                f"{location}.result.config.{field}",
                "does not match trial lock",
            )
    config_job_id = config.get("job_id")
    if config_job_id is not None and config_job_id != expected_job_id:
        raise _fail(
            f"{location}.result.config.job_id",
            "does not match top-level job id",
        )

    config_task = _object(config.get("task"), f"{location}.result.config.task")
    lock_task = _object(trial_lock.get("task"), f"{location}.lock.task")
    task_type = lock_task.get("type")
    if config_task.get("source") != lock_task.get("source"):
        raise _fail(
            f"{location}.result.config.task.source",
            "does not match trial lock",
        )
    if task_type == "local":
        if config_task.get("path") != lock_task.get("path"):
            raise _fail(
                f"{location}.result.config.task.path",
                "does not match local trial lock",
            )
        if any(
            config_task.get(field) is not None
            for field in ("name", "ref", "git_url", "git_commit_id")
        ):
            raise _fail(f"{location}.result.config.task", "contradicts local task lock")
    elif task_type == "git":
        for field in ("path", "git_url"):
            if config_task.get(field) != lock_task.get(field):
                raise _fail(
                    f"{location}.result.config.task.{field}",
                    "does not match git trial lock",
                )
        configured_commit = config_task.get("git_commit_id")
        if (
            isinstance(configured_commit, str)
            and len(configured_commit) == 40
            and configured_commit != lock_task.get("git_commit_id")
        ):
            raise _fail(
                f"{location}.result.config.task.git_commit_id",
                "contradicts resolved git trial lock",
            )
        if config_task.get("name") is not None or config_task.get("ref") is not None:
            raise _fail(f"{location}.result.config.task", "contradicts git task lock")
    elif task_type == "package":
        if config_task.get("name") != lock_task.get("name"):
            raise _fail(
                f"{location}.result.config.task.name",
                "does not match package trial lock",
            )
        ref = config_task.get("ref")
        if isinstance(ref, str) and ref.startswith("sha256:") and ref != lock_task.get("digest"):
            raise _fail(
                f"{location}.result.config.task.ref",
                "contradicts resolved package digest",
            )
        if any(
            config_task.get(field) is not None
            for field in ("path", "git_url", "git_commit_id")
        ):
            raise _fail(
                f"{location}.result.config.task",
                "contradicts package task lock",
            )

    config_agent = _object(config.get("agent"), f"{location}.result.config.agent")
    lock_agent = _object(trial_lock.get("agent"), f"{location}.lock.agent")
    if _normalized_mapping(
        config_agent, _AGENT_DEFAULTS, exclude={"skills"}
    ) != _normalized_mapping(lock_agent, _AGENT_DEFAULTS, exclude={"skills"}):
        raise _fail(f"{location}.result.config.agent", "does not match trial lock")
    config_skills = config_agent.get("skills", [])
    if not isinstance(config_skills, list):
        raise _fail(f"{location}.result.config.agent.skills", "expected an array")
    lock_skills = _array(trial_lock.get("skills", []), f"{location}.lock.skills")
    for index, skill in enumerate(lock_skills):
        skill_obj = _object(skill, f"{location}.lock.skills[{index}]")
        _validate_digest(skill_obj.get("digest"), f"{location}.lock.skills[{index}].digest")
    if [str(item) for item in config_skills] != [
        _string(item.get("source"), f"{location}.lock.skills.source")
        for item in lock_skills
    ]:
        raise _fail(
            f"{location}.result.config.agent.skills",
            "does not match resolved trial-lock skills",
        )

    config_environment = _object(
        config.get("environment"), f"{location}.result.config.environment"
    )
    lock_environment = _object(
        trial_lock.get("environment"), f"{location}.lock.environment"
    )
    if _normalized_mapping(
        config_environment,
        _ENVIRONMENT_DEFAULTS,
        exclude={"extra_docker_compose"},
    ) != _normalized_mapping(
        lock_environment,
        _ENVIRONMENT_DEFAULTS,
        exclude={"extra_docker_compose"},
    ):
        raise _fail(
            f"{location}.result.config.environment",
            "does not match trial lock",
        )
    config_compose = config_environment.get("extra_docker_compose", [])
    if not isinstance(config_compose, list):
        raise _fail(
            f"{location}.result.config.environment.extra_docker_compose",
            "expected an array",
        )
    lock_compose = trial_lock.get("extra_docker_compose") or []
    lock_compose = _array(lock_compose, f"{location}.lock.extra_docker_compose")
    for index, item in enumerate(lock_compose):
        item_obj = _object(item, f"{location}.lock.extra_docker_compose[{index}]")
        _validate_digest(
            item_obj.get("digest"),
            f"{location}.lock.extra_docker_compose[{index}].digest",
        )
    if [str(item) for item in config_compose] != [
        _string(item.get("path"), f"{location}.lock.extra_docker_compose.path")
        for item in lock_compose
    ]:
        raise _fail(
            f"{location}.result.config.environment.extra_docker_compose",
            "does not match resolved trial lock",
        )

    config_verifier = _object(
        config.get("verifier"), f"{location}.result.config.verifier"
    )
    lock_verifier = _object(trial_lock.get("verifier"), f"{location}.lock.verifier")
    if _normalized_mapping(
        config_verifier, _VERIFIER_DEFAULTS, exclude={"environment_mode"}
    ) != _normalized_mapping(
        lock_verifier, _VERIFIER_DEFAULTS, exclude={"environment_mode"}
    ):
        raise _fail(
            f"{location}.result.config.verifier",
            "does not match trial lock",
        )
    if result.get("verifier_environment_mode") != lock_verifier.get(
        "environment_mode"
    ):
        raise _fail(
            f"{location}.result.verifier_environment_mode",
            "does not match resolved trial lock",
        )

    config_instructions = config.get("extra_instruction_paths", [])
    if not isinstance(config_instructions, list):
        raise _fail(
            f"{location}.result.config.extra_instruction_paths",
            "expected an array",
        )
    lock_instructions = trial_lock.get("extra_instructions") or []
    lock_instructions = _array(lock_instructions, f"{location}.lock.extra_instructions")
    for index, item in enumerate(lock_instructions):
        item_obj = _object(item, f"{location}.lock.extra_instructions[{index}]")
        _validate_digest(
            item_obj.get("digest"),
            f"{location}.lock.extra_instructions[{index}].digest",
        )
    if [str(item) for item in config_instructions] != [
        _string(item.get("path"), f"{location}.lock.extra_instructions.path")
        for item in lock_instructions
    ]:
        raise _fail(
            f"{location}.result.config.extra_instruction_paths",
            "does not match resolved trial lock",
        )


def _validate_job_result(
    value: Any,
    *,
    expected_trials: int,
    actual_trial_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, datetime, datetime]:
    result = _object(value, "job result")
    job_id = _string(result.get("id"), "job result.id")
    started = _job_timestamp(result.get("started_at"), "job result.started_at")
    finished = _job_timestamp(result.get("finished_at"), "job result.finished_at")
    updated = _optional_job_timestamp(
        result.get("updated_at"), "job result.updated_at"
    )
    awareness = {value.tzinfo is not None for value in (started, finished)}
    if updated is not None:
        awareness.add(updated.tzinfo is not None)
    if len(awareness) != 1:
        raise _fail("job result", "job timestamps mix aware and naive datetimes")
    if finished < started:
        raise _fail("job result", "finished_at precedes started_at")
    if updated is not None and not (started <= updated <= finished):
        raise _fail("job result.updated_at", "must fall within job timing")
    if result.get("n_total_trials") != expected_trials:
        raise _fail(
            "job result.n_total_trials",
            f"expected {expected_trials}, got {result.get('n_total_trials')!r}",
        )
    stats = _object(result.get("stats"), "job result.stats")
    completed = stats.get("n_completed_trials")
    if completed != expected_trials:
        raise _fail(
            "job result.stats.n_completed_trials",
            f"job is incomplete: expected {expected_trials}, got {completed!r}",
        )
    for field in ("n_running_trials", "n_pending_trials"):
        if stats.get(field) != 0:
            raise _fail(f"job result.stats.{field}", "completed job must report zero")
    expected_exceptions = sorted(
        (
            _string(
                _object(
                    item.get("exception_info"),
                    "trial result.exception_info",
                ).get("exception_type"),
                "trial result.exception_info.exception_type",
            ),
            _string(item.get("trial_name"), "trial result.trial_name"),
        )
        for item in actual_trial_results
        if item.get("exception_info") is not None
    )
    expected_cancelled = sum(
        exception_type == "CancelledError"
        for exception_type, _trial_name in expected_exceptions
    )
    if stats.get("n_errored_trials") != len(expected_exceptions):
        raise _fail(
            "job result.stats.n_errored_trials",
            f"expected {len(expected_exceptions)!r}",
        )
    if stats.get("n_cancelled_trials") != expected_cancelled:
        raise _fail(
            "job result.stats.n_cancelled_trials",
            f"expected {expected_cancelled!r}",
        )
    _integer(stats.get("n_retries"), "job result.stats.n_retries")

    expected_reward_entries = sorted(
        (
            _string(item.get("trial_name"), "trial result.trial_name"),
            _number(
                _object(
                    _object(
                        item.get("verifier_result"),
                        "trial result.verifier_result",
                    ).get("rewards"),
                    "trial result.verifier_result.rewards",
                ).get("reward"),
                "trial result.verifier_result.rewards.reward",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for item in actual_trial_results
        if item.get("verifier_result") is not None
    )
    actual_reward_entries: list[tuple[str, float]] = []
    actual_exceptions: list[tuple[str, str]] = []
    evals = _object(stats.get("evals"), "job result.stats.evals")
    for eval_name, raw_eval in evals.items():
        eval_stats = _object(raw_eval, f"job result.stats.evals.{eval_name}")
        reward_stats = _object(
            eval_stats.get("reward_stats"),
            f"job result.stats.evals.{eval_name}.reward_stats",
        )
        reward_groups = _object(
            reward_stats.get("reward"),
            f"job result.stats.evals.{eval_name}.reward_stats.reward",
        )
        eval_trial_count = 0
        for raw_score, raw_names in reward_groups.items():
            try:
                parsed_score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise _fail(
                    f"job result.stats.evals.{eval_name}.reward_stats.reward",
                    "reward key must be numeric",
                ) from exc
            score = _number(
                parsed_score,
                f"job result.stats.evals.{eval_name}.reward_stats.reward",
                minimum=0.0,
                maximum=1.0,
            )
            names = _array(
                raw_names,
                f"job result.stats.evals.{eval_name}.reward_stats.reward.{raw_score}",
            )
            for trial_name in names:
                actual_reward_entries.append(
                    (
                        _string(
                            trial_name,
                            f"job result.stats.evals.{eval_name}.trial_name",
                        ),
                        score,
                    )
                )
                eval_trial_count += 1
        if eval_stats.get("n_trials") != eval_trial_count:
            raise _fail(
                f"job result.stats.evals.{eval_name}.n_trials",
                f"expected {eval_trial_count!r}",
            )
        exception_stats = _object(
            eval_stats.get("exception_stats"),
            f"job result.stats.evals.{eval_name}.exception_stats",
        )
        eval_error_count = 0
        for exception_type, raw_names in exception_stats.items():
            exception_type = _string(
                exception_type,
                f"job result.stats.evals.{eval_name}.exception_stats",
            )
            names = _array(
                raw_names,
                f"job result.stats.evals.{eval_name}.exception_stats.{exception_type}",
            )
            for trial_name in names:
                actual_exceptions.append(
                    (
                        exception_type,
                        _string(
                            trial_name,
                            f"job result.stats.evals.{eval_name}.exception_stats."
                            f"{exception_type}",
                        ),
                    )
                )
                eval_error_count += 1
        if eval_stats.get("n_errors") != eval_error_count:
            raise _fail(
                f"job result.stats.evals.{eval_name}.n_errors",
                f"expected {eval_error_count!r}",
            )
    if sorted(actual_reward_entries) != expected_reward_entries:
        raise _fail(
            "job result.stats.evals",
            "reward aggregates do not match enumerated trial directories",
        )
    if sorted(actual_exceptions) != expected_exceptions:
        raise _fail(
            "job result.stats.evals",
            "exception aggregates do not match enumerated trial directories",
        )
    aggregate_fields = {
        "n_input_tokens": "n_input_tokens",
        "n_cache_tokens": "n_cache_tokens",
        "n_output_tokens": "n_output_tokens",
        "cost_usd": "cost_usd",
    }
    for stats_field, result_field in aggregate_fields.items():
        values = []
        for trial_result in actual_trial_results:
            raw_agent_result = trial_result.get("agent_result")
            if raw_agent_result is None:
                continue
            agent_result = _object(
                raw_agent_result,
                f"trial result.agent_result for aggregate {stats_field}",
            )
            value = agent_result.get(result_field)
            if value is not None:
                values.append(
                    _number(
                        value,
                        f"trial result.agent_result.{result_field}",
                        minimum=0.0,
                    )
                )
        expected: float | int | None
        if not values:
            expected = None
        else:
            total = sum(values)
            expected = total if stats_field == "cost_usd" else int(total)
        actual = stats.get(stats_field)
        if stats_field == "cost_usd" and actual is not None and expected is not None:
            if (
                not isinstance(actual, (int, float))
                or isinstance(actual, bool)
                or not math.isclose(float(actual), float(expected), abs_tol=1e-9)
            ):
                raise _fail(
                    f"job result.stats.{stats_field}",
                    f"expected aggregate {expected!r}, got {actual!r}",
                )
        elif actual != expected:
            raise _fail(
                f"job result.stats.{stats_field}",
                f"expected aggregate {expected!r}, got {actual!r}",
            )
    return result, job_id, started, finished


def _trial_directories(job_dir: Path) -> list[Path]:
    directories: list[Path] = []
    for child in sorted(job_dir.iterdir(), key=lambda path: path.name):
        if child.is_symlink():
            raise _fail("job directory", f"symlink is not accepted: {child}")
        if not child.is_dir():
            continue
        evidence_names = {
            "lock.json",
            "result.json",
            "config.json",
            "agent",
            "verifier",
            "artifacts",
            "steps",
        }
        if not any((child / name).exists() for name in evidence_names):
            raise _fail("job directory", f"unexpected directory: {child.name}")
        directories.append(child)
    if not directories:
        raise _fail("job directory", "contains no completed trial directories")
    return directories


def _validate_timing(
    result: dict[str, Any], location: str
) -> tuple[datetime, datetime, float | None, float | None, float | None]:
    started = _timestamp(result.get("started_at"), f"{location}.started_at")
    finished = _timestamp(result.get("finished_at"), f"{location}.finished_at")
    if finished < started:
        raise _fail(location, "finished_at precedes started_at")

    phases: list[tuple[str, datetime, datetime]] = []
    durations: dict[str, float | None] = {}
    for name in ("environment_setup", "agent_setup", "agent_execution", "verifier"):
        phase = result.get(name)
        if phase is None:
            durations[name] = None
            continue
        phase_obj = _object(phase, f"{location}.{name}")
        phase_start = _timestamp(
            phase_obj.get("started_at"), f"{location}.{name}.started_at"
        )
        phase_finish = _timestamp(
            phase_obj.get("finished_at"), f"{location}.{name}.finished_at"
        )
        if not (started <= phase_start <= phase_finish <= finished):
            raise _fail(
                f"{location}.{name}",
                "phase timing must be ordered within trial timing",
            )
        phases.append((name, phase_start, phase_finish))
        durations[name] = round((phase_finish - phase_start).total_seconds(), 3)
    for previous, current in zip(phases, phases[1:]):
        if current[1] < previous[2]:
            raise _fail(
                location,
                f"{current[0]} starts before {previous[0]} finishes",
            )
    env_duration_values = [
        durations["environment_setup"],
        durations["agent_setup"],
    ]
    env_duration = (
        round(sum(value for value in env_duration_values if value is not None), 3)
        if any(value is not None for value in env_duration_values)
        else None
    )
    return (
        started,
        finished,
        env_duration,
        durations["agent_execution"],
        durations["verifier"],
    )


def _validate_exception_info(
    value: Any,
    *,
    started: datetime,
    finished: datetime,
    location: str,
) -> tuple[str, str, str] | None:
    if value is None:
        return None
    exception = _object(value, f"{location}.result.exception_info")
    expected_fields = {
        "exception_type",
        "exception_message",
        "exception_traceback",
        "occurred_at",
    }
    if set(exception) != expected_fields:
        raise _fail(
            f"{location}.result.exception_info",
            "unexpected or missing exception fields",
        )
    exception_type = _string(
        exception.get("exception_type"),
        f"{location}.result.exception_info.exception_type",
    )
    for field in ("exception_message", "exception_traceback"):
        if not isinstance(exception.get(field), str):
            raise _fail(
                f"{location}.result.exception_info.{field}",
                "expected a string",
            )
    occurred_at = _timestamp(
        exception.get("occurred_at"),
        f"{location}.result.exception_info.occurred_at",
    )
    if not started <= occurred_at <= finished:
        raise _fail(
            f"{location}.result.exception_info.occurred_at",
            "must fall within trial timing",
        )
    failure_class, failure_reason = harbor_exception_semantics(exception_type)
    return exception_type, failure_class, failure_reason


def _validate_reward(
    trial_dir: Path, result: dict[str, Any], location: str
) -> tuple[
    float,
    int,
    float | None,
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    reward_json = trial_dir / "verifier" / "reward.json"
    reward_text = trial_dir / "verifier" / "reward.txt"
    existing = [path for path in (reward_json, reward_text) if path.exists()]
    if len(existing) != 1:
        raise _fail(
            f"{location}.reward",
            "expected exactly one of verifier/reward.json or verifier/reward.txt",
        )
    reward_path = existing[0]
    _require_regular_file(reward_path, f"{location}.reward")
    if reward_path.name == "reward.json":
        rewards = _object(_read_json(reward_path, f"{location}.reward"), f"{location}.reward")
    else:
        try:
            rewards = {"reward": float(reward_path.read_text(encoding="utf-8").strip())}
        except (OSError, UnicodeError, ValueError) as exc:
            raise _fail(f"{location}.reward", "reward.txt must contain one number") from exc
    if set(rewards) != {"reward"}:
        raise _fail(f"{location}.reward", "expected exactly the scalar 'reward' metric")
    score = _number(
        rewards["reward"], f"{location}.reward.reward", minimum=0.0, maximum=1.0
    )
    verifier_result = _object(
        result.get("verifier_result"), f"{location}.result.verifier_result"
    )
    if verifier_result.get("rewards") != rewards:
        raise _fail(
            f"{location}.result.verifier_result.rewards",
            "does not match verifier reward evidence",
        )
    evidence_path = trial_dir / "verifier" / "openbench-verifier-evidence.json"
    evidence = _object(
        _read_json(evidence_path, f"{location}.verifier_evidence"),
        f"{location}.verifier_evidence",
    )
    if set(evidence) != {
        "schema_version",
        "openbench_task_content_digest",
        "openbench_harbor_export",
        "checker_exit",
        "parsed_score",
        "reward",
        "verifier_duration_seconds",
    }:
        raise _fail(
            f"{location}.verifier_evidence",
            "unexpected or missing evidence fields",
        )
    if evidence.get("schema_version") != VERIFIER_EVIDENCE_SCHEMA_VERSION:
        raise _fail(
            f"{location}.verifier_evidence.schema_version",
            f"expected {VERIFIER_EVIDENCE_SCHEMA_VERSION!r}",
        )
    openbench_task_content_digest = _validate_openbench_task_content_digest(
        evidence.get("openbench_task_content_digest"),
        f"{location}.verifier_evidence.openbench_task_content_digest",
    )
    openbench_harbor_export = _validate_openbench_harbor_export(
        evidence.get("openbench_harbor_export"),
        f"{location}.verifier_evidence.openbench_harbor_export",
    )
    checker_exit = evidence.get("checker_exit")
    if not isinstance(checker_exit, int) or isinstance(checker_exit, bool):
        raise _fail(
            f"{location}.verifier_evidence.checker_exit",
            "expected an integer",
        )
    parsed_score_value = evidence.get("parsed_score")
    parsed_score = (
        None
        if parsed_score_value is None
        else _number(
            parsed_score_value,
            f"{location}.verifier_evidence.parsed_score",
            minimum=0.0,
            maximum=1.0,
        )
    )
    evidence_reward = _number(
        evidence.get("reward"),
        f"{location}.verifier_evidence.reward",
        minimum=0.0,
        maximum=1.0,
    )
    if evidence_reward != score:
        raise _fail(
            f"{location}.verifier_evidence.reward",
            "does not match reward file and trial result",
        )
    duration = evidence.get("verifier_duration_seconds")
    checker_duration = None
    if duration is not None:
        checker_duration = _number(
            duration,
            f"{location}.verifier_evidence.verifier_duration_seconds",
            minimum=0.0,
        )
    expected_score = 1.0 if checker_exit == 0 else (
        parsed_score if parsed_score is not None else 0.0
    )
    if score != expected_score:
        raise _fail(
            f"{location}.verifier_evidence",
            "checker exit, parsed score, and Harbor reward are incoherent",
        )
    return (
        score,
        checker_exit,
        checker_duration,
        reward_path,
        evidence_path,
        openbench_task_content_digest,
        openbench_harbor_export,
    )


def _validate_artifact_source(value: Any, location: str) -> str:
    source = _string(value, location)
    pure_source = PurePosixPath(source)
    if (
        "\0" in source
        or source.startswith("//")
        or not pure_source.is_absolute()
        or source == "/"
        or ".." in pure_source.parts
        or source != pure_source.as_posix()
    ):
        raise _fail(location, "must be a canonical non-root absolute POSIX path")
    return source


def _validate_relative_artifact_path(value: Any, location: str) -> str:
    destination = _string(value, location)
    pure_destination = PurePosixPath(destination)
    if (
        "\0" in destination
        or pure_destination.is_absolute()
        or ".." in pure_destination.parts
        or destination != pure_destination.as_posix()
        or pure_destination.as_posix() == "."
    ):
        raise _fail(location, "must be a canonical relative POSIX path")
    return destination


def _validate_manifest_artifact_path(value: Any, location: str) -> str:
    destination = _validate_relative_artifact_path(value, location)
    parts = PurePosixPath(destination).parts
    if len(parts) < 2 or parts[0] != "artifacts":
        raise _fail(location, "must be a safe path below artifacts/")
    return destination


def _path_is_at_or_below(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def _validate_artifacts(
    trial_dir: Path,
    result: dict[str, Any],
    location: str,
    *,
    required: bool,
) -> tuple[Path | None, str | None]:
    config = _object(result.get("config"), f"{location}.result.config")
    artifacts = _array(config.get("artifacts"), f"{location}.result.config.artifacts")
    configured_workspace_entries: list[tuple[str, str]] = []
    for index, artifact in enumerate(artifacts):
        entry_location = f"{location}.result.config.artifacts[{index}]"
        entry = _object(artifact, entry_location)
        source = _validate_artifact_source(
            entry.get("source"), f"{entry_location}.source"
        )
        destination = _validate_relative_artifact_path(
            entry.get("destination"), f"{entry_location}.destination"
        )
        if (
            source == FINAL_WORKSPACE_SOURCE
            or _path_is_at_or_below(destination, FINAL_WORKSPACE_DESTINATION)
        ):
            if set(entry) - {"source", "destination", "service"}:
                raise _fail(
                    entry_location,
                    "workspace config contains unsupported semantic fields",
                )
            if entry.get("service") is not None:
                raise _fail(
                    f"{entry_location}.service",
                    "workspace config service must be null or absent",
                )
            configured_workspace_entries.append((source, destination))
    if artifacts and configured_workspace_entries != [
        (FINAL_WORKSPACE_SOURCE, FINAL_WORKSPACE_DESTINATION)
    ]:
        raise _fail(
            f"{location}.result.config.artifacts",
            "non-empty config must declare exactly one non-conflicting "
            f"{FINAL_WORKSPACE_SOURCE} to {FINAL_WORKSPACE_DESTINATION} artifact",
        )

    manifest_path = trial_dir / "artifacts" / "manifest.json"
    workspace_path = trial_dir / FINAL_WORKSPACE_MANIFEST_DESTINATION
    if not manifest_path.exists():
        if required:
            raise _fail(
                f"{location}.artifacts",
                "required artifact manifest is missing",
            )
        if workspace_path.exists() or workspace_path.is_symlink():
            raise _fail(
                f"{location}.artifacts",
                "workspace exists without an artifact manifest",
            )
        return None, None
    manifest = _array(
        _read_json(manifest_path, f"{location}.artifacts"),
        f"{location}.artifacts",
    )
    seen_sources: set[tuple[str, str | None]] = set()
    seen_destinations: set[str] = set()
    final_entry: dict[str, Any] | None = None
    for index, raw_entry in enumerate(manifest):
        entry_location = f"{location}.artifacts[{index}]"
        entry = _object(raw_entry, entry_location)
        if set(entry) - {"source", "destination", "type", "status", "service"}:
            raise _fail(entry_location, "contains unknown manifest fields")
        entry_source = _validate_artifact_source(
            entry.get("source"), f"{entry_location}.source"
        )
        entry_destination = _validate_manifest_artifact_path(
            entry.get("destination"), f"{entry_location}.destination"
        )
        if entry.get("type") not in {"file", "directory"}:
            raise _fail(f"{entry_location}.type", "invalid artifact type")
        if entry.get("status") not in {"ok", "failed", "empty", "skipped"}:
            raise _fail(f"{entry_location}.status", "invalid artifact status")
        service = entry.get("service")
        if service is not None and not isinstance(service, str):
            raise _fail(f"{entry_location}.service", "must be a string or null")
        source_key = (entry_source, service)
        if source_key in seen_sources or entry_destination in seen_destinations:
            raise _fail(entry_location, "duplicate artifact source or destination")
        seen_sources.add(source_key)
        seen_destinations.add(entry_destination)
        if (
            entry_source == FINAL_WORKSPACE_SOURCE
            or _path_is_at_or_below(
                entry_destination,
                FINAL_WORKSPACE_MANIFEST_DESTINATION,
            )
        ):
            if final_entry is not None:
                raise _fail(
                    entry_location,
                    "duplicate or contradictory workspace artifact entry",
                )
            final_entry = entry

    if final_entry is None:
        if not required:
            return manifest_path, None
        raise _fail(
            f"{location}.artifacts",
            "manifest does not contain the required workspace",
        )
    if (
        final_entry.get("source") != FINAL_WORKSPACE_SOURCE
        or final_entry.get("destination") != FINAL_WORKSPACE_MANIFEST_DESTINATION
    ):
        raise _fail(
            f"{location}.artifacts",
            "workspace entry must map "
            f"{FINAL_WORKSPACE_SOURCE} to {FINAL_WORKSPACE_MANIFEST_DESTINATION}",
        )
    if final_entry.get("type") != "directory" or final_entry.get("service") is not None:
        raise _fail(
            f"{location}.artifacts",
            "workspace must have type='directory' and service=null",
        )
    if final_entry.get("status") != "ok":
        if required:
            raise _fail(
                f"{location}.artifacts",
                "workspace must have status='ok'",
            )
        if workspace_path.is_symlink():
            raise _fail(f"{location}.artifacts", "workspace must not be a symlink")
        if workspace_path.exists():
            if not workspace_path.is_dir():
                raise _fail(f"{location}.artifacts", "workspace must be a directory")
            if any(workspace_path.iterdir()):
                raise _fail(
                    f"{location}.artifacts",
                    "non-ok workspace artifact must not contain files",
                )
        return manifest_path, None
    return manifest_path, _tree_digest(workspace_path)


def _validate_atif(
    trial_dir: Path,
    result: dict[str, Any],
    agent_name: str,
    agent_version: str,
    lock_model: str,
    canonical_model: str,
    reported_model_name: str,
    location: str,
) -> tuple[Path, dict[str, Any], int, dict[str, Any]]:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory = _object(
        _read_json(trajectory_path, f"{location}.ATIF"),
        f"{location}.ATIF",
    )
    errors = validate_trajectory(trajectory)
    if errors:
        raise _fail(f"{location}.ATIF", "; ".join(errors))
    agent = _object(trajectory.get("agent"), f"{location}.ATIF.agent")
    if agent.get("name") != agent_name or agent.get("version") != agent_version:
        raise _fail(
            f"{location}.ATIF.agent",
            "does not match trial result agent identity",
        )
    if agent.get("model_name") not in (
        None,
        reported_model_name,
        lock_model,
        canonical_model,
    ):
        raise _fail(
            f"{location}.ATIF.agent.model_name",
            "does not match trial model identity",
        )
    for index, step in enumerate(trajectory.get("steps", [])):
        if (
            isinstance(step, dict)
            and step.get("source") == "agent"
            and step.get("model_name") not in (
                None,
                lock_model,
                canonical_model,
            )
        ):
            raise _fail(
                f"{location}.ATIF.steps[{index}].model_name",
                "does not match trial model identity",
            )

    agent_result = _object(result.get("agent_result"), f"{location}.result.agent_result")
    final_metrics_value = trajectory.get("final_metrics")
    final_metrics = (
        {}
        if final_metrics_value is None
        else _object(final_metrics_value, f"{location}.ATIF.final_metrics")
    )
    metric_pairs = (
        ("n_input_tokens", "total_prompt_tokens"),
        ("n_cache_tokens", "total_cached_tokens"),
        ("n_output_tokens", "total_completion_tokens"),
        ("cost_usd", "total_cost_usd"),
    )
    for result_name, atif_name in metric_pairs:
        result_value = agent_result.get(result_name)
        atif_value = final_metrics.get(atif_name)
        if result_value is not None and final_metrics_value is None:
            raise _fail(
                f"{location}.ATIF.final_metrics",
                "required when the agent result reports usage",
            )
        if result_value != atif_value:
            raise _fail(
                f"{location}.ATIF.final_metrics.{atif_name}",
                f"does not match result.agent_result.{result_name}",
            )
    turns = sum(
        1
        for step in trajectory.get("steps", [])
        if isinstance(step, dict) and step.get("source") == "agent"
    )
    return trajectory_path, agent_result, turns, trajectory


def _usage_fields(agent_result: dict[str, Any], location: str) -> dict[str, Any]:
    names = ("n_input_tokens", "n_cache_tokens", "n_output_tokens")
    values = [agent_result.get(name) for name in names]
    if all(value is None for value in values):
        return {
            "tokens": None,
            "tokens_input_uncached": None,
            "tokens_cache_read": None,
            "tokens_cache_write": None,
            "tokens_output": None,
            "tokens_reasoning": None,
            "tokens_fresh": None,
            "usage_raw": None,
            "token_basis": "unmetered",
        }
    if any(value is None for value in values):
        raise _fail(f"{location}.result.agent_result", "token usage is partial")
    input_tokens = _integer(values[0], f"{location}.result.agent_result.n_input_tokens")
    cache_tokens = _integer(values[1], f"{location}.result.agent_result.n_cache_tokens")
    output_tokens = _integer(values[2], f"{location}.result.agent_result.n_output_tokens")
    if cache_tokens > input_tokens:
        raise _fail(
            f"{location}.result.agent_result.n_cache_tokens",
            "cannot exceed n_input_tokens",
        )
    uncached = input_tokens - cache_tokens
    return {
        "tokens": uncached + output_tokens,
        "tokens_input_uncached": uncached,
        "tokens_cache_read": cache_tokens,
        "tokens_cache_write": None,
        "tokens_output": output_tokens,
        "tokens_reasoning": None,
        "tokens_fresh": uncached + output_tokens,
        "usage_raw": {
            "source": "harbor_agent_result",
            "n_input_tokens": input_tokens,
            "n_cache_tokens": cache_tokens,
            "n_output_tokens": output_tokens,
            "cost_usd": agent_result.get("cost_usd"),
        },
        "token_basis": "harbor_agent_reported",
    }


def _validate_trial(
    trial_dir: Path,
    expected_locks: Counter[str],
    seen_ids: set[str],
    seen_names: set[str],
    expected_job_id: str,
    *,
    comparison_plan_requested: bool,
) -> dict[str, Any]:
    location = f"trial {trial_dir.name!r}"
    trial_lock_path = trial_dir / "lock.json"
    result_path = trial_dir / "result.json"
    trial_lock = _validate_trial_lock(
        _read_json(trial_lock_path, f"{location}.lock"), f"{location}.lock"
    )
    canonical_lock = _canonical_lock(trial_lock)
    if expected_locks[canonical_lock] <= 0:
        raise _fail(f"{location}.lock", "is not an unconsumed trial from the job lock")
    expected_locks[canonical_lock] -= 1

    result = _object(_read_json(result_path, f"{location}.result"), f"{location}.result")
    trial_id = _string(result.get("id"), f"{location}.result.id")
    trial_name = _string(result.get("trial_name"), f"{location}.result.trial_name")
    if trial_name != trial_dir.name:
        raise _fail(f"{location}.result.trial_name", "does not match directory name")
    if trial_id in seen_ids:
        raise _fail(f"{location}.result.id", f"duplicate trial id {trial_id!r}")
    if trial_name in seen_names:
        raise _fail(f"{location}.result.trial_name", "duplicate trial name")
    seen_ids.add(trial_id)
    seen_names.add(trial_name)
    if result.get("step_results") is not None:
        raise _fail(f"{location}.result.step_results", "multi-step trials are not supported")
    _validate_lock_backed_config(
        result,
        trial_lock,
        expected_job_id=expected_job_id,
        location=location,
    )

    task = _object(trial_lock.get("task"), f"{location}.lock.task")
    task_name = _string(result.get("task_name"), f"{location}.result.task_name")
    if _normalized_task_name(task_name) != _normalized_task_name(
        _string(task.get("name"), f"{location}.lock.task.name")
    ):
        raise _fail(f"{location}.result.task_name", "does not match trial lock")
    task_digest = _validate_digest(task.get("digest"), f"{location}.lock.task.digest")
    checksum = _string(result.get("task_checksum"), f"{location}.result.task_checksum")
    checksum_hex = checksum.removeprefix("sha256:")
    if len(checksum_hex) != 64 or any(
        char not in "0123456789abcdef" for char in checksum_hex
    ):
        raise _fail(
            f"{location}.result.task_checksum",
            "expected Harbor's 64-character lowercase dirhash checksum",
        )
    result_task_id = _object(result.get("task_id"), f"{location}.result.task_id")
    config_task = _object(
        _object(result.get("config"), f"{location}.result.config").get("task"),
        f"{location}.result.config.task",
    )
    task_type = task.get("type")
    identity_field = "path" if task_type == "local" else "name"
    if (
        result_task_id.get(identity_field) != config_task.get(identity_field)
        or config_task.get(identity_field) != task.get(identity_field)
    ):
        raise _fail(
            f"{location}.result.task_id",
            "does not match result config and resolved task lock",
        )

    agent_lock = _object(trial_lock.get("agent"), f"{location}.lock.agent")
    agent_config_name = _agent_config_identity(
        agent_lock, f"{location}.lock.agent"
    )
    if (
        not comparison_plan_requested
        and agent_config_name not in HARBOR_STOCK_AGENT_CONFIG_NAMES
    ):
        raise _fail(
            f"{location}.lock.agent",
            "custom Harbor agents require an exact OpenBench comparison plan",
        )
    agent_info = _object(result.get("agent_info"), f"{location}.result.agent_info")
    agent_name = _string(agent_info.get("name"), f"{location}.result.agent_info.name")
    expected_agent_name = HARBOR_AGENT_SEMANTIC_NAME_ALIASES.get(
        agent_config_name
    )
    if expected_agent_name is None and agent_config_name in HARBOR_STOCK_AGENT_CONFIG_NAMES:
        expected_agent_name = agent_config_name
    if expected_agent_name is not None and agent_name != expected_agent_name:
        raise _fail(
            f"{location}.result.agent_info.name",
            "does not match immutable trial-lock agent identity "
            f"{agent_config_name!r} (expected {expected_agent_name!r})",
        )
    agent_version = _string(
        agent_info.get("version"), f"{location}.result.agent_info.version"
    )
    model_info = _object(
        agent_info.get("model_info"), f"{location}.result.agent_info.model_info"
    )
    model_name = _string(
        model_info.get("name"), f"{location}.result.agent_info.model_info.name"
    )
    provider = model_info.get("provider")
    if provider is not None and not isinstance(provider, str):
        raise _fail(
            f"{location}.result.agent_info.model_info.provider",
            "expected a string or null",
        )
    model = f"{provider}/{model_name}" if provider else model_name
    if model != agent_lock.get("model_name"):
        raise _fail(
            f"{location}.result.agent_info.model_info",
            "does not match trial lock model identity",
        )
    if agent_config_name in HARBOR_STOCK_AGENT_CONFIG_NAMES:
        try:
            canonical_model = canonical_openbench_model(agent_config_name, model)
        except HarborProfileError as exc:
            raise _fail(f"{location}.lock.agent.model_name", str(exc)) from exc
    else:
        # Planned custom labels are applied only after the exact config, lock,
        # and result chain is verified. ATIF remains bound to this raw lock model.
        canonical_model = model

    started, finished, t_env, t_agent, harbor_verifier_time = _validate_timing(
        result, location
    )
    exception = _validate_exception_info(
        result.get("exception_info"),
        started=started,
        finished=finished,
        location=location,
    )
    terminal_failure = exception is not None

    reward_files = (
        trial_dir / "verifier" / "reward.json",
        trial_dir / "verifier" / "reward.txt",
        trial_dir / "verifier" / "openbench-verifier-evidence.json",
    )
    has_reward_evidence = (
        result.get("verifier_result") is not None
        or any(path.exists() or path.is_symlink() for path in reward_files)
    )
    score = None
    checker_exit = None
    t_checker = None
    reward_path = None
    verifier_evidence_path = None
    openbench_task_content_digest = None
    openbench_harbor_export = None
    if not terminal_failure or has_reward_evidence:
        (
            score,
            checker_exit,
            t_checker,
            reward_path,
            verifier_evidence_path,
            openbench_task_content_digest,
            openbench_harbor_export,
        ) = _validate_reward(trial_dir, result, location)

    manifest_path, workspace_digest = _validate_artifacts(
        trial_dir,
        result,
        location,
        required=not terminal_failure,
    )

    trajectory_candidate = trial_dir / "agent" / "trajectory.json"
    has_trajectory = trajectory_candidate.exists() or trajectory_candidate.is_symlink()
    raw_agent_result = result.get("agent_result")
    has_agent_usage = (
        isinstance(raw_agent_result, dict)
        and any(
            raw_agent_result.get(field) is not None
            for field in (
                "n_input_tokens",
                "n_cache_tokens",
                "n_output_tokens",
                "cost_usd",
            )
        )
    )
    trajectory_path = None
    trajectory = None
    turns = None
    usage = _usage_fields({}, location)
    if not terminal_failure or has_trajectory:
        trajectory_path, agent_result, turns, trajectory = _validate_atif(
            trial_dir,
            result,
            agent_name,
            agent_version,
            model,
            canonical_model,
            model_name,
            location,
        )
        usage = _usage_fields(agent_result, location)
    elif has_agent_usage:
        raise _fail(
            f"{location}.result.agent_result",
            "usage-bearing terminal result requires matching ATIF evidence",
        )
    elif raw_agent_result is not None:
        _object(raw_agent_result, f"{location}.result.agent_result")

    proxy_required = agent_config_name in HARBOR_PROXY_REQUIRED_AGENTS
    metering_path = trial_dir / "agent" / "harbor-metering"
    metering_evidence = None
    metering_evidence_path = None
    has_metering = metering_path.exists() or metering_path.is_symlink()
    if proxy_required and trajectory is not None:
        try:
            metering_evidence = verify_evidence_dir(
                metering_path,
                expected_trial_id=trial_dir.name,
                expected_harness=agent_name,
                proxy_required=True,
                expected_agent_usage=UsageCounters.from_atif_trajectory(
                    trajectory
                ),
            )
        except HarborMeteringError as exc:
            raise _fail(f"{location}.metering", str(exc)) from exc
        metering_evidence_path = metering_path / "harbor-metering.json"
    elif proxy_required and has_metering:
        try:
            verify_evidence_dir(
                metering_path,
                expected_trial_id=trial_dir.name,
                expected_harness=agent_name,
                proxy_required=False,
                expected_agent_usage=UsageCounters(None, None, None, None),
            )
        except HarborMeteringError as exc:
            raise _fail(f"{location}.metering", str(exc)) from exc
    return {
        "trial_dir": trial_dir,
        "result": result,
        "trial_lock_path": trial_lock_path,
        "result_path": result_path,
        "reward_path": reward_path,
        "verifier_evidence_path": verifier_evidence_path,
        "manifest_path": manifest_path,
        "trajectory_path": trajectory_path,
        "trial_id": trial_id,
        "trial_name": trial_name,
        "task_name": task_name,
        "task_lock": task,
        "task_digest": task_digest,
        "openbench_task_content_digest": openbench_task_content_digest,
        "openbench_harbor_export": openbench_harbor_export,
        "task_checksum": checksum,
        "agent_config_name": agent_config_name,
        "agent_config_match_key": _agent_config_match_key(agent_lock),
        "agent_name": agent_name,
        "agent_version": agent_version,
        "model": canonical_model,
        "harbor_model_name": model,
        "started": started,
        "finished": finished,
        "t_env_setup_s": t_env,
        "t_agent_s": t_agent,
        "t_checker_s": t_checker,
        "harbor_verifier_time_s": harbor_verifier_time,
        "exception": exception,
        "score": score,
        "checker_exit": checker_exit,
        "turns": turns,
        "usage": usage,
        "proxy_required": proxy_required,
        "metering_evidence": metering_evidence,
        "metering_evidence_path": metering_evidence_path,
        "workspace_digest": workspace_digest,
    }


def _run_ids_from_jsonl(contents: bytes) -> set[str]:
    run_ids: set[str] = set()
    try:
        lines = contents.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise _fail("output", f"existing JSONL is not UTF-8: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fail("output", f"corrupt JSONL at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise _fail("output", f"line {line_number} is not a JSON object")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise _fail("output", f"line {line_number} has no valid run_id")
        if run_id in run_ids:
            raise _fail("output", f"duplicate existing run_id {run_id!r}")
        run_ids.add(run_id)
    return run_ids


def _normalized_task_name(task_name: str) -> str:
    return task_name.removeprefix("openbench/")


def _local_task_selector(
    task: dict[str, Any],
    *,
    dataset_root: Path,
    task_id_map: dict[str, str],
    location: str,
) -> str:
    if task.get("type") != "local":
        raise _fail(location, "local comparison plan resolved a non-local task")
    raw_path = Path(_string(task.get("path"), f"{location}.path")).expanduser()
    if raw_path.is_symlink():
        raise _fail(f"{location}.path", "local task path must not be a symlink")
    task_path = raw_path.resolve()
    if task_path.parent != dataset_root:
        raise _fail(
            f"{location}.path",
            "must be one direct child of the configured local task set",
        )
    selector = task_path.name
    if selector not in task_id_map:
        raise _fail(
            f"{location}.path",
            "does not identify a task selector in the comparison plan",
        )
    config_path = task_path / "task.toml"
    if config_path.is_symlink() or not config_path.is_file():
        raise _fail(
            f"{location}.path",
            "selected local task has no safe task.toml",
        )
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise _fail(
            f"{location}.path",
            f"cannot read selected local task.toml: {exc}",
        ) from exc
    task_config = config.get("task")
    canonical_id = (
        task_config.get("name") if isinstance(task_config, dict) else None
    )
    if canonical_id != task_id_map[selector]:
        raise _fail(
            f"{location}.path",
            "task.toml identity does not match comparison plan.task_id_map",
        )
    return selector


def _validate_comparison_plan(
    path: str | os.PathLike[str],
    *,
    submitted_job_config_path: str | os.PathLike[str],
    job_root: Path,
    job_lock: dict[str, Any],
    trials: list[dict[str, Any]],
) -> tuple[
    str,
    dict[str, Any],
    tuple[str, ...],
    dict[str, dict[str, Any]],
]:
    plan_path = Path(path).expanduser()
    plan = _object(
        _read_json(plan_path, "comparison plan"),
        "comparison plan",
    )
    expected_fields = {
        "schema_version",
        "harbor_version",
        "harbor_git_commit_hash",
        "job_name",
        "submitted_job_config_sha256",
        "effective_job_config_sha256",
        "attempts",
        "dataset",
        "tasks",
        "task_id_map",
        "arms",
    }
    if set(plan) != expected_fields:
        raise _fail(
            "comparison plan",
            "unexpected or missing fields",
        )
    if plan_path.read_bytes() != canonical_comparison_plan_bytes(plan):
        raise _fail(
            "comparison plan",
            "must use canonical OpenBench JSON encoding",
        )
    if plan.get("schema_version") != COMPARISON_PLAN_SCHEMA_VERSION:
        raise _fail(
            "comparison plan.schema_version",
            f"expected {COMPARISON_PLAN_SCHEMA_VERSION!r}",
        )
    if plan.get("harbor_version") != HARBOR_VERSION:
        raise _fail(
            "comparison plan.harbor_version",
            f"expected {HARBOR_VERSION!r}",
        )
    if plan.get("harbor_git_commit_hash") != HARBOR_GIT_COMMIT:
        raise _fail(
            "comparison plan.harbor_git_commit_hash",
            f"expected {HARBOR_GIT_COMMIT}",
        )
    job_name = _string(plan.get("job_name"), "comparison plan.job_name")
    if job_name != job_root.name:
        raise _fail(
            "comparison plan.job_name",
            "does not match Harbor job directory",
        )
    submitted_config_digest = _string(
        plan.get("submitted_job_config_sha256"),
        "comparison plan.submitted_job_config_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{64}", submitted_config_digest) is None:
        raise _fail(
            "comparison plan.submitted_job_config_sha256",
            "expected a lowercase SHA-256",
        )
    effective_config_digest = _string(
        plan.get("effective_job_config_sha256"),
        "comparison plan.effective_job_config_sha256",
    )
    if re.fullmatch(r"[0-9a-f]{64}", effective_config_digest) is None:
        raise _fail(
            "comparison plan.effective_job_config_sha256",
            "expected a lowercase SHA-256",
        )

    submitted_config_path = Path(submitted_job_config_path).expanduser()
    submitted_config = _object(
        _read_json(submitted_config_path, "submitted job config"),
        "submitted job config",
    )
    if _sha256_file(submitted_config_path) != submitted_config_digest:
        raise _fail(
            "comparison plan.submitted_job_config_sha256",
            "does not match OpenBench's submitted job config",
        )
    try:
        submitted_effective_digest = canonical_harbor_job_config_sha256(
            submitted_config
        )
    except ValueError as exc:
        raise _fail(
            "submitted job config",
            f"cannot derive pinned semantic identity: {exc}",
        ) from exc
    if submitted_effective_digest != effective_config_digest:
        raise _fail(
            "comparison plan.effective_job_config_sha256",
            "does not match OpenBench's submitted semantic job config",
        )

    job_config_path = job_root / "config.json"
    job_config = _object(
        _read_json(job_config_path, "job config"),
        "job config",
    )
    try:
        persisted_effective_digest = canonical_harbor_job_config_sha256(
            job_config
        )
    except ValueError as exc:
        raise _fail(
            "job config",
            f"cannot derive pinned semantic identity: {exc}",
        ) from exc
    if persisted_effective_digest != effective_config_digest:
        raise _fail(
            "comparison plan.effective_job_config_sha256",
            "does not match Harbor's persisted semantic job config",
        )
    attempts = _integer(
        plan.get("attempts"),
        "comparison plan.attempts",
        minimum=1,
    )
    if job_config.get("job_name") != job_name:
        raise _fail(
            "job config.job_name",
            "does not match comparison plan",
        )
    if job_config.get("n_attempts", 1) != attempts:
        raise _fail(
            "job config.n_attempts",
            "does not match comparison plan",
        )

    datasets = _array(job_config.get("datasets"), "job config.datasets")
    if len(datasets) != 1:
        raise _fail(
            "job config.datasets",
            "comparison plan requires exactly one dataset",
        )
    dataset = _object(datasets[0], "job config.datasets[0]")
    if job_config.get("tasks", []) != []:
        raise _fail(
            "job config.tasks",
            "comparison plan requires one dataset and no standalone tasks",
        )

    plan_dataset = plan.get("dataset")
    raw_tasks = plan.get("tasks")
    raw_task_id_map = plan.get("task_id_map")
    if plan_dataset is None:
        if "path" not in dataset or "name" in dataset:
            raise _fail(
                "comparison plan.dataset",
                "null is valid only for a local task set",
            )
        task_values = _array(raw_tasks, "comparison plan.tasks")
        tasks = tuple(
            _string(task, f"comparison plan.tasks[{index}]")
            for index, task in enumerate(task_values)
        )
        if dataset.get("task_names") != list(tasks):
            raise _fail(
                "job config.datasets[0].task_names",
                "does not match comparison plan",
            )
        dataset_path = Path(
            _string(dataset.get("path"), "job config.datasets[0].path")
        ).expanduser()
        if dataset_path.is_symlink():
            raise _fail(
                "job config.datasets[0].path",
                "local task-set path must not be a symlink",
            )
        dataset_root = dataset_path.resolve()
        if not dataset_root.is_dir():
            raise _fail(
                "job config.datasets[0].path",
                "local task-set path is not a directory",
            )
        task_id_map = _object(
            raw_task_id_map,
            "comparison plan.task_id_map",
        )
        if set(task_id_map) != set(tasks):
            raise _fail(
                "comparison plan.task_id_map",
                "keys must match every Harbor task selector exactly",
            )
        canonical_task_ids = tuple(
            _string(
                task_id_map[task],
                f"comparison plan.task_id_map[{task!r}]",
            )
            for task in tasks
        )
        if any(
            task_id != task_id.strip() or "\x00" in task_id
            for task_id in canonical_task_ids
        ):
            raise _fail(
                "comparison plan.task_id_map",
                "canonical task IDs contain unsafe whitespace or NUL",
            )
        if len(set(canonical_task_ids)) != len(canonical_task_ids):
            raise _fail(
                "comparison plan.task_id_map",
                "canonical task IDs must be unique",
            )
    else:
        descriptor = _object(plan_dataset, "comparison plan.dataset")
        if descriptor != dataset:
            raise _fail(
                "comparison plan.dataset",
                "does not match the immutable Harbor dataset descriptor",
            )
        if raw_tasks is not None:
            raise _fail(
                "comparison plan.tasks",
                "must be null when Harbor resolves an immutable dataset",
            )
        if raw_task_id_map is not None:
            raise _fail(
                "comparison plan.task_id_map",
                "must be null when Harbor resolves an immutable dataset",
            )
        tasks = ()
        task_id_map = {}
        dataset_root = None
    if tasks and tuple(sorted(set(tasks))) != tasks:
        raise _fail(
            "comparison plan.tasks",
            "must be a sorted unique list",
        )
    raw_arms = _array(plan.get("arms"), "comparison plan.arms")
    arms_by_digest: dict[str, dict[str, str]] = {}
    arm_ids: set[str] = set()
    for index, raw_arm in enumerate(raw_arms):
        location = f"comparison plan.arms[{index}]"
        arm = _object(raw_arm, location)
        expected_arm_fields = {
            "arm_id",
            "agent_config_name",
            "harbor_model_name",
            "agent_config_sha256",
            "canonical_harness",
            "canonical_model",
        }
        if set(arm) != expected_arm_fields:
            raise _fail(location, "unexpected or missing fields")
        arm_id = _string(arm.get("arm_id"), f"{location}.arm_id")
        if "\\" in arm_id or "\x00" in arm_id:
            raise _fail(f"{location}.arm_id", "contains an unsafe character")
        agent_config_name = _string(
            arm.get("agent_config_name"),
            f"{location}.agent_config_name",
        )
        harbor_model_name = _string(
            arm.get("harbor_model_name"),
            f"{location}.harbor_model_name",
        )
        agent_config_sha256 = _string(
            arm.get("agent_config_sha256"),
            f"{location}.agent_config_sha256",
        )
        if re.fullmatch(r"[0-9a-f]{64}", agent_config_sha256) is None:
            raise _fail(
                f"{location}.agent_config_sha256",
                "expected a lowercase SHA-256",
            )
        canonical_harness = _string(
            arm.get("canonical_harness"),
            f"{location}.canonical_harness",
        )
        canonical_model = _string(
            arm.get("canonical_model"),
            f"{location}.canonical_model",
        )
        if arm_id in arm_ids:
            raise _fail(f"{location}.arm_id", "duplicate comparison arm")
        if agent_config_sha256 in arms_by_digest:
            raise _fail(
                location,
                "duplicate rendered agent config digest",
            )
        arm_ids.add(arm_id)
        arms_by_digest[agent_config_sha256] = {
            "arm_id": arm_id,
            "agent_config_name": agent_config_name,
            "harbor_model_name": harbor_model_name,
            "agent_config_sha256": agent_config_sha256,
            "canonical_harness": canonical_harness,
            "canonical_model": canonical_model,
        }
    if not arms_by_digest:
        raise _fail("comparison plan.arms", "must not be empty")

    config_agents_by_digest: dict[str, dict[str, Any]] = {}
    config_digest_by_match_key: dict[str, str] = {}
    for index, raw_agent in enumerate(
        _array(job_config.get("agents"), "job config.agents")
    ):
        location = f"job config.agents[{index}]"
        agent = _object(raw_agent, location)
        agent_config_sha256 = canonical_agent_config_sha256(agent)
        match_key = _agent_config_match_key(agent)
        if (
            agent_config_sha256 in config_agents_by_digest
            or match_key in config_digest_by_match_key
        ):
            raise _fail(
                "job config.agents",
                "contains duplicate rendered agent configs",
            )
        config_agents_by_digest[agent_config_sha256] = agent
        config_digest_by_match_key[match_key] = agent_config_sha256
    if set(config_agents_by_digest) != set(arms_by_digest):
        raise _fail(
            "job config.agents",
            "rendered config digests do not match comparison plan arms",
        )
    for agent_config_sha256, agent in config_agents_by_digest.items():
        arm = arms_by_digest[agent_config_sha256]
        location = f"comparison arm {arm['arm_id']!r}"
        if (
            _agent_config_identity(agent, location)
            != arm["agent_config_name"]
            or _string(agent.get("model_name"), f"{location}.model_name")
            != arm["harbor_model_name"]
        ):
            raise _fail(
                location,
                "identity does not match the rendered Harbor agent config",
            )

    task_digests_by_name: dict[str, set[str]] = defaultdict(set)
    for index, raw_lock in enumerate(
        _array(job_lock.get("trials"), "job lock.trials")
    ):
        task = _object(
            _object(raw_lock, f"job lock.trials[{index}]").get("task"),
            f"job lock.trials[{index}].task",
        )
        task_name = (
            _local_task_selector(
                task,
                dataset_root=dataset_root,
                task_id_map=task_id_map,
                location=f"job lock.trials[{index}].task",
            )
            if tasks
            else _string(
                task.get("name"),
                f"job lock.trials[{index}].task.name",
            )
        )
        task_digests_by_name[task_name].add(
            _validate_digest(
                task.get("digest"),
                f"job lock.trials[{index}].task.digest",
            )
        )
    conflicting_task_digests = {
        task_name: digests
        for task_name, digests in task_digests_by_name.items()
        if len(digests) != 1
    }
    if conflicting_task_digests:
        raise _fail(
            "job lock.trials",
            "comparison arms resolved different immutable digests for the same task",
        )
    task_digest_by_name = {
        task_name: next(iter(digests))
        for task_name, digests in task_digests_by_name.items()
    }
    resolved_harbor_tasks = tuple(sorted(task_digest_by_name))
    if not resolved_harbor_tasks:
        raise _fail("job lock.trials", "resolved task set must not be empty")
    if tasks and resolved_harbor_tasks != tasks:
        raise _fail(
            "comparison plan.tasks",
            "does not match the resolved Harbor lock task set",
        )
    resolved_tasks = (
        tuple(sorted(task_id_map[task] for task in resolved_harbor_tasks))
        if tasks
        else resolved_harbor_tasks
    )
    expected_cells = Counter(
        (task, task_digest_by_name[task], agent_config_sha256)
        for task in resolved_harbor_tasks
        for agent_config_sha256 in arms_by_digest
        for _attempt in range(attempts)
    )
    actual_cells: Counter[tuple[str, str, str]] = Counter()
    for index, raw_lock in enumerate(
        _array(job_lock.get("trials"), "job lock.trials")
    ):
        lock = _object(raw_lock, f"job lock.trials[{index}]")
        task = _object(lock.get("task"), f"job lock.trials[{index}].task")
        agent = _object(lock.get("agent"), f"job lock.trials[{index}].agent")
        match_key = _agent_config_match_key(agent)
        agent_config_sha256 = config_digest_by_match_key.get(match_key)
        if agent_config_sha256 is None:
            raise _fail(
                f"job lock.trials[{index}].agent",
                "does not match any exact rendered agent config",
            )
        task_name = (
            _local_task_selector(
                task,
                dataset_root=dataset_root,
                task_id_map=task_id_map,
                location=f"job lock.trials[{index}].task",
            )
            if tasks
            else _string(
                task.get("name"),
                f"job lock.trials[{index}].task.name",
            )
        )
        actual_cells[
            (
                task_name,
                _validate_digest(
                    task.get("digest"),
                    f"job lock.trials[{index}].task.digest",
                ),
                agent_config_sha256,
            )
        ] += 1
    if actual_cells != expected_cells:
        raise _fail(
            "comparison plan",
            "task/arm/attempt matrix does not match immutable Harbor job lock",
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in trials:
        agent_config_sha256 = config_digest_by_match_key.get(
            item["agent_config_match_key"]
        )
        if agent_config_sha256 is None:
            raise _fail(
                f"trial {item['trial_name']!r}.lock.agent",
                "does not match any exact rendered agent config",
            )
        task_name = (
            _local_task_selector(
                item["task_lock"],
                dataset_root=dataset_root,
                task_id_map=task_id_map,
                location=f"trial {item['trial_name']!r}.lock.task",
            )
            if tasks
            else _string(
                item["task_lock"].get("name"),
                f"trial {item['trial_name']!r}.lock.task.name",
            )
        )
        if item["task_digest"] != task_digest_by_name.get(task_name):
            raise _fail(
                f"trial {item['trial_name']!r}.lock.task.digest",
                "does not match the immutable task digest in the Harbor job lock",
            )
        grouped[(task_name, agent_config_sha256)].append(item)
    coordinates: dict[str, dict[str, Any]] = {}
    for key, group_trials in grouped.items():
        harbor_task, agent_config_sha256 = key
        arm = arms_by_digest[agent_config_sha256]
        arm_id = arm["arm_id"]
        ordered = sorted(
            group_trials,
            key=lambda item: (item["trial_name"], item["trial_id"]),
        )
        if len(ordered) != attempts:
            raise _fail(
                "comparison plan",
                f"resolved cell count disagrees for {harbor_task!r} and {arm_id!r}",
            )
        for block_index, item in enumerate(ordered, 1):
            coordinates[item["trial_id"]] = {
                "arm_id": arm_id,
                "task": task_id_map.get(harbor_task, harbor_task),
                "index": block_index,
                "agent_config_sha256": agent_config_sha256,
                "canonical_harness": arm["canonical_harness"],
                "canonical_model": arm["canonical_model"],
            }
    if len(coordinates) != len(trials):
        raise _fail(
            "comparison plan",
            "did not assign every completed Harbor trial",
        )
    return _sha256_file(plan_path), plan, resolved_tasks, coordinates


def load_rows(
    job_dir: str | os.PathLike[str],
    *,
    comparison_plan_path: str | os.PathLike[str] | None = None,
    submitted_job_config_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate a complete Harbor 0.20.0 job and return normalized rows."""
    if (comparison_plan_path is None) != (submitted_job_config_path is None):
        raise _fail(
            "comparison identity",
            "comparison plan and submitted job config must be provided together",
        )
    requested_root = Path(job_dir).expanduser()
    if requested_root.is_symlink():
        raise _fail("job directory", f"symlink is not accepted: {requested_root}")
    root = requested_root.resolve()
    if not root.is_dir():
        raise _fail("job directory", f"not a directory: {root}")
    job_lock_path = root / "lock.json"
    job_result_path = root / "result.json"
    job_lock, expected_locks = _validate_job_lock(
        _read_json(job_lock_path, "job lock")
    )
    job_result_value = _read_json(job_result_path, "job result")
    expected_job_id = _string(
        _object(job_result_value, "job result").get("id"),
        "job result.id",
    )
    trial_dirs = _trial_directories(root)
    expected_count = sum(expected_locks.values())
    if len(trial_dirs) != expected_count:
        raise _fail(
            "job directory",
            f"incomplete or random job: expected {expected_count} trials, found {len(trial_dirs)}",
        )

    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    trials = [
        _validate_trial(
            path,
            expected_locks,
            seen_ids,
            seen_names,
            expected_job_id,
            comparison_plan_requested=comparison_plan_path is not None,
        )
        for path in trial_dirs
    ]
    if any(expected_locks.values()):
        raise _fail("job directory", "not every job-lock trial has completed evidence")
    comparison_plan_sha256 = None
    comparison_plan = None
    comparison_resolved_tasks: tuple[str, ...] | None = None
    comparison_coordinates: dict[str, dict[str, Any]] = {}
    if comparison_plan_path is not None:
        (
            comparison_plan_sha256,
            comparison_plan,
            comparison_resolved_tasks,
            comparison_coordinates,
        ) = _validate_comparison_plan(
            comparison_plan_path,
            submitted_job_config_path=submitted_job_config_path,
            job_root=root,
            job_lock=job_lock,
            trials=trials,
        )
    checksums_by_digest: dict[str, set[str]] = defaultdict(set)
    content_digests_by_task: dict[str, set[str]] = defaultdict(set)
    content_digests_by_harbor_digest: dict[str, set[str]] = defaultdict(set)
    for item in trials:
        checksums_by_digest[item["task_digest"]].add(item["task_checksum"])
        if item["openbench_task_content_digest"] is None:
            continue
        content_sha256 = item["openbench_task_content_digest"]["sha256"]
        comparison_coordinate = comparison_coordinates.get(item["trial_id"])
        content_task = (
            comparison_coordinate["task"]
            if comparison_coordinate is not None
            else _normalized_task_name(item["task_name"])
        )
        content_digests_by_task[content_task].add(content_sha256)
        content_digests_by_harbor_digest[item["task_digest"]].add(content_sha256)
    for task_digest, checksums in checksums_by_digest.items():
        if len(checksums) != 1:
            raise _fail(
                "trial results.task_checksum",
                f"inconsistent legacy checksums for locked task {task_digest}",
            )
    for task_name, content_digests in content_digests_by_task.items():
        if len(content_digests) != 1:
            raise _fail(
                "trial results.openbench_task_content_digest",
                f"inconsistent OpenBench content digests for task {task_name!r}",
            )
    for task_digest, content_digests in content_digests_by_harbor_digest.items():
        if len(content_digests) != 1:
            raise _fail(
                "trial results.openbench_task_content_digest",
                f"inconsistent OpenBench content digests for locked task {task_digest}",
            )

    job_result, job_id, job_started, job_finished = _validate_job_result(
        job_result_value,
        expected_trials=expected_count,
        actual_trial_results=[item["result"] for item in trials],
    )
    job_retry_count = _integer(
        _object(job_result.get("stats"), "job result.stats").get("n_retries"),
        "job result.stats.n_retries",
    )
    job_max_retries = _integer(
        _object(job_lock.get("retry"), "job lock.retry").get("max_retries"),
        "job lock.retry.max_retries",
    )
    for item in trials:
        if (
            job_started.tzinfo is not None
            and not (
                job_started
                <= item["started"]
                <= item["finished"]
                <= job_finished
            )
        ):
            raise _fail(
                f"trial {item['trial_name']!r}",
                "trial timing falls outside job timing",
            )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in trials:
        comparison_coordinate = comparison_coordinates.get(item["trial_id"])
        if comparison_coordinate is None:
            key = (
                _normalized_task_name(item["task_name"]),
                item["agent_name"],
                item["model"],
                "",
            )
        else:
            key = (
                comparison_coordinate["task"],
                comparison_coordinate["canonical_harness"],
                comparison_coordinate["canonical_model"],
                comparison_coordinate["agent_config_sha256"],
            )
        grouped[key].append(item)

    job_lock_digest = _sha256_file(job_lock_path)
    job_result_digest = _sha256_file(job_result_path)
    rows: list[dict[str, Any]] = []
    for group in sorted(grouped):
        task, harness, model, _variant_digest = group
        group_trials = sorted(
            grouped[group],
            key=lambda item: (item["trial_name"], item["trial_id"]),
        )
        for trial_number, item in enumerate(group_trials, 1):
            comparison_coordinate = comparison_coordinates.get(item["trial_id"])
            if comparison_coordinate is not None:
                trial_number = comparison_coordinate["index"]
            score = item["score"]
            exception = item["exception"]
            terminal_failure = exception is not None
            success = not terminal_failure and item["checker_exit"] == 0
            if terminal_failure:
                exception_type, failure_class, failure_reason = exception
            else:
                exception_type = None
                failure_class = "solved" if success else "wrong_answer"
                failure_reason = None
            provenance = {
                "kind": "harbor_job",
                "harbor_version": HARBOR_VERSION,
                "harbor_git_commit_hash": HARBOR_GIT_COMMIT,
                "harbor_job_id": job_id,
                "harbor_trial_id": item["trial_id"],
                "harbor_trial_name": item["trial_name"],
                "job_lock_sha256": job_lock_digest,
                "job_result_sha256": job_result_digest,
                "trial_lock_sha256": _sha256_file(item["trial_lock_path"]),
                "trial_result_sha256": _sha256_file(item["result_path"]),
                "reward_sha256": _optional_sha256_file(item["reward_path"]),
                "openbench_verifier_evidence_sha256": _optional_sha256_file(
                    item["verifier_evidence_path"]
                ),
                "atif_sha256": _optional_sha256_file(item["trajectory_path"]),
                "artifact_manifest_sha256": _optional_sha256_file(
                    item["manifest_path"]
                ),
                "final_workspace_sha256": item["workspace_digest"],
                "task_digest": item["task_digest"],
                "openbench_task_content_digest": (
                    None
                    if item["openbench_task_content_digest"] is None
                    else dict(item["openbench_task_content_digest"])
                ),
                "openbench_harbor_export": (
                    None
                    if item["openbench_harbor_export"] is None
                    else dict(item["openbench_harbor_export"])
                ),
                "harbor_task_checksum": item["task_checksum"],
                "harbor_agent_config_name": item["agent_config_name"],
                "harbor_model_name": item["harbor_model_name"],
                "agent_config_sha256": (
                    comparison_coordinate["agent_config_sha256"]
                    if comparison_coordinate is not None
                    else None
                ),
                "harbor_verifier_time_s": item["harbor_verifier_time_s"],
                "harbor_job_retries": job_retry_count,
                "harbor_job_max_retries": job_max_retries,
                "harbor_exception_type": exception_type,
                "comparison_plan_schema_version": (
                    COMPARISON_PLAN_SCHEMA_VERSION
                    if comparison_coordinate is not None
                    else None
                ),
                "comparison_plan_sha256": (
                    comparison_plan_sha256
                    if comparison_coordinate is not None
                    else None
                ),
                "comparison_plan": (
                    dict(comparison_plan)
                    if comparison_coordinate is not None
                    else None
                ),
                "comparison_arm_id": (
                    comparison_coordinate["arm_id"]
                    if comparison_coordinate is not None
                    else None
                ),
                "comparison_resolved_tasks": (
                    list(comparison_resolved_tasks)
                    if comparison_coordinate is not None
                    else None
                ),
                "comparison_block": (
                    {
                        "task": comparison_coordinate["task"],
                        "index": comparison_coordinate["index"],
                    }
                    if comparison_coordinate is not None
                    else None
                ),
                "usage_source": item["usage"]["token_basis"],
                "proxy_measured": False,
                "harbor_metering": None,
                "trial_mapping": (
                    "openbench_comparison_plan_v4"
                    if comparison_coordinate is not None
                    else "lexicographic_name_within_task_agent_model"
                ),
                "temporal_matched_block_claim": False,
            }
            row = {field: None for field in ROW_FIELDS}
            row.update(
                {
                    "run_id": make_run_id(
                        harness,
                        task,
                        model,
                        trial_number,
                        candidate_digest=(
                            comparison_coordinate["agent_config_sha256"]
                            if comparison_coordinate is not None
                            else None
                        ),
                        full_candidate_digest=comparison_coordinate is not None,
                    ),
                    "ts_iso": item["started"].isoformat(),
                    "harness": harness,
                    "model": model,
                    "task": task,
                    "trial": trial_number,
                    "success": success,
                    "completed": not terminal_failure,
                    "error": (
                        None
                        if exception_type is None
                        else f"Harbor terminal failure: {exception_type}"
                    ),
                    "wall_time_s": round(
                        (item["finished"] - item["started"]).total_seconds(), 3
                    ),
                    "t_env_setup_s": item["t_env_setup_s"],
                    "t_agent_s": item["t_agent_s"],
                    "t_checker_s": item["t_checker_s"],
                    "turns": item["turns"],
                    "exec_mode": "harbor",
                    "score": score,
                    "checker_exit": item["checker_exit"],
                    "harness_version": item["agent_version"],
                    "harness_version_source": "harbor_trial_result",
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "candidate_provenance": provenance,
                    "version_drift": False,
                    "workspace_source": (
                        None
                        if item["workspace_digest"] is None
                        else {
                            "kind": "harbor_artifact",
                            "sha256": item["workspace_digest"],
                        }
                    ),
                }
            )
            row.update(item["usage"])
            if item["metering_evidence"] is not None:
                row = apply_to_imported_row(
                    row,
                    item["metering_evidence"],
                    proxy_required=item["proxy_required"],
                )
                harbor_metering = row["candidate_provenance"]["harbor_metering"]
                harbor_metering.update(
                    {
                        "proxy_required": item["proxy_required"],
                        "evidence_sha256": _sha256_file(
                            item["metering_evidence_path"]
                        ),
                        "ledger_sha256": item["metering_evidence"][
                            "ledger_seal"
                        ]["ledger_sha256"],
                    }
                )
            else:
                grade, ranking_eligible, exclusion_reason = harbor_usage_policy(
                    row.get("token_basis"),
                    proxy_required=False,
                )
                row["usage_evidence_grade"] = grade
                row["usage_ranking_eligible"] = ranking_eligible
                row["usage_ranking_exclusion_reason"] = exclusion_reason
            rows.append(row)
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise _fail("normalized rows", "duplicate run_id after deterministic mapping")
    return rows


def import_results(
    job_dir: str | os.PathLike[str],
    results_path: str | os.PathLike[str],
    *,
    comparison_plan_path: str | os.PathLike[str] | None = None,
    submitted_job_config_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate, lock, collision-check, and append one rollback-safe batch."""
    rows = load_rows(
        job_dir,
        comparison_plan_path=comparison_plan_path,
        submitted_job_config_path=submitted_job_config_path,
    )
    requested_output = Path(results_path).expanduser()
    if requested_output.is_symlink():
        raise _fail("output", f"symlink is not accepted: {requested_output}")
    output = requested_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps({key: row.get(key) for key in ROW_FIELDS}) + "\n"
        for row in rows
    ).encode("utf-8")
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        with results_file_lock(output):
            try:
                descriptor = os.open(output, flags, 0o666)
            except OSError as exc:
                raise _fail("output", f"cannot open {output}: {exc}") from exc
            with os.fdopen(descriptor, "r+b", buffering=0) as handle:
                handle.seek(0)
                existing_bytes = handle.read()
                existing = _run_ids_from_jsonl(existing_bytes)
                collisions = sorted(
                    existing.intersection(row["run_id"] for row in rows)
                )
                if collisions:
                    raise _fail(
                        "output", f"run_id already exists: {collisions[0]!r}"
                    )
                original_size = len(existing_bytes)
                try:
                    handle.seek(0, os.SEEK_END)
                    written = 0
                    while written < len(payload):
                        count = handle.write(payload[written:])
                        if not count:
                            raise OSError(
                                "short write while appending Harbor result batch"
                            )
                        written += count
                    os.fsync(handle.fileno())
                except OSError as exc:
                    os.ftruncate(handle.fileno(), original_size)
                    os.fsync(handle.fileno())
                    raise _fail(
                        "output", f"append failed and was rolled back: {exc}"
                    ) from exc
                except BaseException:
                    os.ftruncate(handle.fileno(), original_size)
                    os.fsync(handle.fileno())
                    raise
    except ResultsLogError as exc:
        raise _fail("output", str(exc)) from exc
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="obench import harbor-results",
        description="Import a complete Harbor 0.20.0 job as OpenBench JSONL.",
    )
    parser.add_argument("job_dir", help="completed Harbor job directory")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="OpenBench results JSONL to append",
    )
    parser.add_argument(
        "--comparison-plan",
        help=(
            "OpenBench comparison-plan sidecar emitted beside the Harbor "
            "job config"
        ),
    )
    parser.add_argument(
        "--job-config",
        help=(
            "exact OpenBench-submitted Harbor job config; required with "
            "--comparison-plan"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = import_results(
            args.job_dir,
            args.output,
            comparison_plan_path=args.comparison_plan,
            submitted_job_config_path=args.job_config,
        )
    except HarborResultsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Imported {len(rows)} Harbor trial(s) into {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
