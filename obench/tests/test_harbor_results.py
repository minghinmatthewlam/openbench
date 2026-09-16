#!/usr/bin/env python3
"""Golden Harbor 0.20.0 job-result ingestion tests."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from obench import harbor_results, run, stats
from obench.harbor_job import (
    COMPARISON_PLAN_SCHEMA_VERSION,
    canonical_agent_config_sha256,
    canonical_harbor_job_config_sha256,
)
from obench.harbor_results import (
    HARBOR_GIT_COMMIT,
    HARBOR_VERSION,
    HarborResultsError,
    _tree_digest,
    import_results,
    load_rows,
)
from obench.harbor_oauth import AGENT_IMPORT_PATH
from obench.harbor_profiles import (
    CURSOR_PROFILE_IMPORT,
    DEVIN_PROFILE_IMPORT,
    OPENCODE_PROFILE_IMPORT,
)
from obench.run import ROW_FIELDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(_REPO_ROOT)
    + os.pathsep
    + os.environ.get("PYTHONPATH", ""),
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _comparison_arm(
    arm_id: str,
    agent: dict,
    *,
    canonical_harness: str | None = None,
    canonical_model: str | None = None,
) -> dict:
    return {
        "arm_id": arm_id,
        "agent_config_name": agent.get("name") or agent["import_path"],
        "harbor_model_name": agent["model_name"],
        "agent_config_sha256": canonical_agent_config_sha256(agent),
        "canonical_harness": canonical_harness or arm_id,
        "canonical_model": canonical_model or agent["model_name"],
    }


def _openbench_task_content_digest(task: str) -> dict:
    return {
        "scheme": 2,
        "sha256": ("1" if task == "alpha" else "2") * 64,
    }


def _trial_lock(task: str, task_root: Path, model: str = "model-x") -> dict:
    return {
        "schema_version": 2,
        "task": {
            "name": f"openbench/{task}",
            "type": "local",
            "digest": "sha256:" + ("a" if task == "alpha" else "b") * 64,
            "path": str(task_root / task),
        },
        "install_only": False,
        "timeout_multiplier": 1.0,
        "agent": {
            "name": "codex",
            "model_name": model,
            "skills": [],
            "resume_trajectory": False,
            "extra_allowed_hosts": [],
            "kwargs": {},
            "mcp_servers": [],
        },
        "skills": [],
        "environment": {
            "type": "docker",
            "force_build": False,
            "delete": True,
            "cpu_enforcement_policy": "auto",
            "memory_enforcement_policy": "auto",
            "extra_docker_compose": [],
            "kwargs": {},
            "extra_allowed_hosts": [],
        },
        "verifier": {"disable": False, "environment_mode": "shared"},
    }


def _trial_result(
    name: str,
    task: str,
    task_root: Path,
    trial_id: str,
    score: float,
    *,
    model: str = "model-x",
    offset_minutes: int = 0,
) -> dict:
    minute = offset_minutes
    return {
        "id": trial_id,
        "task_name": f"openbench/{task}",
        "trial_name": name,
        "trial_uri": f"file:///tmp/job/{name}",
        "task_id": {"path": str(task_root / task)},
        "source": "openbench",
        "task_checksum": ("a" if task == "alpha" else "b") * 64,
        "config": {
            "task": {"path": str(task_root / task)},
            "trial_name": name,
            "trials_dir": "/tmp/job",
            "job_id": "10000000-0000-0000-0000-000000000000",
            "agent": {"name": "codex", "model_name": model, "kwargs": {}},
            "environment": {"type": "docker", "kwargs": {}},
            "verifier": {"disable": False, "kwargs": {}},
            "artifacts": [],
            "extra_instruction_paths": [],
        },
        "agent_info": {
            "name": "codex",
            "version": "1.2.3",
            "model_info": {"name": model, "provider": None},
        },
        "agent_result": {
            "n_input_tokens": 100,
            "n_cache_tokens": 25,
            "n_output_tokens": 40,
            "cost_usd": 0.5,
        },
        "verifier_result": {"rewards": {"reward": score}},
        "verifier_environment_mode": "shared",
        "exception_info": None,
        "started_at": f"2026-07-20T10:{minute:02d}:00+00:00",
        "finished_at": f"2026-07-20T10:{minute + 4:02d}:00+00:00",
        "environment_setup": {
            "started_at": f"2026-07-20T10:{minute:02d}:00+00:00",
            "finished_at": f"2026-07-20T10:{minute:02d}:30+00:00",
        },
        "agent_setup": {
            "started_at": f"2026-07-20T10:{minute:02d}:30+00:00",
            "finished_at": f"2026-07-20T10:{minute + 1:02d}:00+00:00",
        },
        "agent_execution": {
            "started_at": f"2026-07-20T10:{minute + 1:02d}:00+00:00",
            "finished_at": f"2026-07-20T10:{minute + 3:02d}:00+00:00",
        },
        "verifier": {
            "started_at": f"2026-07-20T10:{minute + 3:02d}:00+00:00",
            "finished_at": f"2026-07-20T10:{minute + 4:02d}:00+00:00",
        },
        "step_results": None,
    }


def _trajectory(model: str = "model-x") -> dict:
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "codex", "version": "1.2.3"},
        "steps": [
            {
                "step_id": 1,
                "source": "user",
                "message": "Complete the task.",
                "timestamp": "2026-07-20T10:01:00+00:00",
            },
            {
                "step_id": 2,
                "source": "agent",
                "message": "Done.",
                "timestamp": "2026-07-20T10:03:00+00:00",
                "model_name": model,
                "metrics": {
                    "prompt_tokens": 100,
                    "cached_tokens": 25,
                    "completion_tokens": 40,
                    "cost_usd": 0.5,
                },
            },
        ],
        "final_metrics": {
            "total_steps": 2,
            "total_prompt_tokens": 100,
            "total_cached_tokens": 25,
            "total_completion_tokens": 40,
            "total_cost_usd": 0.5,
        },
    }


class GoldenHarborJob:
    """Synthetic fixture shaped from Harbor v0.20.0 model serialization."""

    def __init__(self, root: Path, specs: list[dict] | None = None):
        self.root = root
        self.task_root = root.parent / f"{root.name}-tasks"
        self.submitted_config_path = root.parent / f"{root.name}-submitted.json"
        self.specs = specs or [
            {
                "name": "alpha__zeta",
                "task": "alpha",
                "id": "00000000-0000-0000-0000-000000000002",
                "score": 0.0,
                "offset": 5,
            },
            {
                "name": "alpha__able",
                "task": "alpha",
                "id": "00000000-0000-0000-0000-000000000001",
                "score": 1.0,
                "offset": 0,
            },
            {
                "name": "beta__only",
                "task": "beta",
                "id": "00000000-0000-0000-0000-000000000003",
                "score": 0.5,
                "offset": 10,
            },
        ]
        self.write()

    def write(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.task_root.mkdir(parents=True, exist_ok=True)
        for task in sorted({spec["task"] for spec in self.specs}):
            task_dir = self.task_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.toml").write_text(
                'schema_version = "1.4"\n\n'
                "[task]\n"
                f'name = "{task}"\n'
            )
        locks = [
            _trial_lock(spec["task"], self.task_root)
            for spec in self.specs
        ]
        _write_json(
            self.root / "lock.json",
            {
                "schema_version": 3,
                "created_at": "2026-07-20T09:59:00+00:00",
                "harbor": {
                    "version": HARBOR_VERSION,
                    "git_commit_hash": HARBOR_GIT_COMMIT,
                    "is_editable": False,
                },
                "n_concurrent_trials": 2,
                "retry": {
                    "max_retries": 0,
                    "include_exceptions": None,
                    "exclude_exceptions": None,
                },
                "trials": locks,
            },
        )
        results = []
        for spec, lock in zip(self.specs, locks):
            trial_dir = self.root / spec["name"]
            result = _trial_result(
                spec["name"],
                spec["task"],
                self.task_root,
                spec["id"],
                spec["score"],
                offset_minutes=spec["offset"],
            )
            _write_json(trial_dir / "lock.json", lock)
            _write_json(trial_dir / "result.json", result)
            _write_json(trial_dir / "agent" / "trajectory.json", _trajectory())
            (trial_dir / "verifier").mkdir(parents=True, exist_ok=True)
            (trial_dir / "verifier" / "reward.txt").write_text(
                f"{spec['score']}\n"
            )
            _write_json(
                trial_dir
                / "verifier"
                / "openbench-verifier-evidence.json",
                {
                    "schema_version": "openbench-verifier-evidence-v2",
                    "openbench_task_content_digest": (
                        _openbench_task_content_digest(spec["task"])
                    ),
                    "openbench_harbor_export": {
                        "schema_version": 1,
                        "base_image": "python:3.11-slim",
                        "network_mode": "no-network",
                    },
                    "checker_exit": 0 if spec["score"] == 1.0 else 1,
                    "parsed_score": None if spec["score"] == 0.0 else spec["score"],
                    "reward": spec["score"],
                    "verifier_duration_seconds": 60,
                },
            )
            _write_json(
                trial_dir / "artifacts" / "manifest.json",
                [
                    {
                        "source": "/logs/artifacts",
                        "destination": "artifacts/logs/artifacts",
                        "type": "directory",
                        "status": "empty",
                        "service": None,
                    },
                    {
                        "source": "/app",
                        "destination": "artifacts/workspace",
                        "type": "directory",
                        "status": "ok",
                        "service": None,
                    },
                ],
            )
            workspace = trial_dir / "artifacts" / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "answer.txt").write_text(f"{spec['name']}\n")
            results.append(result)
        _write_json(
            self.root / "result.json",
            {
                "id": "10000000-0000-0000-0000-000000000000",
                "started_at": "2026-07-20T09:59:00+00:00",
                "updated_at": "2026-07-20T10:14:00+00:00",
                "finished_at": "2026-07-20T10:15:00+00:00",
                "n_total_trials": len(results),
                "stats": self._stats(results),
            },
        )

    def trial(self, index: int = 0) -> Path:
        return self.root / self.specs[index]["name"]

    def sync_aggregate(self) -> None:
        job_result_path = self.root / "result.json"
        job_result = json.loads(job_result_path.read_text())
        results = [
            json.loads((self.trial(index) / "result.json").read_text())
            for index in range(len(self.specs))
        ]
        job_result["stats"] = self._stats(results)
        _write_json(job_result_path, job_result)

    def make_terminal(
        self,
        exception_type: str,
        *,
        index: int = 0,
        retain_optional_evidence: bool = False,
    ) -> None:
        trial_dir = self.trial(index)
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["exception_info"] = {
            "exception_type": exception_type,
            "exception_message": "terminal failure",
            "exception_traceback": "traceback",
            "occurred_at": result["finished_at"],
        }
        if not retain_optional_evidence:
            result["agent_result"] = None
            result["verifier_result"] = None
            result["agent_execution"] = None
            result["verifier"] = None
            shutil.rmtree(trial_dir / "agent")
            shutil.rmtree(trial_dir / "verifier")
            shutil.rmtree(trial_dir / "artifacts")
        _write_json(result_path, result)
        self.sync_aggregate()

    def set_builtin_agent(
        self,
        index: int,
        agent_name: str,
        *,
        model_name: str = "model-x",
    ) -> None:
        trial_dir = self.trial(index)
        trial_lock_path = trial_dir / "lock.json"
        trial_lock = json.loads(trial_lock_path.read_text())
        trial_lock["agent"]["name"] = agent_name
        trial_lock["agent"]["model_name"] = model_name
        _write_json(trial_lock_path, trial_lock)

        job_lock_path = self.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"][index] = trial_lock
        _write_json(job_lock_path, job_lock)

        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["name"] = agent_name
        result["config"]["agent"]["model_name"] = model_name
        result["agent_info"]["name"] = agent_name
        result["agent_info"]["model_info"] = {
            "name": model_name,
            "provider": None,
        }
        _write_json(result_path, result)

        trajectory_path = trial_dir / "agent" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["agent"]["name"] = agent_name
        trajectory["steps"][1]["model_name"] = model_name
        _write_json(trajectory_path, trajectory)

    def set_custom_agent(
        self,
        index: int,
        agent: dict,
        *,
        reported_name: str,
    ) -> None:
        trial_dir = self.trial(index)
        lock_path = trial_dir / "lock.json"
        lock = json.loads(lock_path.read_text())
        lock_agent = {
            "import_path": agent["import_path"],
            "model_name": agent["model_name"],
            "skills": [],
            "resume_trajectory": False,
            "extra_allowed_hosts": agent.get("extra_allowed_hosts", []),
            "kwargs": agent.get("kwargs", {}),
            "env": agent.get("env", {}),
            "mcp_servers": [],
        }
        for field in ("n_concurrent", "concurrency_group"):
            if field in agent:
                lock_agent[field] = agent[field]
        lock["agent"] = lock_agent
        _write_json(lock_path, lock)

        job_lock_path = self.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"][index] = lock
        _write_json(job_lock_path, job_lock)

        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"] = dict(agent)
        result["agent_info"]["name"] = reported_name
        result["agent_info"]["model_info"] = {
            "name": agent["model_name"],
            "provider": None,
        }
        _write_json(result_path, result)

        trajectory_path = trial_dir / "agent" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["agent"]["name"] = reported_name
        trajectory["agent"]["model_name"] = agent["model_name"]
        trajectory["steps"][1]["model_name"] = agent["model_name"]
        _write_json(trajectory_path, trajectory)

    def write_comparison_plan(
        self,
        *,
        attempts: int,
        arms: list[dict[str, str]],
        agents: list[dict] | None = None,
        dataset_descriptor: dict | None = None,
        task_id_map: dict[str, str] | None = None,
    ) -> Path:
        tasks = sorted({spec["task"] for spec in self.specs})
        if agents is None:
            agents = [
                {
                    "name": arm["agent_config_name"],
                    "model_name": arm["harbor_model_name"],
                }
                for arm in arms
            ]
        config = {
            "job_name": self.root.name,
            "n_attempts": attempts,
            "agents": agents,
            "datasets": [
                dataset_descriptor
                or {"path": str(self.task_root), "task_names": tasks}
            ],
            "tasks": [],
        }
        config_path = self.root / "config.json"
        _write_json(config_path, config)
        _write_json(self.submitted_config_path, config)
        plan = {
            "schema_version": COMPARISON_PLAN_SCHEMA_VERSION,
            "harbor_version": HARBOR_VERSION,
            "harbor_git_commit_hash": HARBOR_GIT_COMMIT,
            "job_name": self.root.name,
            "submitted_job_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "effective_job_config_sha256": (
                canonical_harbor_job_config_sha256(config)
            ),
            "attempts": attempts,
            "dataset": dataset_descriptor,
            "tasks": None if dataset_descriptor is not None else tasks,
            "task_id_map": (
                None
                if dataset_descriptor is not None
                else task_id_map or {task: task for task in tasks}
            ),
            "arms": arms,
        }
        if dataset_descriptor is None:
            for selector, canonical_id in plan["task_id_map"].items():
                (self.task_root / selector / "task.toml").write_text(
                    'schema_version = "1.4"\n\n'
                    "[task]\n"
                    f'name = "{canonical_id}"\n'
                )
        path = self.root.parent / f"{self.root.name}-comparison-plan.json"
        _write_json(path, plan)
        return path

    @staticmethod
    def _stats(results: list[dict]) -> dict:
        reward_stats: dict[str, list[str]] = {}
        exception_stats: dict[str, list[str]] = {}
        usage_totals = {
            "n_input_tokens": [],
            "n_cache_tokens": [],
            "n_output_tokens": [],
            "cost_usd": [],
        }
        for result in results:
            verifier_result = result.get("verifier_result")
            if verifier_result is not None:
                score = verifier_result["rewards"]["reward"]
                reward_stats.setdefault(str(score), []).append(result["trial_name"])
            exception = result.get("exception_info")
            if exception is not None:
                exception_stats.setdefault(
                    exception["exception_type"], []
                ).append(result["trial_name"])
            agent_result = result.get("agent_result")
            if isinstance(agent_result, dict):
                for field in usage_totals:
                    if agent_result.get(field) is not None:
                        usage_totals[field].append(agent_result[field])
        n_errors = sum(len(names) for names in exception_stats.values())
        return {
            "n_completed_trials": len(results),
            "n_errored_trials": n_errors,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": len(exception_stats.get("CancelledError", [])),
            "n_retries": 0,
            "evals": {
                "codex__model-x__openbench": {
                    "n_trials": sum(len(names) for names in reward_stats.values()),
                    "n_errors": n_errors,
                    "metrics": [],
                    "pass_at_k": {},
                    "reward_stats": {"reward": reward_stats},
                    "exception_stats": exception_stats,
                }
            },
            **{
                field: (sum(values) if values else None)
                for field, values in usage_totals.items()
            },
        }


class HarborResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self) -> GoldenHarborJob:
        return GoldenHarborJob(self.root / "job")

    def custom_environment_fixture(self, name, *, null_in_lock=False):
        fixture = GoldenHarborJob(self.root / name)
        job_lock = json.loads((fixture.root / "lock.json").read_text())
        for index in range(len(fixture.specs)):
            lock_path = fixture.trial(index) / "lock.json"
            lock = json.loads(lock_path.read_text())
            lock["environment"] = {"import_path": "example.environment:CustomEnvironment"}
            result_path = fixture.trial(index) / "result.json"
            result = json.loads(result_path.read_text())
            result["config"]["environment"] = dict(lock["environment"])
            if null_in_lock:
                lock["environment"]["type"] = None
            else:
                result["config"]["environment"]["type"] = None
            job_lock["trials"][index] = lock
            _write_json(lock_path, lock)
            _write_json(result_path, result)
        _write_json(fixture.root / "lock.json", job_lock)
        fixture.sync_aggregate()
        return fixture

    def test_custom_environment_accepts_omitted_and_null_type(self):
        for null_in_lock in (False, True):
            with self.subTest(null_in_lock=null_in_lock):
                fixture = self.custom_environment_fixture(
                    f"custom-{null_in_lock}", null_in_lock=null_in_lock)
                self.assertEqual(len(load_rows(fixture.root)), len(fixture.specs))

    def test_custom_environment_still_rejects_selector_and_import_path_drift(self):
        for key, value in (("type", "docker"), ("import_path", "example.environment:OtherEnvironment")):
            with self.subTest(field=key):
                fixture = self.custom_environment_fixture(f"drift-{key}")
                self.assertEqual(len(load_rows(fixture.root)), len(fixture.specs))
                result_path = fixture.trial() / "result.json"
                result = json.loads(result_path.read_text())
                result["config"]["environment"][key] = value
                _write_json(result_path, result)
                fixture.sync_aggregate()
                with self.assertRaisesRegex(HarborResultsError, r"result.config.environment.*does not match trial lock"):
                    load_rows(fixture.root)

    def qualified_model_fixture(
        self,
        name: str,
        *,
        top_level_model: str,
        step_model: str,
    ) -> GoldenHarborJob:
        fixture = GoldenHarborJob(
            self.root / name,
            specs=[
                {
                    "name": "alpha__qualified",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000009",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["agent"]["model_name"] = "openai/gpt-5.6-sol"
        _write_json(lock_path, trial_lock)

        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"][0] = trial_lock
        _write_json(job_lock_path, job_lock)

        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["model_name"] = "openai/gpt-5.6-sol"
        result["agent_info"]["model_info"] = {
            "name": "gpt-5.6-sol",
            "provider": "openai",
        }
        _write_json(result_path, result)

        trajectory_path = fixture.trial() / "agent" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["agent"]["model_name"] = top_level_model
        trajectory["steps"][1]["model_name"] = step_model
        _write_json(trajectory_path, trajectory)
        return fixture

    def test_golden_job_maps_groups_deterministically_without_temporal_claim(self):
        fixture = self.fixture()
        rows = load_rows(fixture.root)

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [(row["task"], row["trial"]) for row in rows],
            [("alpha", 1), ("alpha", 2), ("beta", 1)],
        )
        self.assertEqual(
            rows[0]["candidate_provenance"]["harbor_trial_name"], "alpha__able"
        )
        self.assertEqual(
            rows[1]["candidate_provenance"]["harbor_trial_name"], "alpha__zeta"
        )
        self.assertFalse(
            rows[0]["candidate_provenance"]["temporal_matched_block_claim"]
        )
        self.assertEqual(rows[0]["exec_mode"], "harbor")
        self.assertEqual(rows[0]["token_basis"], "harbor_agent_reported")
        self.assertEqual(rows[0]["usage_evidence_grade"], "harbor_reported")
        self.assertTrue(rows[0]["usage_ranking_eligible"])
        self.assertIsNone(rows[0]["usage_ranking_exclusion_reason"])
        self.assertEqual(rows[0]["tokens_input_uncached"], 75)
        self.assertEqual(rows[0]["tokens_cache_read"], 25)
        self.assertEqual(rows[0]["tokens_output"], 40)
        self.assertEqual(rows[0]["tokens"], 115)
        self.assertIsNone(rows[0]["token_basis_proxy"])
        self.assertEqual(
            rows[0]["candidate_provenance"]["harbor_job_retries"],
            0,
        )
        self.assertEqual(
            rows[0]["candidate_provenance"]["harbor_job_max_retries"],
            0,
        )
        self.assertEqual(
            rows[0]["candidate_provenance"]["harbor_model_name"],
            "model-x",
        )
        self.assertIsNone(rows[0]["tokens_proxy_calls"])
        self.assertFalse(rows[0]["candidate_provenance"]["proxy_measured"])
        self.assertEqual(
            rows[0]["candidate_provenance"]["openbench_task_content_digest"],
            _openbench_task_content_digest("alpha"),
        )
        self.assertEqual(
            rows[0]["candidate_provenance"]["openbench_harbor_export"],
            {
                "schema_version": 1,
                "base_image": "python:3.11-slim",
                "network_mode": "no-network",
            },
        )
        self.assertEqual(rows[0]["failure_class"], "solved")
        self.assertEqual(rows[1]["failure_class"], "wrong_answer")
        self.assertEqual(rows[2]["score"], 0.5)
        self.assertEqual(rows[2]["failure_class"], "wrong_answer")
        for row in rows:
            self.assertEqual(set(row), set(ROW_FIELDS))
            self.assertIsInstance(row["checker_exit"], int)
            self.assertIsNone(row["checker_stdout"])
            self.assertIsNone(row["checker_stderr"])

    def test_comparison_plan_binds_every_arm_to_shared_block_coordinates(self):
        fixture = GoldenHarborJob(
            self.root / "comparison-job",
            specs=[
                {
                    "name": "alpha__codex_zeta",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000011",
                    "score": 1.0,
                    "offset": 0,
                },
                {
                    "name": "alpha__codex_able",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000012",
                    "score": 0.0,
                    "offset": 2,
                },
                {
                    "name": "alpha__pi_able",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000013",
                    "score": 1.0,
                    "offset": 5,
                },
                {
                    "name": "alpha__pi_zeta",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000014",
                    "score": 0.0,
                    "offset": 7,
                },
            ],
        )
        fixture.set_builtin_agent(2, "pi")
        fixture.set_builtin_agent(3, "pi")
        arms = [
            _comparison_arm(
                "codex",
                {"name": "codex", "model_name": "model-x"},
            ),
            _comparison_arm(
                "pi",
                {"name": "pi", "model_name": "model-x"},
            ),
        ]
        plan_path = fixture.write_comparison_plan(
            attempts=2,
            arms=arms,
        )

        rows = load_rows(
            fixture.root,
            comparison_plan_path=plan_path,
            submitted_job_config_path=fixture.submitted_config_path,
        )

        plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        by_harness = {
            harness: sorted(
                (row for row in rows if row["harness"] == harness),
                key=lambda row: row["trial"],
            )
            for harness in ("codex", "pi")
        }
        for harness, arm_rows in by_harness.items():
            self.assertEqual(
                [row["trial"] for row in arm_rows],
                [1, 2],
            )
            self.assertEqual(
                [
                    row["candidate_provenance"]["comparison_block"]
                    for row in arm_rows
                ],
                [
                    {"task": "alpha", "index": 1},
                    {"task": "alpha", "index": 2},
                ],
            )
            for row in arm_rows:
                provenance = row["candidate_provenance"]
                self.assertEqual(
                    provenance["comparison_plan_schema_version"],
                    COMPARISON_PLAN_SCHEMA_VERSION,
                )
                self.assertEqual(
                    provenance["comparison_plan_sha256"],
                    plan_sha256,
                )
                self.assertEqual(provenance["comparison_arm_id"], harness)
                self.assertEqual(
                    provenance["trial_mapping"],
                    "openbench_comparison_plan_v4",
                )
                self.assertFalse(
                    provenance["temporal_matched_block_claim"]
                )

        self.assertEqual(
            by_harness["codex"][0]["candidate_provenance"][
                "harbor_trial_name"
            ],
            "alpha__codex_able",
        )
        self.assertEqual(
            by_harness["pi"][0]["candidate_provenance"]["harbor_trial_name"],
            "alpha__pi_able",
        )

    def test_comparison_plan_rejects_config_or_lock_matrix_drift(self):
        fixture = GoldenHarborJob(
            self.root / "comparison-drift",
            specs=[
                {
                    "name": "alpha__one",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000021",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        arms = [
            _comparison_arm(
                "codex",
                {"name": "codex", "model_name": "model-x"},
            )
        ]
        plan_path = fixture.write_comparison_plan(
            attempts=1,
            arms=arms,
        )
        config_path = fixture.root / "config.json"
        config = json.loads(config_path.read_text())
        config["n_attempts"] = 2
        _write_json(config_path, config)
        with self.assertRaisesRegex(
            HarborResultsError,
            "persisted semantic job config",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )
        plan_path = fixture.write_comparison_plan(
            attempts=2,
            arms=arms,
        )
        with self.assertRaisesRegex(
            HarborResultsError,
            "matrix does not match immutable Harbor job lock",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

    def test_comparison_plan_rejects_v2_and_submitted_config_tampering(self):
        fixture = GoldenHarborJob(
            self.root / "comparison-identity",
            specs=[
                {
                    "name": "alpha__one",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000022",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        arms = [
            _comparison_arm(
                "codex",
                {"name": "codex", "model_name": "model-x"},
            )
        ]
        plan_path = fixture.write_comparison_plan(attempts=1, arms=arms)

        with self.assertRaisesRegex(
            HarborResultsError,
            "must be provided together",
        ):
            load_rows(fixture.root, comparison_plan_path=plan_path)

        submitted = json.loads(fixture.submitted_config_path.read_text())
        submitted["debug"] = True
        _write_json(fixture.submitted_config_path, submitted)
        with self.assertRaisesRegex(
            HarborResultsError,
            "submitted job config",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

        fixture.write_comparison_plan(attempts=1, arms=arms)
        plan = json.loads(plan_path.read_text())
        plan["schema_version"] = "openbench-harbor-comparison-plan-v2"
        _write_json(plan_path, plan)
        with self.assertRaisesRegex(
            HarborResultsError,
            "openbench-harbor-comparison-plan-v4",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

    def test_custom_variants_use_bound_labels_and_config_digest_run_ids(self):
        fixture = GoldenHarborJob(
            self.root / "custom-variants",
            specs=[
                {
                    "name": "alpha__strict",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000031",
                    "score": 1.0,
                    "offset": 0,
                },
                {
                    "name": "alpha__fast",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000032",
                    "score": 0.0,
                    "offset": 5,
                },
            ],
        )
        agents = [
            {
                "import_path": "acme.agent:Agent",
                "model_name": "provider/model-x",
                "kwargs": {"mode": "strict"},
            },
            {
                "import_path": "acme.agent:Agent",
                "model_name": "provider/model-x",
                "kwargs": {"mode": "fast"},
            },
        ]
        for index, agent in enumerate(agents):
            fixture.set_custom_agent(index, agent, reported_name="acme-runtime")
        with self.assertRaisesRegex(
            HarborResultsError,
            "custom Harbor agents require an exact OpenBench comparison plan",
        ):
            load_rows(fixture.root)
        arms = [
            _comparison_arm(
                "strict",
                agents[0],
                canonical_harness="acme",
                canonical_model="model-x",
            ),
            _comparison_arm(
                "fast",
                agents[1],
                canonical_harness="acme",
                canonical_model="model-x",
            ),
        ]
        plan_path = fixture.write_comparison_plan(
            attempts=1,
            arms=arms,
            agents=agents,
        )

        rows = load_rows(
            fixture.root,
            comparison_plan_path=plan_path,
            submitted_job_config_path=fixture.submitted_config_path,
        )

        self.assertEqual({row["harness"] for row in rows}, {"acme"})
        self.assertEqual({row["model"] for row in rows}, {"model-x"})
        self.assertEqual({row["trial"] for row in rows}, {1})
        self.assertEqual(len({row["run_id"] for row in rows}), 2)
        self.assertEqual(
            {
                row["candidate_provenance"]["comparison_arm_id"]
                for row in rows
            },
            {"strict", "fast"},
        )
        self.assertEqual(
            {
                row["candidate_provenance"]["agent_config_sha256"]
                for row in rows
            },
            {arm["agent_config_sha256"] for arm in arms},
        )
        self.assertTrue(
            all(
                f"@{row['candidate_provenance']['agent_config_sha256']}:"
                in row["run_id"]
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["candidate_provenance"]["comparison_resolved_tasks"]
                == ["alpha"]
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["candidate_provenance"]["temporal_matched_block_claim"]
                is False
                for row in rows
            )
        )
        job_lock_path = fixture.root / "lock.json"
        trial_lock_path = fixture.root / "alpha__fast" / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        trial_lock = json.loads(trial_lock_path.read_text())
        original_digest = trial_lock["task"]["digest"]
        job_lock["trials"][1]["task"]["digest"] = "sha256:" + "f" * 64
        trial_lock["task"]["digest"] = "sha256:" + "f" * 64
        _write_json(job_lock_path, job_lock)
        _write_json(trial_lock_path, trial_lock)
        with self.assertRaisesRegex(
            HarborResultsError,
            "different immutable digests for the same task",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )
        job_lock["trials"][1]["task"]["digest"] = original_digest
        trial_lock["task"]["digest"] = original_digest
        _write_json(job_lock_path, job_lock)
        _write_json(trial_lock_path, trial_lock)

        plan = json.loads(plan_path.read_text())
        plan["arms"][0]["agent_config_sha256"] = "f" * 64
        _write_json(plan_path, plan)
        with self.assertRaisesRegex(
            HarborResultsError,
            "rendered config digests",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

    def test_comparison_plan_rejects_agent_digest_and_dataset_drift(self):
        fixture = GoldenHarborJob(
            self.root / "dataset-plan",
            specs=[
                {
                    "name": "alpha__one",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000041",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        agent = {"name": "codex", "model_name": "model-x"}
        arm = _comparison_arm("codex", agent)
        descriptor = {
            "name": "terminal-bench",
            "version": "2.0",
            "task_names": ["alpha"],
        }
        plan_path = fixture.write_comparison_plan(
            attempts=1,
            arms=[arm],
            agents=[agent],
            dataset_descriptor=descriptor,
        )
        rows = load_rows(
            fixture.root,
            comparison_plan_path=plan_path,
            submitted_job_config_path=fixture.submitted_config_path,
        )
        self.assertEqual(
            rows[0]["candidate_provenance"]["comparison_resolved_tasks"],
            ["openbench/alpha"],
        )

        plan = json.loads(plan_path.read_text())
        plan["arms"][0]["agent_config_sha256"] = "f" * 64
        _write_json(plan_path, plan)
        with self.assertRaisesRegex(
            HarborResultsError,
            "rendered config digests",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

        plan["arms"][0] = arm
        plan["dataset"]["version"] = "2.1"
        _write_json(plan_path, plan)
        with self.assertRaisesRegex(
            HarborResultsError,
            "immutable Harbor dataset descriptor",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

    def test_package_dataset_plan_binds_lock_resolved_tasks(self):
        fixture = GoldenHarborJob(
            self.root / "package-plan",
            specs=[
                {
                    "name": "alpha__one",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000051",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        agent = {"name": "codex", "model_name": "model-x"}
        descriptor = {
            "name": "openbench/core",
            "ref": "sha256:" + "3" * 64,
        }
        plan_path = fixture.write_comparison_plan(
            attempts=1,
            arms=[_comparison_arm("codex", agent)],
            agents=[agent],
            dataset_descriptor=descriptor,
        )

        row = load_rows(
            fixture.root,
            comparison_plan_path=plan_path,
            submitted_job_config_path=fixture.submitted_config_path,
        )[0]

        self.assertEqual(
            row["candidate_provenance"]["comparison_plan"]["dataset"],
            descriptor,
        )
        self.assertIsNone(
            row["candidate_provenance"]["comparison_plan"]["tasks"]
        )
        self.assertEqual(
            row["candidate_provenance"]["comparison_resolved_tasks"],
            ["openbench/alpha"],
        )

    def test_local_comparison_plan_maps_harbor_selector_to_canonical_task_id(self):
        fixture = GoldenHarborJob(
            self.root / "canonical-task-id",
            specs=[
                {
                    "name": "fix-failing-test__one",
                    "task": "fix-failing-test",
                    "id": "00000000-0000-0000-0000-000000000071",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        agent = {"name": "codex", "model_name": "model-x"}
        plan_path = fixture.write_comparison_plan(
            attempts=1,
            arms=[_comparison_arm("codex", agent)],
            agents=[agent],
            task_id_map={
                "fix-failing-test": "openbench/fix-failing-test",
            },
        )

        row = load_rows(
            fixture.root,
            comparison_plan_path=plan_path,
            submitted_job_config_path=fixture.submitted_config_path,
        )[0]

        self.assertEqual(row["task"], "openbench/fix-failing-test")
        self.assertEqual(
            row["candidate_provenance"]["comparison_plan"]["tasks"],
            ["fix-failing-test"],
        )
        self.assertEqual(
            row["candidate_provenance"]["comparison_plan"]["task_id_map"],
            {"fix-failing-test": "openbench/fix-failing-test"},
        )
        self.assertEqual(
            row["candidate_provenance"]["comparison_resolved_tasks"],
            ["openbench/fix-failing-test"],
        )
        self.assertEqual(
            row["candidate_provenance"]["comparison_block"],
            {"task": "openbench/fix-failing-test", "index": 1},
        )

        (
            fixture.task_root / "fix-failing-test" / "task.toml"
        ).write_text(
            'schema_version = "1.4"\n\n'
            '[task]\nname = "openbench/tampered"\n'
        )
        with self.assertRaisesRegex(
            HarborResultsError,
            r"task\.toml identity",
        ):
            load_rows(
                fixture.root,
                comparison_plan_path=plan_path,
                submitted_job_config_path=fixture.submitted_config_path,
            )

    def test_terminal_failures_map_without_fabricated_optional_evidence(self):
        exception_types = (
            "AgentTimeoutError",
            "ApiUsageLimitError",
            "AgentAuthenticationError",
            "AgentSafetyRefusalError",
            "RuntimeError",
        )
        specs = [
            {
                "name": f"alpha__terminal_{index}",
                "task": "alpha",
                "id": f"00000000-0000-0000-0000-{index:012d}",
                "score": 0.0,
                "offset": index,
            }
            for index in range(1, len(exception_types) + 1)
        ]
        fixture = GoldenHarborJob(self.root / "terminal-job", specs=specs)
        for index, exception_type in enumerate(exception_types):
            fixture.make_terminal(exception_type, index=index)

        rows = load_rows(fixture.root)
        by_type = {
            row["candidate_provenance"]["harbor_exception_type"]: row
            for row in rows
        }

        expected = {
            "AgentTimeoutError": ("timeout", "harbor_timeout:AgentTimeoutError"),
            "ApiUsageLimitError": (
                "rate_limited",
                "harbor_rate_limit:ApiUsageLimitError",
            ),
            "AgentAuthenticationError": (
                "infra",
                "harbor_auth:AgentAuthenticationError",
            ),
            "AgentSafetyRefusalError": (
                "infra",
                "harbor_safety:AgentSafetyRefusalError",
            ),
            "RuntimeError": (
                "infra",
                "harbor_infrastructure:RuntimeError",
            ),
        }
        self.assertEqual(set(by_type), set(expected))
        for exception_type, (failure_class, failure_reason) in expected.items():
            row = by_type[exception_type]
            self.assertFalse(row["success"])
            self.assertFalse(row["completed"])
            self.assertEqual(row["failure_class"], failure_class)
            self.assertEqual(row["failure_reason"], failure_reason)
            self.assertEqual(
                row["error"], f"Harbor terminal failure: {exception_type}"
            )
            for field in (
                "score",
                "checker_exit",
                "t_agent_s",
                "t_checker_s",
                "turns",
                "tokens",
                "usage_raw",
                "workspace_source",
            ):
                self.assertIsNone(row[field], (exception_type, field))
            provenance = row["candidate_provenance"]
            for field in (
                "reward_sha256",
                "openbench_verifier_evidence_sha256",
                "atif_sha256",
                "artifact_manifest_sha256",
                "final_workspace_sha256",
                "openbench_task_content_digest",
                "openbench_harbor_export",
            ):
                self.assertIsNone(provenance[field], (exception_type, field))
            self.assertEqual(row["usage_evidence_grade"], "usage_unavailable")
            self.assertFalse(row["usage_ranking_eligible"])

        filtered = stats.filter_rows(rows, [])
        self.assertEqual(len(filtered["countable_rows"]), 1)
        self.assertEqual(filtered["excluded_counts"], {"infra": 3, "rate_limited": 1})

    def test_terminal_failure_preserves_valid_optional_evidence(self):
        fixture = GoldenHarborJob(
            self.root / "terminal-with-evidence",
            specs=[
                {
                    "name": "alpha__terminal",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000099",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        fixture.make_terminal(
            "AgentAuthenticationError",
            retain_optional_evidence=True,
        )

        row = load_rows(fixture.root)[0]

        self.assertFalse(row["success"])
        self.assertFalse(row["completed"])
        self.assertEqual(row["failure_class"], "infra")
        self.assertEqual(row["checker_exit"], 0)
        self.assertEqual(row["score"], 1.0)
        self.assertEqual(row["tokens"], 115)
        self.assertEqual(row["turns"], 1)
        self.assertIsNotNone(row["workspace_source"])
        self.assertIsNotNone(
            row["candidate_provenance"]["openbench_verifier_evidence_sha256"]
        )

    def test_terminal_evidence_still_rejects_internal_contradictions(self):
        fixture = GoldenHarborJob(self.root / "terminal-contradiction")
        fixture.make_terminal("AgentTimeoutError")
        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["exception_info"]["occurred_at"] = "2026-07-21T10:00:00+00:00"
        _write_json(result_path, result)
        with self.assertRaisesRegex(HarborResultsError, "within trial timing"):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "terminal-partial-reward")
        fixture.make_terminal(
            "AgentAuthenticationError",
            retain_optional_evidence=True,
        )
        (fixture.trial() / "verifier" / "openbench-verifier-evidence.json").unlink()
        with self.assertRaisesRegex(HarborResultsError, "verifier_evidence"):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "terminal-partial-usage")
        fixture.make_terminal("AgentTimeoutError")
        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["agent_result"] = {
            "n_input_tokens": 1,
            "n_cache_tokens": 0,
            "n_output_tokens": 1,
            "cost_usd": 0.0,
        }
        _write_json(result_path, result)
        fixture.sync_aggregate()
        with self.assertRaisesRegex(HarborResultsError, "requires matching ATIF"):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "terminal-workspace-without-manifest")
        fixture.make_terminal("RuntimeError")
        workspace = fixture.trial() / "artifacts" / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "unbound.txt").write_text("unbound\n")
        with self.assertRaisesRegex(HarborResultsError, "without an artifact manifest"):
            load_rows(fixture.root)

    def test_cursor_and_devin_profile_aliases_are_non_proxy_harbor_reported(self):
        profiles = (
            (CURSOR_PROFILE_IMPORT, "cursor", "gpt-5.6-sol-medium"),
            (DEVIN_PROFILE_IMPORT, "devin", "gpt-5-6-sol-medium"),
        )
        for index, (import_path, harness, harbor_model) in enumerate(profiles):
            with self.subTest(harness=harness):
                fixture = GoldenHarborJob(
                    self.root / f"profile-{harness}",
                    specs=[
                        {
                            "name": f"alpha__{harness}",
                            "task": "alpha",
                            "id": f"00000000-0000-0000-0000-{index + 50:012d}",
                            "score": 1.0,
                            "offset": 0,
                        }
                    ],
                )
                trial_lock_path = fixture.trial() / "lock.json"
                trial_lock = json.loads(trial_lock_path.read_text())
                trial_lock["agent"].pop("name")
                trial_lock["agent"]["import_path"] = import_path
                trial_lock["agent"]["model_name"] = harbor_model
                _write_json(trial_lock_path, trial_lock)
                job_lock_path = fixture.root / "lock.json"
                job_lock = json.loads(job_lock_path.read_text())
                job_lock["trials"] = [trial_lock]
                _write_json(job_lock_path, job_lock)

                result_path = fixture.trial() / "result.json"
                result = json.loads(result_path.read_text())
                result["config"]["agent"]["name"] = None
                result["config"]["agent"]["import_path"] = import_path
                result["config"]["agent"]["model_name"] = harbor_model
                result["agent_info"]["name"] = harness
                result["agent_info"]["model_info"] = {
                    "name": harbor_model,
                    "provider": None,
                }
                _write_json(result_path, result)
                trajectory_path = fixture.trial() / "agent" / "trajectory.json"
                trajectory = json.loads(trajectory_path.read_text())
                trajectory["agent"]["name"] = harness
                trajectory["agent"]["model_name"] = harbor_model
                trajectory["steps"][1]["model_name"] = harbor_model
                _write_json(trajectory_path, trajectory)

                row = load_rows(fixture.root)[0]
                self.assertEqual(row["harness"], harness)
                self.assertEqual(row["model"], harbor_model)
                self.assertEqual(row["usage_evidence_grade"], "harbor_reported")
                self.assertFalse(row["candidate_provenance"]["proxy_measured"])

    def test_custom_agent_import_path_is_the_lock_identity(self):
        fixture = self.fixture()
        for index in range(len(fixture.specs)):
            trial_dir = fixture.trial(index)
            lock_path = trial_dir / "lock.json"
            lock = json.loads(lock_path.read_text())
            lock["agent"].pop("name")
            lock["agent"]["import_path"] = AGENT_IMPORT_PATH
            _write_json(lock_path, lock)

            result_path = trial_dir / "result.json"
            result = json.loads(result_path.read_text())
            result["config"]["agent"]["name"] = None
            result["config"]["agent"]["import_path"] = AGENT_IMPORT_PATH
            _write_json(result_path, result)

        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"] = [
            json.loads((fixture.trial(index) / "lock.json").read_text())
            for index in range(len(fixture.specs))
        ]
        _write_json(job_lock_path, job_lock)

        rows = load_rows(fixture.root)

        self.assertEqual(len(rows), len(fixture.specs))
        self.assertTrue(all(row["harness"] == "codex" for row in rows))
        self.assertTrue(
            all(
                row["candidate_provenance"]["harbor_agent_config_name"]
                == AGENT_IMPORT_PATH
                for row in rows
            )
        )

    def test_agent_lock_requires_name_or_import_path(self):
        fixture = self.fixture()
        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"][0]["agent"].pop("name")
        _write_json(job_lock_path, job_lock)

        with self.assertRaisesRegex(
            HarborResultsError,
            r"job lock\.trials\[0\]\.agent\.import_path",
        ):
            load_rows(fixture.root)

    def test_import_appends_row_fields_and_rejects_duplicate_without_mutation(self):
        fixture = self.fixture()
        output = self.root / "results.jsonl"
        seed = {field: None for field in ROW_FIELDS}
        seed["run_id"] = "seed:task:model:trial1"
        output.write_text(json.dumps(seed) + "\n")

        rows = import_results(fixture.root, output)
        written = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(written), 4)
        self.assertEqual(written[1:], rows)
        before = output.read_bytes()
        with self.assertRaisesRegex(HarborResultsError, "already exists"):
            import_results(fixture.root, output)
        self.assertEqual(output.read_bytes(), before)

    def test_corrupt_existing_jsonl_fails_before_append(self):
        fixture = self.fixture()
        output = self.root / "results.jsonl"
        output.write_text('{"run_id":"ok"}\n{broken\n')
        before = output.read_bytes()
        with self.assertRaisesRegex(HarborResultsError, "corrupt JSONL"):
            import_results(fixture.root, output)
        self.assertEqual(output.read_bytes(), before)

    def test_concurrent_imports_cannot_append_duplicate_batches(self):
        fixture = self.fixture()
        output = self.root / "concurrent.jsonl"
        successes = []
        failures = []

        def run_import():
            try:
                successes.append(import_results(fixture.root, output))
            except HarborResultsError as exc:
                failures.append(str(exc))

        threads = [threading.Thread(target=run_import) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("already exists", failures[0])
        self.assertEqual(len(output.read_text().splitlines()), 3)

    def test_harbor_import_waits_for_normal_append_then_detects_collision(self):
        fixture = self.fixture()
        output = self.root / "normal-first.jsonl"
        colliding_row = load_rows(fixture.root)[0]
        normal_at_fsync = threading.Event()
        release_normal = threading.Event()
        harbor_lock_attempted = threading.Event()
        harbor_lock_acquired = threading.Event()
        real_fsync = os.fsync
        real_results_file_lock = harbor_results.results_file_lock
        outcomes = {}

        def controlled_fsync(descriptor):
            if threading.current_thread().name == "normal-append":
                normal_at_fsync.set()
                if not release_normal.wait(timeout=5):
                    raise AssertionError("normal append was not released")
            return real_fsync(descriptor)

        @contextmanager
        def observed_harbor_lock(path):
            harbor_lock_attempted.set()
            with real_results_file_lock(path):
                harbor_lock_acquired.set()
                yield

        def append_normal():
            try:
                outcomes["normal"] = run.append_row(
                    output, colliding_row, reject_duplicate=True
                )
            except BaseException as exc:
                outcomes["normal_error"] = exc

        def import_harbor():
            try:
                outcomes["harbor"] = import_results(fixture.root, output)
            except BaseException as exc:
                outcomes["harbor_error"] = exc

        normal_thread = threading.Thread(
            target=append_normal, name="normal-append"
        )
        harbor_thread = threading.Thread(
            target=import_harbor, name="harbor-import"
        )
        with (
            mock.patch.object(run.os, "fsync", side_effect=controlled_fsync),
            mock.patch.object(
                harbor_results,
                "results_file_lock",
                observed_harbor_lock,
            ),
        ):
            normal_thread.start()
            try:
                self.assertTrue(normal_at_fsync.wait(timeout=5))
                harbor_thread.start()
                self.assertTrue(harbor_lock_attempted.wait(timeout=5))
                self.assertFalse(harbor_lock_acquired.wait(timeout=0.2))
            finally:
                release_normal.set()
                normal_thread.join(timeout=5)
                if harbor_thread.ident is not None:
                    harbor_thread.join(timeout=5)

        self.assertFalse(normal_thread.is_alive())
        self.assertFalse(harbor_thread.is_alive())
        self.assertNotIn("normal_error", outcomes)
        self.assertTrue(outcomes["normal"])
        self.assertIsInstance(outcomes.get("harbor_error"), HarborResultsError)
        self.assertIn("already exists", str(outcomes["harbor_error"]))
        written = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([row["run_id"] for row in written], [colliding_row["run_id"]])

    def test_harbor_rollback_finishes_before_waiting_normal_append(self):
        fixture = self.fixture()
        output = self.root / "cross-writer-rollback.jsonl"
        seed = {field: None for field in ROW_FIELDS}
        seed["run_id"] = "seed:task:model:trial1"
        self.assertTrue(run.append_row(output, seed))
        normal_row = dict(seed)
        normal_row["run_id"] = "normal:task:model:trial1"

        harbor_at_fsync = threading.Event()
        release_harbor = threading.Event()
        normal_lock_attempted = threading.Event()
        normal_lock_acquired = threading.Event()
        real_fsync = os.fsync
        real_results_file_lock = run.results_file_lock
        harbor_fsync_calls = 0
        outcomes = {}

        def controlled_fsync(descriptor):
            nonlocal harbor_fsync_calls
            if threading.current_thread().name == "harbor-import":
                harbor_fsync_calls += 1
                if harbor_fsync_calls == 1:
                    harbor_at_fsync.set()
                    if not release_harbor.wait(timeout=5):
                        raise AssertionError("Harbor import was not released")
                    raise OSError("injected disk failure")
            return real_fsync(descriptor)

        @contextmanager
        def observed_normal_lock(path):
            normal_lock_attempted.set()
            with real_results_file_lock(path):
                normal_lock_acquired.set()
                yield

        def import_harbor():
            try:
                outcomes["harbor"] = import_results(fixture.root, output)
            except BaseException as exc:
                outcomes["harbor_error"] = exc

        def append_normal():
            try:
                outcomes["normal"] = run.append_row(
                    output, normal_row, reject_duplicate=True
                )
            except BaseException as exc:
                outcomes["normal_error"] = exc

        harbor_thread = threading.Thread(
            target=import_harbor, name="harbor-import"
        )
        normal_thread = threading.Thread(
            target=append_normal, name="normal-append"
        )
        with (
            mock.patch.object(
                harbor_results.os,
                "fsync",
                side_effect=controlled_fsync,
            ),
            mock.patch.object(
                run,
                "results_file_lock",
                observed_normal_lock,
            ),
        ):
            harbor_thread.start()
            try:
                self.assertTrue(harbor_at_fsync.wait(timeout=5))
                normal_thread.start()
                self.assertTrue(normal_lock_attempted.wait(timeout=5))
                self.assertFalse(normal_lock_acquired.wait(timeout=0.2))
            finally:
                release_harbor.set()
                harbor_thread.join(timeout=5)
                if normal_thread.ident is not None:
                    normal_thread.join(timeout=5)

        self.assertFalse(harbor_thread.is_alive())
        self.assertFalse(normal_thread.is_alive())
        self.assertIsInstance(outcomes.get("harbor_error"), HarborResultsError)
        self.assertIn("rolled back", str(outcomes["harbor_error"]))
        self.assertNotIn("normal_error", outcomes)
        self.assertTrue(outcomes["normal"])
        written = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(
            [row["run_id"] for row in written],
            [seed["run_id"], normal_row["run_id"]],
        )

    def test_append_io_failure_rolls_back_whole_batch(self):
        fixture = self.fixture()
        output = self.root / "rollback.jsonl"
        output.write_text('{"run_id":"seed"}\n')
        before = output.read_bytes()
        with mock.patch("obench.harbor_results.os.fsync", side_effect=[OSError("disk"), None]):
            with self.assertRaisesRegex(HarborResultsError, "rolled back"):
                import_results(fixture.root, output)
        self.assertEqual(output.read_bytes(), before)

    def test_workspace_digest_is_unambiguous_across_file_boundaries(self):
        first = self.root / "tree-one"
        second = self.root / "tree-two"
        first.mkdir()
        second.mkdir()
        (first / "a").write_bytes(b"Xf\0b\0Y")
        (second / "a").write_bytes(b"X")
        (second / "b").write_bytes(b"Y")
        self.assertNotEqual(_tree_digest(first), _tree_digest(second))

    def test_rejects_unresolved_or_wrong_harbor_build(self):
        for field, value in (
            ("version", None),
            ("version", "0.20.1"),
            ("git_commit_hash", None),
            ("git_commit_hash", "0" * 40),
        ):
            with self.subTest(field=field, value=value):
                fixture = GoldenHarborJob(self.root / f"job-{field}-{value}")
                path = fixture.root / "lock.json"
                lock = json.loads(path.read_text())
                lock["harbor"][field] = value
                _write_json(path, lock)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_rejects_editable_harbor_install(self):
        fixture = GoldenHarborJob(self.root / "job-editable")
        path = fixture.root / "lock.json"
        lock = json.loads(path.read_text())
        lock["harbor"]["is_editable"] = True
        _write_json(path, lock)

        with self.assertRaisesRegex(
            HarborResultsError,
            "editable Harbor installations are not publication-grade",
        ):
            load_rows(fixture.root)

    def test_rejects_missing_evidence_files(self):
        relative_paths = (
            "lock.json",
            "result.json",
            "agent/trajectory.json",
            "verifier/reward.txt",
            "verifier/openbench-verifier-evidence.json",
            "artifacts/manifest.json",
        )
        for index, relative in enumerate(relative_paths):
            with self.subTest(relative=relative):
                fixture = GoldenHarborJob(self.root / f"job-missing-{index}")
                (fixture.trial() / relative).unlink()
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_checker_exit_is_authoritative_even_when_partial_score_is_one(self):
        fixture = GoldenHarborJob(
            self.root / "job-nonzero-one",
            specs=[
                {
                    "name": "alpha__one",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000001",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        evidence_path = (
            fixture.trial()
            / "verifier"
            / "openbench-verifier-evidence.json"
        )
        evidence = json.loads(evidence_path.read_text())
        evidence["checker_exit"] = 7
        evidence["parsed_score"] = 1.0
        _write_json(evidence_path, evidence)

        row = load_rows(fixture.root)[0]
        self.assertEqual(row["checker_exit"], 7)
        self.assertEqual(row["score"], 1.0)
        self.assertFalse(row["success"])
        self.assertEqual(row["failure_class"], "wrong_answer")

    def test_rejects_invalid_or_mismatched_verifier_evidence(self):
        for index, mutation in enumerate(
            (
                {"checker_exit": "1"},
                {"reward": 0.25},
                {"checker_exit": 1, "parsed_score": None},
                {
                    "openbench_task_content_digest": {
                        "scheme": 1,
                        "sha256": "1" * 64,
                    }
                },
                {
                    "openbench_task_content_digest": {
                        "scheme": 2,
                        "sha256": "A" * 64,
                    }
                },
            )
        ):
            with self.subTest(mutation=mutation):
                fixture = GoldenHarborJob(
                    self.root / f"job-evidence-{index}",
                    specs=[
                        {
                            "name": "alpha__one",
                            "task": "alpha",
                            "id": "00000000-0000-0000-0000-000000000001",
                            "score": 1.0,
                            "offset": 0,
                        }
                    ],
                )
                evidence_path = (
                    fixture.trial()
                    / "verifier"
                    / "openbench-verifier-evidence.json"
                )
                evidence = json.loads(evidence_path.read_text())
                evidence.update(mutation)
                _write_json(evidence_path, evidence)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_rejects_inconsistent_content_digest_for_same_task(self):
        fixture = self.fixture()
        evidence_path = (
            fixture.trial(1)
            / "verifier"
            / "openbench-verifier-evidence.json"
        )
        evidence = json.loads(evidence_path.read_text())
        evidence["openbench_task_content_digest"]["sha256"] = "3" * 64
        _write_json(evidence_path, evidence)

        with self.assertRaisesRegex(
            HarborResultsError,
            "inconsistent OpenBench content digests for task 'alpha'",
        ):
            load_rows(fixture.root)

    def test_accepts_populated_workspace_config_compatibility(self):
        fixture = self.fixture()
        for index in range(len(fixture.specs)):
            result_path = fixture.trial(index) / "result.json"
            result = json.loads(result_path.read_text())
            result["config"]["artifacts"] = [
                {"source": "/app", "destination": "workspace"}
            ]
            _write_json(result_path, result)
        fixture.sync_aggregate()

        self.assertEqual(len(load_rows(fixture.root)), len(fixture.specs))

    def test_rejects_conflicting_or_unsafe_workspace_config(self):
        cases = (
            [{"source": "/app", "destination": "other"}],
            [{"source": "/tmp", "destination": "workspace"}],
            [
                {"source": "/app", "destination": "workspace"},
                {"source": "/app", "destination": "workspace"},
            ],
            [{"source": "/app/../tmp", "destination": "workspace"}],
            [{"source": "/app", "destination": "../workspace"}],
            [{"source": "/logs/artifacts", "destination": "logs"}],
            [{"source": "/app", "destination": "workspace", "service": "db"}],
            [{"source": "/app", "destination": "workspace", "type": "directory"}],
            [
                {"source": "/app", "destination": "workspace"},
                {"source": "/tmp", "destination": "workspace/injected"},
            ],
        )
        for index, artifacts in enumerate(cases):
            with self.subTest(artifacts=artifacts):
                fixture = GoldenHarborJob(self.root / f"config-artifacts-{index}")
                result_path = fixture.trial() / "result.json"
                result = json.loads(result_path.read_text())
                result["config"]["artifacts"] = artifacts
                _write_json(result_path, result)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_rejects_tampered_manifest_workspace_evidence(self):
        cases = (
            lambda entry: entry.update(source="/tmp"),
            lambda entry: entry.update(destination="artifacts/other"),
            lambda entry: entry.update(status="failed"),
            lambda entry: entry.update(type="file"),
            lambda entry: entry.update(service="agent"),
            lambda entry: entry.update(source="/app/../tmp"),
            lambda entry: entry.update(destination="artifacts/../workspace"),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(index=index):
                fixture = GoldenHarborJob(self.root / f"manifest-artifacts-{index}")
                manifest_path = fixture.trial() / "artifacts" / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                mutate(manifest[1])
                _write_json(manifest_path, manifest)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "manifest-artifacts-duplicate")
        manifest_path = fixture.trial() / "artifacts" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.append(dict(manifest[1]))
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            HarborResultsError,
            "duplicate artifact source or destination",
        ):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "manifest-artifacts-nested")
        manifest_path = fixture.trial() / "artifacts" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.append(
            {
                "source": "/tmp/injected",
                "destination": "artifacts/workspace/injected",
                "type": "directory",
                "status": "ok",
                "service": None,
            }
        )
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(
            HarborResultsError,
            "duplicate or contradictory workspace artifact entry",
        ):
            load_rows(fixture.root)

    def test_accepts_qualified_atif_top_level_model_compatibility(self):
        fixture = self.qualified_model_fixture(
            "qualified-top-level-model",
            top_level_model="openai/gpt-5.6-sol",
            step_model="openai/gpt-5.6-sol",
        )

        self.assertEqual(
            load_rows(fixture.root)[0]["model"],
            "openai/gpt-5.6-sol",
        )

    def test_profile_atif_steps_accept_only_the_canonical_model_identity(self):
        fixture = self.qualified_model_fixture(
            "profile-canonical-step-model",
            top_level_model="openai/gpt-5.6-sol",
            step_model="gpt-5.6-sol",
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["agent"].pop("name")
        trial_lock["agent"]["import_path"] = OPENCODE_PROFILE_IMPORT
        _write_json(lock_path, trial_lock)

        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"][0] = trial_lock
        _write_json(job_lock_path, job_lock)

        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["name"] = None
        result["config"]["agent"]["import_path"] = OPENCODE_PROFILE_IMPORT
        result["agent_info"]["name"] = "opencode"
        _write_json(result_path, result)

        trajectory_path = fixture.trial() / "agent" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["agent"]["name"] = "opencode"
        _write_json(trajectory_path, trajectory)

        row = load_rows(fixture.root)[0]

        self.assertEqual(row["model"], "gpt-5.6-sol")
        self.assertEqual(
            row["candidate_provenance"]["harbor_model_name"],
            "openai/gpt-5.6-sol",
        )

    def test_rejects_unrelated_or_misplaced_atif_model_identities(self):
        cases = (
            ("other-provider/gpt-5.6-sol", "openai/gpt-5.6-sol"),
            ("gpt-5.6-sol", "gpt-5.6-sol"),
            ("gpt-5.6-sol", "openai/gpt-5.6-terra"),
        )
        for index, (top_level_model, step_model) in enumerate(cases):
            with self.subTest(
                top_level_model=top_level_model,
                step_model=step_model,
            ):
                fixture = self.qualified_model_fixture(
                    f"tampered-model-{index}",
                    top_level_model=top_level_model,
                    step_model=step_model,
                )
                with self.assertRaisesRegex(
                    HarborResultsError,
                    "does not match trial model identity",
                ):
                    load_rows(fixture.root)

    def test_real_0200_artifact_shape_regression_without_credentials(self):
        fixture = GoldenHarborJob(
            self.root / "vertical-slice-public",
            specs=[
                {
                    "name": "make-it-run__qT2fWKm",
                    "task": "make-it-run",
                    "id": "1c7aae27-b206-449e-bb14-440c861da8f2",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["task"]["name"] = "make-it-run"
        trial_lock["agent"]["name"] = "codex"
        trial_lock["agent"]["model_name"] = "openai/gpt-5.6-sol"
        _write_json(lock_path, trial_lock)
        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["harbor"]["is_editable"] = False
        job_lock["trials"] = [trial_lock]
        _write_json(job_lock_path, job_lock)

        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["source"] = None
        result["task_checksum"] = "c" * 64
        result["config"]["agent"] = {
            "name": "codex",
            "model_name": "openai/gpt-5.6-sol",
            "skills": [],
            "resume_trajectory": False,
            "extra_allowed_hosts": [],
            "kwargs": {},
            "mcp_servers": [],
        }
        result["agent_info"] = {
            "name": "codex",
            "version": "0.1.0",
            "model_info": {"name": "gpt-5.6-sol", "provider": "openai"},
        }
        result["agent_result"] = {
            "n_input_tokens": None,
            "n_cache_tokens": None,
            "n_output_tokens": None,
            "cost_usd": None,
            "rollout_details": None,
            "metadata": {"artifact_shape": "no-token-usage"},
        }
        _write_json(result_path, result)
        _write_json(
            fixture.trial() / "agent" / "trajectory.json",
            {
                "schema_version": "ATIF-v1.7",
                "session_id": result["id"],
                "agent": {
                    "name": "codex",
                    "version": "0.1.0",
                    "model_name": "gpt-5.6-sol",
                },
                "steps": [
                    {"step_id": 1, "source": "user", "message": "Fix the task."},
                    {
                        "step_id": 2,
                        "source": "agent",
                        "message": "Repaired the task.",
                        "llm_call_count": 0,
                        "model_name": "openai/gpt-5.6-sol",
                    },
                ],
            },
        )
        job_result_path = fixture.root / "result.json"
        job_result = json.loads(job_result_path.read_text())
        job_result["started_at"] = "2026-07-20T05:59:00"
        job_result["updated_at"] = "2026-07-20T06:04:00"
        job_result["finished_at"] = "2026-07-20T06:05:00"
        job_result["stats"] = {
            "n_completed_trials": 1,
            "n_errored_trials": 0,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "n_cancelled_trials": 0,
            "n_retries": 0,
            "evals": {
                "codex__gpt-5.6-sol__adhoc": {
                    "n_trials": 1,
                    "n_errors": 0,
                    "metrics": [{"mean": 1.0}],
                    "pass_at_k": {},
                    "reward_stats": {
                        "reward": {"1.0": ["make-it-run__qT2fWKm"]}
                    },
                    "exception_stats": {},
                }
            },
            "n_input_tokens": None,
            "n_cache_tokens": None,
            "n_output_tokens": None,
            "cost_usd": None,
        }
        _write_json(job_result_path, job_result)

        self.assertEqual(result["config"]["artifacts"], [])
        self.assertEqual(
            json.loads(
                (fixture.trial() / "artifacts" / "manifest.json").read_text()
            ),
            [
                {
                    "source": "/logs/artifacts",
                    "destination": "artifacts/logs/artifacts",
                    "type": "directory",
                    "status": "empty",
                    "service": None,
                },
                {
                    "source": "/app",
                    "destination": "artifacts/workspace",
                    "type": "directory",
                    "status": "ok",
                    "service": None,
                },
            ],
        )
        row = load_rows(fixture.root)[0]
        self.assertEqual(row["harness"], "codex")
        self.assertEqual(row["model"], "openai/gpt-5.6-sol")
        self.assertEqual(row["token_basis"], "unmetered")
        self.assertEqual(
            row["candidate_provenance"]["harbor_git_commit_hash"],
            "72bc40b1e58b47a9cc6e0f14c29aced3a9e53767",
        )

    def test_accepts_pinned_oauth_agent_semantic_alias(self):
        fixture = GoldenHarborJob(
            self.root / "job-oauth-alias",
            specs=[
                {
                    "name": "alpha__oauth",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000004",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["agent"]["name"] = AGENT_IMPORT_PATH
        _write_json(lock_path, trial_lock)

        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"] = [trial_lock]
        _write_json(job_lock_path, job_lock)

        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["name"] = AGENT_IMPORT_PATH
        _write_json(result_path, result)

        row = load_rows(fixture.root)[0]
        self.assertEqual(row["harness"], "codex")
        self.assertEqual(
            row["candidate_provenance"]["harbor_agent_config_name"],
            AGENT_IMPORT_PATH,
        )

    def test_profile_agent_requires_durable_proxy_evidence(self):
        fixture = GoldenHarborJob(
            self.root / "job-profile-metering-required",
            specs=[
                {
                    "name": "alpha__profile",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000014",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        profile_import = (
            "obench.harbor_agents.codex_profile:"
            "OpenBenchCodexOAuthProfile"
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["agent"]["name"] = profile_import
        _write_json(lock_path, trial_lock)

        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"] = [trial_lock]
        _write_json(job_lock_path, job_lock)

        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["name"] = profile_import
        _write_json(result_path, result)

        with self.assertRaisesRegex(
            HarborResultsError,
            "metering",
        ):
            load_rows(fixture.root)

    def test_profile_agent_imports_structurally_valid_usage_mismatch(self):
        fixture = GoldenHarborJob(
            self.root / "job-profile-metering-mismatch",
            specs=[
                {
                    "name": "alpha__profile",
                    "task": "alpha",
                    "id": "00000000-0000-0000-0000-000000000015",
                    "score": 1.0,
                    "offset": 0,
                }
            ],
        )
        profile_import = (
            "obench.harbor_agents.codex_profile:"
            "OpenBenchCodexOAuthProfile"
        )
        lock_path = fixture.trial() / "lock.json"
        trial_lock = json.loads(lock_path.read_text())
        trial_lock["agent"]["name"] = profile_import
        _write_json(lock_path, trial_lock)
        job_lock_path = fixture.root / "lock.json"
        job_lock = json.loads(job_lock_path.read_text())
        job_lock["trials"] = [trial_lock]
        _write_json(job_lock_path, job_lock)
        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["config"]["agent"]["name"] = profile_import
        _write_json(result_path, result)

        metering_path = fixture.trial() / "agent" / "harbor-metering"
        evidence_path = metering_path / "harbor-metering.json"
        _write_json(evidence_path, {"placeholder": True})
        evidence = {
            "schema_version": "openbench.harbor-metering.v2",
            "proxy_complete": True,
            "proxy_measured": {
                "calls": 1,
                "input_tokens": 101,
                "cache_tokens": 25,
                "output_tokens": 40,
            },
            "request_counts": {"total": 1, "model": 1, "auxiliary": 0},
            "ledger_seal": {
                "root_hash": "a" * 64,
                "ledger_sha256": "b" * 64,
            },
            "reconciliation": {"status": "mismatch"},
        }
        with mock.patch(
            "obench.harbor_results.verify_evidence_dir",
            return_value=evidence,
        ):
            row = load_rows(fixture.root)[0]

        self.assertEqual(row["tokens_input_uncached"], 75)
        self.assertEqual(row["tokens_proxy_input_uncached"], 76)
        self.assertEqual(
            row["usage_evidence_grade"],
            "harbor_reported_proxy_mismatch",
        )
        self.assertFalse(row["usage_ranking_eligible"])
        self.assertEqual(
            row["usage_ranking_exclusion_reason"], "proxy_mismatch"
        )

    def test_rejects_coherent_result_and_atif_agent_relabeling(self):
        fixture = self.fixture()
        result_path = fixture.trial() / "result.json"
        result = json.loads(result_path.read_text())
        result["agent_info"]["name"] = "other-agent"
        _write_json(result_path, result)

        trajectory_path = fixture.trial() / "agent" / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text())
        trajectory["agent"]["name"] = "other-agent"
        _write_json(trajectory_path, trajectory)

        with self.assertRaisesRegex(
            HarborResultsError,
            "does not match immutable trial-lock agent identity",
        ):
            load_rows(fixture.root)

    def test_rejects_random_incomplete_and_duplicate_trials(self):
        fixture = GoldenHarborJob(self.root / "job-random")
        (fixture.root / "random-directory").mkdir()
        with self.assertRaisesRegex(HarborResultsError, "unexpected directory"):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "job-incomplete")
        (fixture.trial() / "result.json").unlink()
        with self.assertRaises(HarborResultsError):
            load_rows(fixture.root)

        fixture = GoldenHarborJob(self.root / "job-duplicate")
        second_path = fixture.trial(1) / "result.json"
        second = json.loads(second_path.read_text())
        second["id"] = fixture.specs[0]["id"]
        _write_json(second_path, second)
        fixture.sync_aggregate()
        with self.assertRaisesRegex(HarborResultsError, "duplicate trial id"):
            load_rows(fixture.root)

    def test_rejects_cross_evidence_tampering(self):
        cases = []

        def trial_lock_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "lock.json"
            lock = json.loads(path.read_text())
            lock["task"]["digest"] = "sha256:" + "f" * 64
            _write_json(path, lock)

        cases.append(("trial lock", trial_lock_tamper))

        def checksum_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "result.json"
            result = json.loads(path.read_text())
            result["task_checksum"] = "f" * 64
            _write_json(path, result)
            fixture.sync_aggregate()

        cases.append(("checksum", checksum_tamper))

        def agent_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "result.json"
            result = json.loads(path.read_text())
            result["agent_info"]["name"] = "other-agent"
            _write_json(path, result)
            fixture.sync_aggregate()

        cases.append(("agent", agent_tamper))

        def environment_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "result.json"
            result = json.loads(path.read_text())
            result["config"]["environment"]["force_build"] = True
            _write_json(path, result)
            fixture.sync_aggregate()

        cases.append(("environment", environment_tamper))

        def reward_tamper(fixture: GoldenHarborJob):
            (fixture.trial() / "verifier" / "reward.txt").write_text("0.25\n")

        cases.append(("reward", reward_tamper))

        def atif_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "agent" / "trajectory.json"
            trajectory = json.loads(path.read_text())
            trajectory["final_metrics"]["total_prompt_tokens"] = 99
            _write_json(path, trajectory)

        cases.append(("ATIF", atif_tamper))

        def manifest_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "artifacts" / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest[1]["status"] = "failed"
            _write_json(path, manifest)

        cases.append(("manifest", manifest_tamper))

        def timing_tamper(fixture: GoldenHarborJob):
            path = fixture.trial() / "result.json"
            result = json.loads(path.read_text())
            result["verifier"]["started_at"] = "2026-07-20T10:00:00+00:00"
            _write_json(path, result)
            fixture.sync_aggregate()

        cases.append(("timing", timing_tamper))

        for index, (name, mutate) in enumerate(cases):
            with self.subTest(name=name):
                fixture = GoldenHarborJob(self.root / f"job-tamper-{index}")
                mutate(fixture)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_rejects_contradictory_job_statistics(self):
        for index, (field, value) in enumerate(
            (
                ("n_errored_trials", 1),
                ("n_cancelled_trials", 1),
                ("n_input_tokens", 299),
                ("n_cache_tokens", 74),
                ("n_output_tokens", 119),
                ("cost_usd", 1.4),
            )
        ):
            with self.subTest(field=field):
                fixture = GoldenHarborJob(self.root / f"job-stats-{index}")
                path = fixture.root / "result.json"
                result = json.loads(path.read_text())
                result["stats"][field] = value
                _write_json(path, result)
                with self.assertRaises(HarborResultsError):
                    load_rows(fixture.root)

    def test_rejects_multistep_states(self):
        fixture = GoldenHarborJob(self.root / "job-step_results")
        path = fixture.trial() / "result.json"
        result = json.loads(path.read_text())
        result["step_results"] = []
        _write_json(path, result)
        fixture.sync_aggregate()
        with self.assertRaises(HarborResultsError):
            load_rows(fixture.root)

    def test_cli_e2e_imports_and_reports_failures_without_partial_output(self):
        fixture = self.fixture()
        output = self.root / "cli-results.jsonl"
        command = [
            sys.executable,
            "-m",
            "obench.cli",
            "import",
            "harbor-results",
            str(fixture.root),
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=self.root,
            env=_SUBPROCESS_ENV,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Imported 3 Harbor trial(s)", completed.stdout)
        self.assertEqual(len(output.read_text().splitlines()), 3)

        bad_fixture = GoldenHarborJob(self.root / "bad-job")
        (bad_fixture.trial() / "agent" / "trajectory.json").unlink()
        bad_output = self.root / "bad-results.jsonl"
        failed = subprocess.run(
            [
                *command[:5],
                str(bad_fixture.root),
                "--output",
                str(bad_output),
            ],
            cwd=self.root,
            env=_SUBPROCESS_ENV,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(failed.returncode, 2)
        self.assertIn("ERROR:", failed.stderr)
        self.assertFalse(bad_output.exists())


if __name__ == "__main__":
    unittest.main()
