#!/usr/bin/env python3
"""Benchmark runner core for the agent-harness comparison.

For each (task, harness, trial) cell this runner:
  1. materializes the task workspace into a disposable temp dir (snapshot
     ``workspace/`` copy, or git ``workspace.toml`` archive export),
  2. dynamically imports the harness adapter and calls its ``run()`` per
     ADAPTER_SPEC.md (or uses the built-in ``null`` negative-control adapter),
  3. runs ``tasks/<task>/checker.sh`` with cwd=<temp dir> and env
     ``TASK_DIR=<absolute task dir>`` (checker exit 0 == task success),
  4. appends one JSON line describing the cell to the results log.

The loop is resumable: a cell whose ``run_id`` already appears in the results
log is skipped unless ``--force`` is given. Adapter exceptions are captured
into the row's ``error`` field rather than crashing the loop.

Python3 stdlib only. macOS-compatible (adapters enforce timeouts via
subprocess, never the ``timeout`` command).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager

from .bump_clis import (DOCKERFILE as CLI_PINS_DOCKERFILE, PIN_BY_KEY,
                        image_pin_mismatches, parse_image_pin_labels,
                        pinned_versions, reported_version, resolve_pin_key)
from .failure_class import classify_failure, classify_failure_reason
from .config import load_config
from .paths import (PACKAGE_DIR, SOURCE_ROOT, TasksDirError,
                    default_adapters_dir, default_results_path,
                    default_tasks_dir, ensure_package_path_on_sys_path,
                    resolve_tasks_dir)
from .scrub import build_context as build_scrub_context, scrub_text
from .workspace import (
    WorkspaceError,
    has_git_workspace,
    has_snapshot_workspace,
    materialize_workspace,
)

HERE = PACKAGE_DIR
REPO = SOURCE_ROOT  # checkout root for editable/source installs

DEFAULT_RESULTS_PATH = default_results_path()
DEFAULT_ADAPTERS_DIR = default_adapters_dir()
DEFAULT_TASKS_DIR = default_tasks_dir() or os.path.join(os.getcwd(), "tasks")
DEFAULT_MODEL = "gpt-5.5-medium"
DEFAULT_MAX_CONSECUTIVE_INFRA = 3
NEAR_ZERO_TOKEN_LIMIT = 100
INFRA_FAILURE_CLASSES = frozenset({"infra", "rate_limited"})
PREFLIGHT_TASK = "make-it-run"
PREFLIGHT_REQUIRED_FILES = ("instruction.md", "checker.sh")
PROXY_HARNESSES = {"codex", "pi", "claude", "opencode", "cursor", "devin", "grokbuild"}
PROXY_CODEX_SUBSCRIPTION_MODELS = {
    "gpt-5.5-medium", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
}
PROXY_CHAT_MODELS = {"glm-5.2", "glm-4.7-flash", "deepseek-v4-flash", "kimi-k2.7-code", "kimi-k3"}
PROXY_OPENROUTER_MODELS = {"laguna-s-2.1", "inkling"}
# Gateway/router arms (phase-1 spike): a canonical model name that names both the
# gateway and the model it fronts, run in fixed-model mode (single model, no
# router fallback). Metered through the proxy's ``gateway/<name>`` route. Kept
# here to mirror the existing per-model proxy gating; pi's adapter maps these to
# the gateway's model slug (obench/adapters/pi.py GATEWAY_MODELS).
PROXY_GATEWAY_MODELS = {
    f"{gw}/{slug}"
    for gw in ("openrouter", "vercel", "concentrate")
    for slug in ("openai/gpt-5.6", "anthropic/claude-sonnet-4.5")
}
PROXY_CLAUDE_MODELS = PROXY_CHAT_MODELS | {"claude-opus-4-8", "gpt-5.6-sol"}
CHECKER_CAPTURE_LIMIT = 8000
CHECKER_CAPTURE_TRUNCATED_PREFIX = "[truncated to last 8000 chars]\n"
WORKSPACE_EVIDENCE_MAX_BYTES = 2 * 1024 * 1024
WORKSPACE_EVIDENCE_MAX_TOTAL_BYTES = 8 * 1024 * 1024
WORKSPACE_EVIDENCE_MAX_FILES = 2000
WORKSPACE_EVIDENCE_META_KEY = "\0manifest"
CONTAINER_CLI_VERSIONS_PATH = "/etc/openbench-cli-versions.json"
_CONTAINER_CLI_VERSION_CACHE = {}

# Ordered field list for each results row. New provenance fields are appended
# so older logs that predate them stay readable (report derives a score from
# ``success`` when the field is absent).
ROW_FIELDS = (
    "run_id", "ts_iso", "harness", "model", "task", "trial",
    "success", "completed", "error", "wall_time_s", "t_env_setup_s", "t_agent_s", "t_checker_s", "tokens",
    "tokens_input_uncached", "tokens_cache_read", "tokens_cache_write",
    "tokens_output", "tokens_reasoning", "usage_raw", "token_basis",
    "tokens_proxy_input_uncached", "tokens_proxy_cache_read", "tokens_proxy_cache_write",
    "tokens_proxy_output", "tokens_proxy_reasoning", "tokens_proxy_calls",
    "sampling_observed", "token_basis_proxy", "proxy_capture_truncated",
    "tokens_fresh", "turns", "cmd", "checker_exit", "exec_mode", "score", "harness_version",
    "harness_version_source", "failure_class", "failure_reason", "workspace_changed", "checker_stdout", "checker_stderr", "checker_workspace_files",
    "image_digest", "candidate_provenance", "version_drift", "timeout_s",
    "workspace_source", "served_model", "cost", "upstream_cost",
)


class VersionDriftError(RuntimeError):
    """Raised when a local execution lane does not match Dockerfile CLI pins."""


def _pin_key_for_harness(harness, candidate=None):
    base = harness
    if candidate is not None:
        # Manifest proxy_adapter is accounting metadata; only config variants
        # actually execute their base_adapter CLI.
        base = getattr(candidate, "base_adapter", None)
    if base and base.startswith("codex_"):
        base = "codex"
    try:
        return resolve_pin_key(base)
    except (TypeError, ValueError):
        return None


def host_version_drift(harnesses, candidates=None, dockerfile=CLI_PINS_DOCKERFILE,
                       subprocess_runner=subprocess.run):
    """Return host-vs-pin mismatches for harnesses with Dockerfile CLI pins."""
    candidates = candidates or {}
    pins = pinned_versions(dockerfile)
    drift = []
    checked = set()
    for harness in harnesses:
        key = _pin_key_for_harness(harness, candidates.get(harness))
        if key is None or key in checked:
            continue
        checked.add(key)
        pin = PIN_BY_KEY[key]
        expected = pins.get(key)
        if expected is None:
            raise VersionDriftError(f"Dockerfile has no {pin.arg} pin for {harness}")
        try:
            proc = subprocess_runner(
                [pin.cli, "--version"], capture_output=True, text=True,
                timeout=15, stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            actual = None
            raw = exc.__class__.__name__
        else:
            raw = ((proc.stdout or "") + (proc.stderr or "")).strip()
            actual = reported_version(raw) if proc.returncode == 0 else None
        if actual != expected:
            drift.append({
                "harness": harness, "key": key, "package": pin.package,
                "cli": pin.cli, "expected": expected,
                "actual": actual or "unavailable", "raw": raw,
            })
    return drift


def image_version_drift(image, harnesses, candidates=None,
                        dockerfile=CLI_PINS_DOCKERFILE,
                        subprocess_runner=subprocess.run):
    """Return ``(mismatches, available)`` from one cheap Docker inspect."""
    try:
        proc = subprocess_runner(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", image],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return [], False
    if proc.returncode != 0:
        return [], False

    labels = parse_image_pin_labels(proc.stdout)
    pins = pinned_versions(dockerfile)
    keys = []
    for harness in harnesses:
        key = _pin_key_for_harness(harness, (candidates or {}).get(harness))
        if key is None or key in keys:
            continue
        if key not in pins:
            raise VersionDriftError(f"Dockerfile has no {PIN_BY_KEY[key].arg} pin for {harness}")
        keys.append(key)
    return image_pin_mismatches(pins, labels, keys), True


def version_drift_refusal(host_drift, image_drift=None,
                          image="openbench-harness:latest"):
    image_drift = image_drift or []
    lines = ["Refusing to start: CLI versions do not match Dockerfile pins."]
    for item in host_drift:
        lines.append(
            f"  {item['harness']}: host={item['actual']} pin={item['expected']} "
            f"({item['cli']} --version)"
        )
    for item in image_drift:
        lines.append(
            f"  {item['harness']}: image={item['actual']} pin={item['expected']}"
        )
    if host_drift:
        lines.append("Fix host CLIs: python3 -m obench.bump_clis --sync-host")
    if image_drift:
        lines.append(f"Fix the image: docker build -t {image} obench/docker")
    lines.extend([
        "Or update pins and build: python3 -m obench.bump_clis --apply",
        "To waive once (all rows will record version_drift=true): --allow-version-drift",
    ])
    return "\n".join(lines)


def make_run_id(harness, task, model, trial, candidate_digest=None):
    """Deterministic identity for a cell, including declarative candidate content."""
    group = f"{harness}@{candidate_digest[:12]}" if candidate_digest else harness
    return f"{group}:{task}:{model}:trial{trial}"


def truncate_checker_output(text, limit=CHECKER_CAPTURE_LIMIT):
    """Return ``text`` bounded to its last ``limit`` chars with a marker."""
    text = text or ""
    if len(text) <= limit:
        return text
    return CHECKER_CAPTURE_TRUNCATED_PREFIX + text[-limit:]


def _environment_values(environ=None):
    environ = environ or os.environ
    return sorted({value for value in environ.values() if value and len(value) >= 4},
                  key=len, reverse=True)


def redact_environment_values(text, environ=None):
    """Redact exact inherited environment values from checker output."""
    text = text or ""
    for value in _environment_values(environ):
        text = text.replace(value, "<REDACTED_ENV>")
    return text


class EnvValueRedactor:
    """Streaming exact-value redactor for inherited environment values."""

    def __init__(self, environ=None):
        self.values = _environment_values(environ)
        self.keep = max((len(v) for v in self.values), default=1) - 1
        self.pending = ""

    def feed(self, chunk):
        text = self.pending + (chunk or "")
        redacted = self._redact(text)
        if len(redacted) <= self.keep:
            self.pending = redacted
            return ""
        if self.keep:
            emit, self.pending = redacted[:-self.keep], redacted[-self.keep:]
        else:
            emit, self.pending = redacted, ""
        return emit

    def close(self):
        text = self._redact(self.pending)
        self.pending = ""
        return text

    def _redact(self, text):
        for value in self.values:
            text = text.replace(value, "<REDACTED_ENV>")
        return text


def redact_truncation_boundary(text):
    """Redact the first partial token retained after tail truncation.

    If truncation cuts through a secret, the retained suffix starts immediately
    after the marker and may not match the original env value or generic secret
    regexes. Over-redact that leading token fragment before persistence.
    """
    if not text.startswith(CHECKER_CAPTURE_TRUNCATED_PREFIX):
        return text
    body = text[len(CHECKER_CAPTURE_TRUNCATED_PREFIX):]
    # If truncation retained the suffix of a secret token, exact/pattern scrubbers
    # may not recognize it. Redact a plausible leading secret fragment while
    # leaving punctuation-only tails (common in tests/log separators) intact.
    body = re.sub(
        r"^(\s*)(?=\S{8,})(?=\S*[A-Za-z0-9])\S+",
        r"\1<REDACTED_BOUNDARY>", body, count=1,
    )
    return CHECKER_CAPTURE_TRUNCATED_PREFIX + body


def scrub_workspace_evidence_paths(evidence):
    """Redact sensitive relative paths in workspace evidence keys."""
    if not isinstance(evidence, dict):
        return evidence
    try:
        context = build_scrub_context()
        scrubbed = {}
        for path, item in evidence.items():
            if path == WORKSPACE_EVIDENCE_META_KEY:
                key_base = path
            else:
                key_base = scrub_text(redact_environment_values(str(path)), context)
                key_base = key_base or "<REDACTED_PATH>"
            key = key_base
            n = 2
            while key in scrubbed:
                key = f"{key_base}#{n}"
                n += 1
            scrubbed[key] = item
        return scrubbed
    except Exception:  # noqa: BLE001 - never fail a row while redacting paths
        return {WORKSPACE_EVIDENCE_META_KEY: {"skipped": "path_redaction_failed"}}


def scrub_checker_output(text):
    """Redact high-risk local/secret tokens before persisting checker output."""
    try:
        redacted = redact_environment_values(text or "")
        redacted = scrub_text(redacted, build_scrub_context())
        return redact_truncation_boundary(redacted)
    except Exception:  # noqa: BLE001 - never persist unsanitized checker output
        return "<CHECKER_OUTPUT_REDACTION_FAILED>"


class TailCapture:
    """Bounded text capture that keeps only the final ``limit`` chars."""

    def __init__(self, limit=CHECKER_CAPTURE_LIMIT):
        self.limit = limit
        self.total_chars = 0
        self.tail = ""

    def append(self, chunk):
        if not chunk:
            return
        self.total_chars += len(chunk)
        self.tail = (self.tail + chunk)[-self.limit:]

    def text(self):
        if self.total_chars <= self.limit:
            return self.tail
        return CHECKER_CAPTURE_TRUNCATED_PREFIX + self.tail


class StreamingScoreParser:
    """Incrementally parse SCORE lines without retaining full stdout."""

    def __init__(self, max_line_chars=CHECKER_CAPTURE_LIMIT):
        self.score = None
        self.max_line_chars = max_line_chars
        self._buf = ""

    def feed(self, chunk):
        self._buf += chunk
        lines = self._buf.split("\n")
        self._buf = lines.pop()
        if len(self._buf) > self.max_line_chars:
            self._buf = self._buf[-self.max_line_chars:]
        for line in lines:
            self._consume_line(line)

    def close(self):
        if self._buf:
            self._consume_line(self._buf)
            self._buf = ""
        return self.score

    def _consume_line(self, line):
        line = line.strip()
        if not line.startswith("SCORE:"):
            return
        try:
            val = float(line[len("SCORE:"):].strip())
        except ValueError:
            return
        self.score = max(0.0, min(1.0, val))


def _read_stream(pipe, capture, score_parser=None, redactor=None):
    try:
        with pipe:
            read1 = getattr(getattr(pipe, "buffer", None), "read1", None)
            while True:
                if read1 is not None:
                    data = read1(4096)
                    chunk = data.decode("utf-8", errors="replace") if data else ""
                else:
                    chunk = pipe.read(4096)
                if not chunk:
                    break
                safe_chunk = redactor.feed(chunk) if redactor is not None else chunk
                capture.append(safe_chunk)
                if score_parser is not None:
                    score_parser.feed(chunk)
    except Exception:  # noqa: BLE001 - stream capture must not crash a cell
        pass
    finally:
        if redactor is not None:
            capture.append(redactor.close())


def parse_score(stdout):
    """Return the partial-credit score from a checker's stdout, or None.

    A checker MAY print ``SCORE: <float>`` lines; the **last parseable** one
    wins. Values are clamped to [0.0, 1.0]. A malformed value (not a float) is
    ignored as if absent, so a trailing garbage line can't erase an earlier
    valid score. Returns None when no parseable SCORE line is present.
    """
    score = None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("SCORE:"):
            continue
        try:
            val = float(line[len("SCORE:"):].strip())
        except ValueError:
            continue  # malformed -> treat this line as absent
        score = max(0.0, min(1.0, val))
    return score


def _extract_version(module):
    """Call a module's optional ``version() -> str | None`` defensively.

    Returns the string it yields, or None if the function is missing, returns a
    non-string, or raises. Never propagates an adapter's error.
    """
    fn = getattr(module, "version", None)
    if not callable(fn):
        return None
    try:
        v = fn()
    except Exception:  # noqa: BLE001 - a broken version() must not fail the run
        return None
    return v if isinstance(v, str) else None


def probe_version(harness, adapters_dir, candidate=None):
    """Best-effort host harness version string for local-mode row stamping.

    The built-in ``null`` control reports ``"builtin"``. Real harnesses import
    their adapter and call its optional ``version()``; any failure yields None.
    Callers should cache this (one probe per harness per invocation).
    """
    if harness == "null":
        return "builtin"
    if candidate is not None:
        try:
            return candidate.version()
        except Exception:  # noqa: BLE001
            return None
    try:
        module = load_adapter(adapters_dir, harness)
    except Exception:  # noqa: BLE001
        return None
    return _extract_version(module)


def parse_container_cli_versions(text):
    """Parse ``/etc/openbench-cli-versions.json`` into ``{harness: version}``.

    The Docker image writes a simple JSON object with harness adapter names as
    keys (for example ``grokbuild`` for the ``grok`` CLI). Only string values are
    accepted; malformed JSON, arrays, or non-string values yield an empty dict
    so a broken/old image is visible as ``harness_version=null`` rather than
    crashing an in-flight benchmark cell.
    """
    try:
        data = json.loads(text or "")
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, str)}


def read_container_cli_versions(image, image_digest=None):
    """Return CLI versions stamped inside a Docker image, cached per digest.

    Reads ``CONTAINER_CLI_VERSIONS_PATH`` with ``docker run --rm <image> cat``.
    The cache key is the immutable image digest/ID when available, falling back
    to the image ref only when Docker cannot provide one (for example in unit
    tests or a broken daemon path).
    """
    if not image and not image_digest:
        return {}
    cache_key = image_digest or docker_image_digest(image) or image
    if cache_key in _CONTAINER_CLI_VERSION_CACHE:
        return dict(_CONTAINER_CLI_VERSION_CACHE[cache_key])
    image_ref = image_digest or image
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", image_ref, "cat", CONTAINER_CLI_VERSIONS_PATH],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        versions = {}
    else:
        versions = parse_container_cli_versions(proc.stdout) if proc.returncode == 0 else {}
    _CONTAINER_CLI_VERSION_CACHE[cache_key] = dict(versions)
    return versions


def harness_version_for_source(harness, exec_used, host_version, docker_image=None,
                               image_digest=None,
                               container_versions_reader=read_container_cli_versions):
    """Return ``(version, source)`` for a row after the actual exec lane is known."""
    if exec_used == "docker":
        if harness == "null":
            return "builtin", "container"
        versions = container_versions_reader(docker_image, image_digest)
        # Ablation variants (codex_v1, codex_v2, ...) run the stock codex CLI
        # with a different CODEX_HOME; the binary version is codex's.
        base = harness.split("_", 1)[0] if harness.startswith("codex_") else harness
        return versions.get(harness) or versions.get(base), "container"
    return host_version, "host"


def null_run(instruction, workdir, model, timeout_s):
    """Built-in negative-control adapter: does nothing, claims to complete.

    Because it never edits ``workdir``, the task checker must fail, so every
    cell run with ``--harness null`` should record ``success=false``.
    """
    return {
        "completed": True,
        "error": None,
        "output_tail": "",
        "tokens": None,
        "turns": None,
        "cmd": "null",
    }


def load_adapter(adapters_dir, name):
    """Dynamically import ``<adapters_dir>/<name>.py`` and return the module.

    The module must expose ``run(instruction, workdir, model, timeout_s)`` per
    ADAPTER_SPEC.md. Raises ``FileNotFoundError`` if the adapter file is absent.
    """
    ensure_package_path_on_sys_path()
    path = os.path.join(adapters_dir, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(f"bench_adapter_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"adapter '{name}' has no run() function")
    return module


def _raise_with_exec_used(exc, exec_used, env_setup_s=None, agent_wall_time_s=None):
    setattr(exc, "bench_exec_used", exec_used)
    if isinstance(env_setup_s, (int, float)) and env_setup_s >= 0:
        setattr(exc, "bench_env_setup_s", env_setup_s)
    if isinstance(agent_wall_time_s, (int, float)) and agent_wall_time_s >= 0:
        setattr(exc, "bench_agent_wall_time_s", agent_wall_time_s)
    raise exc


def _with_phase_timings(result, env_setup_s=None, agent_wall_time_s=None):
    """Attach runner phase timing hints to an adapter result when known."""
    if not isinstance(result, dict):
        return result
    if (not isinstance(env_setup_s, (int, float))
            and not isinstance(agent_wall_time_s, (int, float))):
        return result
    result = dict(result)
    if isinstance(env_setup_s, (int, float)) and env_setup_s >= 0:
        result["host_env_setup_s"] = env_setup_s
    if isinstance(agent_wall_time_s, (int, float)) and agent_wall_time_s >= 0:
        result["host_agent_wall_time_s"] = agent_wall_time_s
    return result


@contextmanager
def _temporary_environ(updates):
    """Temporarily overlay environment variables for one adapter call."""
    if not updates:
        yield
        return
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _proxy_env(proxy_ctx, cell_token, for_docker=False):
    if not proxy_ctx:
        return None
    base = proxy_ctx["docker_base_url"] if for_docker else proxy_ctx["local_base_url"]
    return {
        "OPENBENCH_PROXY": "1",
        "OPENBENCH_PROXY_BASE_URL": base,
        "OPENBENCH_PROXY_CELL_TOKEN": cell_token,
    }


def _proxy_docker_args(proxy_ctx):
    if not proxy_ctx:
        return None
    return ["--add-host", "host.docker.internal:host-gateway"]


def proxy_supported_for_cell(harness, model):
    """True when --proxy has proven adapter wiring for this harness/model."""
    if harness == "codex":
        # Open models reach the LiteLLM bridge through the proxy's ``bridge``
        # route (adapters/codex.py _bridge_base_url), so they meter too.
        return (model in PROXY_CODEX_SUBSCRIPTION_MODELS
                or model in PROXY_CHAT_MODELS)
    if harness == "pi":
        # Open models route via the provider extension's proxied baseUrl
        # (adapters/pi.py _pi_provider_ext), same mechanism as opencode.
        # Gateway arms use the same extension pointed at the proxy's
        # gateway/<name> route (adapters/pi.py GATEWAY_MODELS).
        return (model in PROXY_CODEX_SUBSCRIPTION_MODELS
                or model in PROXY_CHAT_MODELS
                or model in PROXY_OPENROUTER_MODELS
                or model in PROXY_GATEWAY_MODELS)
    if harness == "claude":
        return model in PROXY_CLAUDE_MODELS
    if harness == "opencode":
        return model in PROXY_CHAT_MODELS or model in PROXY_OPENROUTER_MODELS
    if harness == "grokbuild":
        return model in {"glm-5.2", "deepseek-v4-flash", "kimi-k2.7-code", "kimi-k3", "laguna-s-2.1", "inkling", "gpt-5.6-sol"}
    # Cursor's model stream requires its private HTTP/2 agent protocol, which
    # the stdlib HTTP/1.1 proxy cannot meter; Devin performs inference behind
    # Cognition's cloud boundary. See both adapter docstrings.
    return False


def _proxy_sampling_for_cell(harness, model):
    """Non-secret sampling metadata requested by the adapter, for ledger context."""
    subscription_models = {
        "gpt-5.5-medium": "gpt-5.5",
        "gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-5.6-terra": "gpt-5.6-terra",
        "gpt-5.6-luna": "gpt-5.6-luna",
    }
    if harness == "codex" and model in subscription_models:
        return {"model": subscription_models[model], "reasoning_effort": "medium"}
    if harness == "pi" and model in subscription_models:
        return {"provider": "openai-codex", "model": subscription_models[model], "thinking": "medium"}
    if harness == "pi" and model in PROXY_GATEWAY_MODELS:
        # Record the requested (fixed) model slug for the ledger; the proxy
        # additionally records the served_model the gateway reports back.
        return {"model": model.split("/", 1)[1], "thinking": "medium"}
    if harness == "opencode" and model in PROXY_CHAT_MODELS | PROXY_OPENROUTER_MODELS:
        return {"model": model, "variant": "medium"}
    if harness == "claude" and model in PROXY_CLAUDE_MODELS:
        return {"model": model, "effort": "medium"}
    if harness == "grokbuild" and proxy_supported_for_cell(harness, model):
        return {"model": model, "reasoning_effort": "medium"}
    return {}


def _write_proxy_cell_metadata(proxy_ctx, cell_token, harness, model):
    if not proxy_ctx or not cell_token:
        return
    ledger_dir = proxy_ctx.get("ledger_dir")
    if not ledger_dir:
        return
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cell_token)
    meta = {
        "source": "runner_configured",
        "harness": harness,
        "model": model,
        "sampling": _proxy_sampling_for_cell(harness, model),
    }
    try:
        os.makedirs(str(ledger_dir), exist_ok=True)
        with open(os.path.join(str(ledger_dir), safe + ".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, sort_keys=True)
    except OSError:
        pass


def task_docker_spec(task_dir):
    """Return an optional pinned image/workdir declared by task.toml."""
    path = os.path.join(task_dir, "task.toml")
    if not os.path.isfile(path):
        return None, "/work"
    try:
        import tomllib
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, ValueError):
        return None, "/work"
    return data.get("docker_image"), data.get("docker_workdir", "/work")


def hydrate_image_workdir(image, container_workdir, workdir):
    """Copy a task image's native workdir into the disposable host workspace."""
    try:
        proc = subprocess.run(
            ["docker", "create", image], capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timed out creating task image {image!r}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"cannot create task image {image!r}: {proc.stderr.strip()}")
    container = proc.stdout.strip()
    try:
        try:
            copied = subprocess.run(
                ["docker", "cp", f"{container}:{container_workdir}/.", workdir],
                capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"timed out hydrating {container_workdir}") from exc
        if copied.returncode != 0:
            raise RuntimeError(f"cannot hydrate {container_workdir}: {copied.stderr.strip()}")
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            pass


def invoke_adapter(exec_mode, harness, instruction, workdir, model, timeout_s,
                   adapters_dir, docker_image, docker_fallback,
                   proxy_ctx=None, cell_token=None, candidate=None,
                   container_workdir="/work"):
    """Run the harness for one cell, honoring the execution mode.

    Returns ``(result_dict, exec_used)`` where ``exec_used`` is ``"local"`` or
    ``"docker"``. In ``docker`` mode each cell runs in a fresh container (the
    same adapter, unchanged, via ``entry.py``); if the daemon or image is
    unavailable and ``docker_fallback`` is set, execution falls back to local
    and ``exec_used`` reflects that. The built-in ``null`` control runs the same
    way in either mode (its container path proves the plumbing without auth).
    """
    fallback_env_setup_s = None
    if exec_mode == "docker":
        from . import docker_exec  # lazy: local mode never needs docker
        docker_start = time.monotonic()
        try:
            result = docker_exec.run_in_container(
                harness, instruction, workdir, model, timeout_s,
                adapters_dir, docker_image,
                extra_docker_args=_proxy_docker_args(proxy_ctx),
                extra_env=_proxy_env(proxy_ctx, cell_token, for_docker=True),
                candidate_spec_bytes=(candidate.spec_bytes if candidate is not None else None),
                candidate_auth_files=candidate.auth_files if candidate is not None else None,
                candidate_pass_env=candidate.pass_env if (candidate is not None and candidate.kind == "manifest") else None,
                candidate_config_dir=(candidate.config_dir if candidate is not None
                                      and candidate.kind == "config-variant" else None),
                candidate_config_contents=(candidate.config_contents if candidate is not None
                                           and candidate.kind == "config-variant" else None),
                candidate_inherit_env=(candidate.inherit_env if candidate is not None
                                       and candidate.kind == "manifest" else False),
                # A manifest's proxy_adapter is accounting metadata only; it
                # must never grant that stock adapter's credentials.
                base_harness=candidate.base_adapter if candidate is not None else None,
                candidate_persist_auth=bool(
                    candidate is not None and getattr(candidate, "persist_auth", False)),
                container_workdir=container_workdir,
            )
            return result, "docker"
        except docker_exec.DockerUnavailable as exc:
            fallback_env_setup_s = round(time.monotonic() - docker_start, 3)
            if not docker_fallback:
                _raise_with_exec_used(
                    exc, "docker", env_setup_s=fallback_env_setup_s,
                    agent_wall_time_s=0.0)
            print(f"WARN docker unavailable ({exc}); falling back to local")
        except Exception as exc:  # noqa: BLE001 - caller records row failure
            _raise_with_exec_used(
                exc,
                "docker",
                env_setup_s=getattr(exc, "bench_env_setup_s", None),
                agent_wall_time_s=getattr(
                    exc, "bench_agent_wall_time_s",
                    round(time.monotonic() - docker_start, 3),
                ),
            )

    local_start = time.monotonic()
    try:
        with _temporary_environ(_proxy_env(proxy_ctx, cell_token, for_docker=False)):
            if harness == "null":
                result = null_run(instruction, workdir, model, timeout_s)
            else:
                adapter = candidate or load_adapter(adapters_dir, harness)
                result = adapter.run(instruction, workdir, model, timeout_s)
        if fallback_env_setup_s is not None:
            result = _with_phase_timings(
                result,
                env_setup_s=fallback_env_setup_s,
                agent_wall_time_s=round(time.monotonic() - local_start, 3),
            )
        return result, "local"
    except Exception as exc:  # noqa: BLE001 - caller records row failure
        _raise_with_exec_used(
            exc,
            "local",
            env_setup_s=fallback_env_setup_s,
            agent_wall_time_s=round(time.monotonic() - local_start, 3),
        )


def read_instruction(task_dir):
    """Return the contents of ``<task_dir>/instruction.md``."""
    with open(os.path.join(task_dir, "instruction.md"), encoding="utf-8") as fh:
        return fh.read()


def capture_workspace_files(workdir, max_bytes=WORKSPACE_EVIDENCE_MAX_BYTES,
                            max_total_bytes=WORKSPACE_EVIDENCE_MAX_TOTAL_BYTES,
                            max_files=WORKSPACE_EVIDENCE_MAX_FILES):
    """Return bounded recursive evidence for files visible to the checker.

    Paths are workspace-relative POSIX strings. Regular files up to
    ``max_bytes`` get sha256, byte size, and mtime; larger or unsafe entries are
    listed as skipped without hashing. Traversal uses directory/file descriptors
    with no-follow opens so symlink races cannot escape the workspace.
    """
    evidence = {}
    hashed_bytes = 0
    entries_seen = 0

    def add_meta(reason):
        evidence[WORKSPACE_EVIDENCE_META_KEY] = {"skipped": reason}

    def add(rel, item):
        if len(evidence) >= max_files:
            add_meta(f"file_count_limit>{max_files}")
            return False
        evidence[rel] = item
        return True

    def rel_join(parent, name):
        return f"{parent}/{name}" if parent else name

    dir_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)

    try:
        root_fd = os.open(workdir, dir_flags | nofollow)
    except OSError as exc:
        add_meta(f"scan_error:{type(exc).__name__}")
        return evidence

    stack = [("", root_fd)]

    def finish():
        while stack:
            _, pending_fd = stack.pop()
            try:
                os.close(pending_fd)
            except OSError:
                pass
        return evidence

    while stack:
        rel_dir, dir_fd = stack.pop()
        try:
            try:
                names = sorted(os.listdir(dir_fd))
            except OSError as exc:
                target = rel_dir or "."
                if not add(target, {"skipped": f"scan_error:{type(exc).__name__}"}):
                    return finish()
                continue
            for name in names:
                if name in {".git", "__pycache__"}:
                    continue
                entries_seen += 1
                if entries_seen > max_files:
                    add_meta(f"entry_count_limit>{max_files}")
                    return finish()
                rel = rel_join(rel_dir, name)
                try:
                    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError as exc:
                    if not add(rel, {"skipped": f"stat_error:{type(exc).__name__}"}):
                        return finish()
                    continue
                item = {"bytes": st.st_size, "mtime": st.st_mtime}
                if stat.S_ISDIR(st.st_mode):
                    try:
                        child_fd = os.open(name, dir_flags | nofollow, dir_fd=dir_fd)
                        child_st = os.fstat(child_fd)
                        if not stat.S_ISDIR(child_st.st_mode):
                            os.close(child_fd)
                            item["skipped"] = "not_directory"
                            if not add(rel, item):
                                return finish()
                        else:
                            stack.append((rel, child_fd))
                    except OSError as exc:
                        item["skipped"] = f"open_dir_error:{type(exc).__name__}"
                        if not add(rel, item):
                            return finish()
                    continue
                if not stat.S_ISREG(st.st_mode):
                    item["skipped"] = "not_regular"
                    if not add(rel, item):
                        return finish()
                    continue
                if st.st_size > max_bytes:
                    item["skipped"] = f"too_large>{max_bytes}"
                    if not add(rel, item):
                        return finish()
                    continue
                if hashed_bytes + st.st_size > max_total_bytes:
                    item["skipped"] = f"total_bytes_limit>{max_total_bytes}"
                    if not add(rel, item):
                        return finish()
                    continue
                try:
                    h = hashlib.sha256()
                    fd = os.open(name, file_flags, dir_fd=dir_fd)
                    try:
                        opened_st = os.fstat(fd)
                        item = {"bytes": opened_st.st_size, "mtime": opened_st.st_mtime}
                        if not stat.S_ISREG(opened_st.st_mode):
                            item["skipped"] = "not_regular"
                        elif opened_st.st_size > max_bytes:
                            item["skipped"] = f"too_large>{max_bytes}"
                        elif hashed_bytes + opened_st.st_size > max_total_bytes:
                            item["skipped"] = f"total_bytes_limit>{max_total_bytes}"
                        else:
                            remaining = opened_st.st_size
                            while remaining > 0:
                                chunk = os.read(fd, min(1024 * 1024, remaining))
                                if not chunk:
                                    item["skipped"] = "changed_during_read"
                                    break
                                h.update(chunk)
                                remaining -= len(chunk)
                            if "skipped" not in item:
                                item["sha256"] = h.hexdigest()
                                hashed_bytes += opened_st.st_size
                    finally:
                        os.close(fd)
                except OSError as exc:
                    item["skipped"] = f"read_error:{type(exc).__name__}"
                if not add(rel, item):
                    return finish()
        finally:
            os.close(dir_fd)
    return evidence


def docker_image_digest(image):
    """Best-effort immutable identity for the docker image used by a cell."""
    if not image:
        return None
    for fmt in ("{{index .RepoDigests 0}}", "{{.Id}}"):
        try:
            proc = subprocess.run(
                ["docker", "inspect", "--format", fmt, image],
                capture_output=True, text=True, timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        value = (proc.stdout or "").strip()
        if proc.returncode == 0 and value and value != "<no value>":
            return value
    return None


def _kill_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        try:
            proc.kill()
        except OSError:
            pass


def _descendant_pids(root_pid):
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, timeout=2,
        )
    except Exception:  # noqa: BLE001 - best-effort cleanup helper
        return set()
    if proc.returncode != 0:
        return set()
    children = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(ppid, set()).add(pid)
    found = set()
    stack = list(children.get(root_pid, ()))
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


def _track_descendants(root_pid, seen, stop_event):
    while not stop_event.is_set():
        seen.update(_descendant_pids(root_pid))
        stop_event.wait(0.05)


def _kill_pids(pids):
    current = os.getpid()
    for pid in sorted(set(pids), reverse=True):
        if pid <= 1 or pid == current:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _close_pipe(pipe):
    if pipe is None:
        return
    try:
        pipe.close()
    except OSError:
        pass


def run_checker(task_dir, workdir, timeout_s, checker_env=None):
    """Run ``<task_dir>/checker.sh`` with cwd=workdir and TASK_DIR set.

    Returns ``(checker_exit, raw_score, stdout, stderr)`` where
    ``checker_exit`` is the integer exit code (or the string ``"timeout"``) and
    ``raw_score`` is the float from the checker's last parseable ``SCORE:`` line,
    or None if it printed none. Captured stdout/stderr are bounded tails. The
    checker decides task success (exit 0 == success); the adapter never does.
    ``checker_env`` may provide a caller-owned sanitized environment.
    """
    checker = os.path.join(task_dir, "checker.sh")
    env = dict(os.environ) if checker_env is None else dict(checker_env)
    env["TASK_DIR"] = os.path.abspath(task_dir)
    env.pop("OPENBENCH_SOLUTION_OVERLAY", None)
    stdout_capture = TailCapture()
    stderr_capture = TailCapture()
    score_parser = StreamingScoreParser()
    proc = subprocess.Popen(
        ["bash", "-c", 'bash "$1"; rc=$?; sleep 0.1; exit "$rc"',
         "checker-wrapper", checker],
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    descendant_pids = set()
    tracker_stop = threading.Event()
    tracker_thread = threading.Thread(
        target=_track_descendants, args=(proc.pid, descendant_pids, tracker_stop),
        daemon=True,
    )
    tracker_thread.start()
    stdout_thread = threading.Thread(
        target=_read_stream,
        args=(proc.stdout, stdout_capture, score_parser, EnvValueRedactor()),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream,
        args=(proc.stderr, stderr_capture, None, EnvValueRedactor()),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        _kill_pids(descendant_pids | _descendant_pids(proc.pid))
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        returncode = "timeout"
    finally:
        descendant_pids.update(_descendant_pids(proc.pid))
        tracker_stop.set()
        tracker_thread.join(timeout=1)
        _kill_pids(descendant_pids)

    # If the checker shell exited but background descendants inherited the
    # pipes, reap the whole checker process group so capture cannot leak threads
    # or processes beyond this cell.
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        _kill_process_group(proc)
        _kill_pids(descendant_pids)
        _close_pipe(proc.stdout)
        _close_pipe(proc.stderr)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    raw_score = None if returncode == "timeout" else score_parser.close()
    return returncode, raw_score, stdout_capture.text(), stderr_capture.text()


class ResultsLogError(RuntimeError):
    """Raised when a results JSONL cannot be safely resumed."""


def load_existing_run_ids(results_path):
    """Return the set of ``run_id`` values already present in the results log.

    Corrupt (non-JSON) lines fail closed: silent skip would drop those run_ids
    from the resume set and risk duplicate cells on the next append.
    """
    ids = set()
    if not os.path.isfile(results_path):
        return ids
    invalid = 0
    line_no = 0
    with open(results_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if not isinstance(row, dict):
                invalid += 1
                continue
            rid = row.get("run_id")
            if rid is not None:
                ids.add(rid)
    if invalid:
        raise ResultsLogError(
            f"{results_path} has {invalid} corrupt JSONL line(s) "
            f"(last scan ended near line {line_no}); fix the file or pass "
            "--force to ignore resume state"
        )
    return ids


def append_row(results_path, row):
    """Append one JSON line (ordered by ROW_FIELDS) to the results log."""
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    ordered = {key: row.get(key) for key in ROW_FIELDS}
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ordered) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def preflight_results_path(results_path):
    """Return the sidecar path for an invocation's smoke-test row."""
    stem, _extension = os.path.splitext(results_path)
    return stem + ".preflight.jsonl"


def is_near_zero_infra(row):
    """True for infra/rate-limit failures with absent or sub-100 token spend."""
    if row.get("failure_class") not in INFRA_FAILURE_CLASSES:
        return False
    tokens = row.get("tokens")
    return tokens is None or (
        isinstance(tokens, (int, float)) and not isinstance(tokens, bool)
        and tokens < NEAR_ZERO_TOKEN_LIMIT
    )


def row_error_summary(row):
    """Return a compact diagnostic for a reliability-gate refusal."""
    for key in ("error", "output_tail", "checker_stderr", "checker_stdout"):
        value = row.get(key)
        if value:
            lines = [line.strip() for line in str(value).splitlines() if line.strip()]
            if lines:
                return lines[-1][-500:]
    return "no error detail reported"


def circuit_breaker_message(streak, row):
    return (
        f"ABORT: infra circuit breaker tripped after {streak} consecutive "
        "near-zero-token infra/rate_limited cells; "
        f"last_error={row_error_summary(row)}"
    )


def preflight_refusal_message(row):
    return (
        "Refusing to start: preflight smoke ended with "
        f"failure_class={row.get('failure_class')} tokens={row.get('tokens')}; "
        f"last_error={row_error_summary(row)}. "
        "Use --allow-preflight-failure to override."
    )


def _task_is_preflight_candidate(task_dir):
    if not all(
        os.path.exists(os.path.join(task_dir, name)) for name in PREFLIGHT_REQUIRED_FILES
    ):
        return False
    return has_snapshot_workspace(task_dir) or has_git_workspace(task_dir)


def select_preflight_task(tasks_dir):
    """Pick a smoke task from ``tasks_dir``.

    Prefer ``make-it-run`` when present (OpenBench checkout behavior). Otherwise
    use the first runnable task (sorted by display name). Raise ``TasksDirError``
    when nothing suitable exists.
    """
    from .validate_tasks import discover_tasks

    if not os.path.isdir(tasks_dir):
        raise TasksDirError(
            f"preflight smoke: tasks directory not found: {tasks_dir}"
        )
    names = []
    for _tier, name, task_dir in discover_tasks([("tasks", tasks_dir)]):
        if _task_is_preflight_candidate(task_dir):
            names.append(name)
    if PREFLIGHT_TASK in names:
        return PREFLIGHT_TASK
    if names:
        return names[0]
    raise TasksDirError(
        "preflight smoke: no runnable task found under "
        f"{tasks_dir}. Add a task with instruction.md, workspace/ (or "
        f"workspace.toml), and checker.sh, or place {PREFLIGHT_TASK!r} there."
    )


def default_transcripts_dir(results_path):
    """Base dir for transcripts: a ``transcripts/`` sibling of the results log.

    Co-locating transcripts with their results log keeps them together and
    means an ephemeral (temp) results path parks its transcripts in the same
    ephemeral tree -- so nothing leaks into the repo during tests. Override with
    ``--transcripts-dir``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(results_path)),
                        "transcripts")


def transcript_path(transcripts_dir, results_stem, run_id):
    """Local path for a cell's transcript: <base>/<results-stem>/<run_id>.txt.

    ``run_id`` contains ``:`` separators; sanitize to a filesystem-safe token so
    the file name is portable and unambiguous.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", run_id)
    return os.path.join(transcripts_dir, results_stem, safe + ".txt")


def write_transcript(path, row, body):
    """Write one cell's full agent transcript to ``path`` (creating dirs).

    LOCAL-ONLY (user directive): transcripts are the raw, UNSCRUBBED harness
    output and may contain absolute home paths, usernames, hostnames, or leaked
    secrets. They are never published as-is -- run ``obench/scrub.py --check``
    for a manual review pass, then ``obench/scrub.py`` to emit scrubbed copies,
    before sharing any transcript. The runner writes originals here and builds
    no publishing path of any kind.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = (
        f"# transcript {row['run_id']}\n"
        f"# harness={row['harness']} model={row['model']} "
        f"task={row['task']} trial={row['trial']} ts={row['ts_iso']}\n"
        "# LOCAL-ONLY -- unscrubbed. Review with obench/scrub.py --check before sharing.\n\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(body or "")


def _adapter_wall_time_s(start_monotonic, result, exec_used):
    """Elapsed adapter wall time, honoring docker_exec's own clock if larger."""
    elapsed = time.monotonic() - start_monotonic
    if exec_used == "docker" and isinstance(result, dict):
        host_elapsed = result.get("host_wall_time_s")
        if isinstance(host_elapsed, (int, float)) and host_elapsed >= 0:
            elapsed = max(elapsed, host_elapsed)
    return round(elapsed, 3)


def _agent_wall_time_s(start_monotonic, result, exec_used):
    """Elapsed agent/CLI wall time, excluding setup/preflight when available."""
    elapsed = time.monotonic() - start_monotonic
    if isinstance(result, dict):
        host_elapsed = result.get("host_agent_wall_time_s")
        if isinstance(host_elapsed, (int, float)) and host_elapsed >= 0:
            elapsed = host_elapsed
    return round(elapsed, 3)


def _add_host_env_setup_s(row, result, exec_used):
    """Fold host-side preflight/staging timing into the setup phase."""
    if not isinstance(result, dict):
        return
    host_setup = result.get("host_env_setup_s")
    if isinstance(host_setup, (int, float)) and host_setup >= 0:
        row["t_env_setup_s"] = round((row.get("t_env_setup_s") or 0.0) + host_setup, 3)


def _num(value, default=None):
    return int(value) if isinstance(value, (int, float)) else default


def _empty_proxy_usage():
    return {
        "tokens_proxy_input_uncached": None,
        "tokens_proxy_cache_read": None,
        "tokens_proxy_cache_write": None,
        "tokens_proxy_output": None,
        "tokens_proxy_reasoning": None,
    }


def proxy_split_from_usage(usage):
    """Normalize provider usage JSON to the row's proxy split fields."""
    out = _empty_proxy_usage()
    if not isinstance(usage, dict):
        return out

    # pi-normalized shape: input, cacheRead, cacheWrite, output, reasoning.
    if {"input", "output"} & set(usage) and "totalTokens" in usage:
        inp = _num(usage.get("input"))
        out_tok = _num(usage.get("output"))
        if inp is not None and out_tok is not None:
            out.update({
                "tokens_proxy_input_uncached": inp,
                "tokens_proxy_cache_read": _num(usage.get("cacheRead"), 0),
                "tokens_proxy_cache_write": _num(usage.get("cacheWrite"), 0),
                "tokens_proxy_output": out_tok,
                "tokens_proxy_reasoning": _num(usage.get("reasoning")),
            })
        return out

    # OpenAI/Codex Responses and Anthropic Messages shapes.
    if "input_tokens" in usage or "output_tokens" in usage:
        inp = _num(usage.get("input_tokens"))
        details = usage.get("input_tokens_details") or {}
        anthropic_cache_shape = (
            "cache_read_input_tokens" in usage
            or "cache_creation_input_tokens" in usage
        )
        cache_read = _num(usage.get("cache_read_input_tokens"), None)
        if cache_read is None:
            cache_read = _num(usage.get("cached_input_tokens"), None)
        if cache_read is None and isinstance(details, dict):
            cache_read = _num(details.get("cached_tokens"), 0)
        cache_write = _num(
            usage.get("cache_creation_input_tokens")
            or usage.get("cache_write_tokens")
            or (details.get("cache_write_tokens") if isinstance(details, dict) else None),
            0,
        )
        out_tok = _num(usage.get("output_tokens"))
        out_details = usage.get("output_tokens_details") or {}
        reasoning = _num(usage.get("reasoning_output_tokens"), None)
        if reasoning is None and isinstance(out_details, dict):
            reasoning = _num(out_details.get("reasoning_tokens"))
        if inp is not None and out_tok is not None:
            input_uncached = inp if anthropic_cache_shape else max(0, inp - (cache_read or 0) - (cache_write or 0))
            out.update({
                "tokens_proxy_input_uncached": input_uncached,
                "tokens_proxy_cache_read": cache_read or 0,
                "tokens_proxy_cache_write": cache_write or 0,
                "tokens_proxy_output": out_tok,
                "tokens_proxy_reasoning": reasoning,
            })
        return out

    # OpenAI-compatible chat completions shape.
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt = _num(usage.get("prompt_tokens"))
        details = usage.get("prompt_tokens_details") or {}
        cache_read = _num(usage.get("prompt_cache_hit_tokens"), None)
        if cache_read is None and isinstance(details, dict):
            cache_read = _num(details.get("cached_tokens"), 0)
        uncached = _num(usage.get("prompt_cache_miss_tokens"), None)
        if uncached is None and prompt is not None:
            uncached = max(0, prompt - (cache_read or 0))
        completion = _num(usage.get("completion_tokens"))
        out_details = usage.get("completion_tokens_details") or {}
        reasoning = _num(usage.get("reasoning_tokens"), None)
        if reasoning is None and isinstance(out_details, dict):
            reasoning = _num(out_details.get("reasoning_tokens"))
        if uncached is not None and completion is not None:
            out.update({
                "tokens_proxy_input_uncached": uncached,
                "tokens_proxy_cache_read": cache_read or 0,
                "tokens_proxy_cache_write": _num(usage.get("prompt_cache_write_tokens"), 0),
                "tokens_proxy_output": completion,
                "tokens_proxy_reasoning": reasoning,
            })
        return out

    return out


def _add_proxy_totals(total, split):
    for key in _empty_proxy_usage():
        val = split.get(key)
        if isinstance(val, int):
            total[key] = (total.get(key) or 0) + val


def read_proxy_ledger(ledger_dir, token, wait_s=0.0, stable_s=0.1):
    if not ledger_dir or not token:
        return []
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", token)
    path = os.path.join(str(ledger_dir), safe + ".jsonl")
    deadline = time.monotonic() + max(wait_s, 0.0)
    last_size = None
    stable_since = None
    while wait_s and time.monotonic() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(0.02)
            continue
        if size > 0 and size == last_size:
            if stable_since is None:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= stable_s:
                break
        else:
            last_size = size
            stable_since = None
        time.sleep(0.02)
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def apply_proxy_ledger(row, ledger_rows):
    """Populate proxy-measured token fields from scrubbed ledger rows.

    Truncated captures are surfaced on the row and must not claim
    ``token_basis_proxy=proxy_measured`` — partial ledgers are not a full meter.
    """
    records = [r for r in (ledger_rows or []) if isinstance(r, dict)]
    truncated = any(r.get("capture_truncated") for r in records)
    if truncated:
        row["proxy_capture_truncated"] = True
    calls = [r for r in records if isinstance(r.get("usage"), dict)]
    # Zero is meaningful evidence for admission checks; do not collapse it to None.
    row["tokens_proxy_calls"] = len(calls)
    if not calls:
        return row
    totals = _empty_proxy_usage()
    samplings = []
    seen_sampling = set()
    for rec in calls:
        _add_proxy_totals(totals, proxy_split_from_usage(rec.get("usage")))
        sampling = rec.get("sampling_observed")
        if isinstance(sampling, dict) and sampling:
            key = json.dumps(sampling, sort_keys=True, separators=(",", ":"))
            if key not in seen_sampling:
                seen_sampling.add(key)
                samplings.append(sampling)
    row.update(totals)
    row["sampling_observed"] = samplings or None
    served = [r.get("served_model") for r in calls
             if isinstance(r.get("served_model"), str) and r.get("served_model")]
    if served:
        # De-duplicated, order-preserving: a router may serve >1 model per cell.
        row["served_model"] = list(dict.fromkeys(served))
    cost = sum(r["cost"] for r in calls
              if isinstance(r.get("cost"), (int, float)) and not isinstance(r.get("cost"), bool))
    if any("cost" in r for r in calls):
        row["cost"] = cost
    upstream = sum(r["upstream_cost"] for r in calls
                  if isinstance(r.get("upstream_cost"), (int, float)) and not isinstance(r.get("upstream_cost"), bool))
    if any("upstream_cost" in r for r in calls):
        row["upstream_cost"] = upstream
    if not truncated:
        row["token_basis_proxy"] = "proxy_measured"
    return row


def _populate_proxy_row(row, proxy_ctx, cell_token, wait_s=0.0):
    if not proxy_ctx or not cell_token:
        return row
    return apply_proxy_ledger(
        row, read_proxy_ledger(proxy_ctx.get("ledger_dir"), cell_token, wait_s=wait_s))


def run_cell(harness, task, model, trial, timeout_s, tasks_dir, adapters_dir,
             checker_timeout_s, exec_mode="local",
             docker_image=None, docker_fallback=False, harness_version=None,
             container_versions_reader=read_container_cli_versions,
             transcripts_dir=None, results_stem="", proxy_ctx=None,
             candidate=None, version_drift=False, workspace_observer=None):
    """Execute one (task, harness, trial) cell and return its results row.

    Materializes the task workspace into a temp dir (snapshot ``workspace/``
    copytree, or git ``workspace.toml`` archive), invokes the adapter (or the
    built-in null adapter), runs the checker, and cleans up. Adapter and
    checker failures are recorded in the row rather than raised. Staging
    failures (including setup scripts) are recorded as ``failure_class=infra``.
    A checker that exceeds ``checker_timeout_s`` records
    ``checker_exit="timeout"``, ``success=false``.

    Scoring: exit 0 => success=true, score=1.0 (a SCORE line can't lower a pass).
    Nonzero exit => success=false, score = the checker's SCORE line if any, else
    0.0. Timeout => score 0.0. Local mode stamps ``harness_version`` from the
    host adapter probe; docker mode stamps it from the in-container versions
    file and records ``harness_version_source`` accordingly.

    When ``transcripts_dir`` is set, the cell's full agent transcript
    (adapter ``full_output`` if present, else ``output_tail``) is persisted
    LOCAL-ONLY to ``<transcripts_dir>/<results_stem>/<run_id>.txt``. See
    ``write_transcript`` for the local-only handling rule.
    """
    run_id = make_run_id(
        harness, task, model, trial,
        candidate.identity_digest if candidate is not None else None,
    )
    cell_token = None
    proxy_harness = (candidate.base_adapter or candidate.proxy_adapter) if candidate is not None else harness
    # Manifests must explicitly route traffic; only config variants inherit the
    # base adapter's proven proxy support.
    if candidate is not None and candidate.kind == "manifest":
        from .candidates import candidate_proxy_capable
        proxy_capable = candidate_proxy_capable(candidate)
    else:
        proxy_capable = proxy_supported_for_cell(proxy_harness, model)
    active_proxy_ctx = proxy_ctx if proxy_capable else None
    if active_proxy_ctx:
        from . import proxy as counting_proxy  # lazy: stdlib proxy only needed for --proxy
        cell_token = counting_proxy.new_cell_token()
        _write_proxy_cell_metadata(active_proxy_ctx, cell_token, proxy_harness, model)
    # Absolute so the checker (run with cwd=temp workdir) and TASK_DIR resolve
    # correctly regardless of the caller's cwd or a relative --tasks-dir.
    task_dir = os.path.abspath(os.path.join(tasks_dir, task))
    task_image, task_workdir = task_docker_spec(task_dir)
    if exec_mode == "docker" and task_image:
        docker_image = task_image
    if os.path.exists(os.path.join(task_dir, "DROPPED.md")):
        raise SystemExit(
            f"task {task!r} is dropped from the active set "
            f"(see {os.path.join(task_dir, 'DROPPED.md')}); refusing to schedule it")
    row = {
        "run_id": run_id,
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "harness": harness,
        "model": model,
        "task": task,
        "trial": trial,
        "success": False,
        "completed": False,
        "error": None,
        "wall_time_s": None,
        "t_env_setup_s": None,
        "t_agent_s": None,
        "t_checker_s": None,
        "tokens": None,
        "tokens_input_uncached": None,
        "tokens_cache_read": None,
        "tokens_cache_write": None,
        "tokens_output": None,
        "tokens_reasoning": None,
        "usage_raw": None,
        "token_basis": None,
        "tokens_proxy_input_uncached": None,
        "tokens_proxy_cache_read": None,
        "tokens_proxy_cache_write": None,
        "tokens_proxy_output": None,
        "tokens_proxy_reasoning": None,
        "tokens_proxy_calls": None,
        "sampling_observed": None,
        "token_basis_proxy": None,
        "tokens_fresh": None,
        "turns": None,
        "cmd": None,
        "output_tail": "",
        "checker_exit": None,
        "exec_mode": None,
        "score": 0.0,
        "harness_version": None,
        "harness_version_source": None,
        "failure_class": None,
        "failure_reason": None,
        "workspace_changed": None,
        "checker_stdout": None,
        "checker_stderr": None,
        "checker_workspace_files": None,
        "image_digest": None,
        "candidate_provenance": candidate.provenance if candidate is not None else None,
        "version_drift": bool(version_drift),
        "timeout_s": timeout_s,
        "workspace_source": None,
    }

    # Namespaced tasks (e.g. terminal-bench/feal) contain "/"; keep the prefix
    # a single path component.
    # Docker mode bind-mounts this dir; on colima the default macOS /var/folders
    # temp path is NOT shared into the VM and mounts as an EMPTY dir, so create
    # it somewhere the VM can see (same policy as docker_exec instruction files).
    env_setup_start = time.monotonic()
    workdir_parent = None
    if exec_mode == "docker":
        workdir_parent = os.environ.get("OPENBENCH_DOCKER_TMPDIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".bench-tmp")
        os.makedirs(workdir_parent, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix=f"bench_{harness}_{task.replace('/', '_')}_", dir=workdir_parent)
    try:
        # Materialize a pristine workspace into the disposable temp dir. Never
        # touch the source under tasks/ (snapshot copy or git archive export).
        # Staging runs on the host for both --exec local and --exec docker; the
        # container bind-mounts the already-staged workdir at /work.
        try:
            if exec_mode == "docker" and task_image:
                staged_workspace = tempfile.mkdtemp(prefix="obench_task_overlay_", dir=workdir_parent)
                try:
                    row["workspace_source"] = materialize_workspace(task_dir, staged_workspace)
                    hydrate_image_workdir(task_image, task_workdir, workdir)
                    for root, dirs, files in os.walk(workdir):
                        for name in dirs + files:
                            if os.path.islink(os.path.join(root, name)):
                                raise RuntimeError("task image workdir contains a symlink")
                    shutil.copytree(staged_workspace, workdir, dirs_exist_ok=True)
                finally:
                    shutil.rmtree(staged_workspace, ignore_errors=True)
                with open(os.path.join(workdir, ".openbench-image-hydrated"), "w", encoding="utf-8") as fh:
                    fh.write(task_image + "\n")
            else:
                row["workspace_source"] = materialize_workspace(task_dir, workdir)
        except (WorkspaceError, RuntimeError) as exc:
            row["error"] = f"workspace materialization failed: {exc}"
            row["t_env_setup_s"] = round(time.monotonic() - env_setup_start, 3)
            row["exec_mode"] = exec_mode
            row["failure_class"] = "infra"
            row["failure_reason"] = "workspace_materialization"
            return _populate_proxy_row(row, active_proxy_ctx, cell_token)
        initial_workspace_files = capture_workspace_files(workdir)

        instruction = read_instruction(task_dir)
        row["t_env_setup_s"] = round(time.monotonic() - env_setup_start, 3)

        start = time.monotonic()
        try:
            result, exec_used = invoke_adapter(
                exec_mode, harness, instruction, workdir, model, timeout_s,
                adapters_dir, docker_image, docker_fallback,
                proxy_ctx=active_proxy_ctx, cell_token=cell_token,
                candidate=candidate, container_workdir=task_workdir,
            )
        except Exception as exc:  # noqa: BLE001 - never crash the loop on an adapter
            exec_used = getattr(exc, "bench_exec_used", exec_mode)
            row["error"] = traceback.format_exc(limit=4).strip()
            env_extra = getattr(exc, "bench_env_setup_s", None)
            if isinstance(env_extra, (int, float)) and env_extra >= 0:
                row["t_env_setup_s"] = round((row.get("t_env_setup_s") or 0.0) + env_extra, 3)
            agent_elapsed = getattr(exc, "bench_agent_wall_time_s", None)
            if not isinstance(agent_elapsed, (int, float)) or agent_elapsed < 0:
                agent_elapsed = time.monotonic() - start
            row["t_agent_s"] = round(agent_elapsed, 3)
            row["wall_time_s"] = round(time.monotonic() - start, 3)
            row["exec_mode"] = exec_used
            version_harness = proxy_harness or harness
            row["harness_version"], row["harness_version_source"] = harness_version_for_source(
                version_harness, exec_used, harness_version, docker_image, None,
                container_versions_reader,
            )
            row["failure_class"] = classify_failure(row, "", timeout_s)
            return _populate_proxy_row(row, active_proxy_ctx, cell_token, wait_s=2.0)
        row["wall_time_s"] = _adapter_wall_time_s(start, result, exec_used)
        row["t_agent_s"] = _agent_wall_time_s(start, result, exec_used)
        _add_host_env_setup_s(row, result, exec_used)
        row["exec_mode"] = exec_used
        if exec_used == "docker":
            row["image_digest"] = result.get("image_digest")
        version_harness = proxy_harness or harness
        row["harness_version"], row["harness_version_source"] = harness_version_for_source(
            version_harness, exec_used, harness_version, docker_image, row["image_digest"],
            container_versions_reader,
        )
        if exec_used == "docker" and result.get("candidate_version"):
            row["harness_version"] = result["candidate_version"]
            row["harness_version_source"] = "container"

        # Fold the adapter's self-reported fields into the row.
        row["completed"] = bool(result.get("completed", False))
        row["error"] = result.get("error")
        row["tokens"] = result.get("tokens")
        row["tokens_input_uncached"] = result.get("tokens_input_uncached")
        row["tokens_cache_read"] = result.get("tokens_cache_read")
        row["tokens_cache_write"] = result.get("tokens_cache_write")
        row["tokens_output"] = result.get("tokens_output")
        row["tokens_reasoning"] = result.get("tokens_reasoning")
        row["usage_raw"] = result.get("usage_raw")
        row["token_basis"] = ("unmetered" if candidate is not None
                              and getattr(candidate, "unmetered", False)
                              else result.get("token_basis"))
        row["tokens_fresh"] = result.get("tokens_fresh")
        if row["tokens_fresh"] is None:
            inp = row["tokens_input_uncached"]
            out = row["tokens_output"]
            if isinstance(inp, int) and isinstance(out, int):
                row["tokens_fresh"] = inp + out
        row["turns"] = result.get("turns")
        row["cmd"] = result.get("cmd")
        row["output_tail"] = result.get("output_tail") or ""
        full_output = result.get("full_output")
        classifier_output = full_output if full_output is not None else row["output_tail"]
        _populate_proxy_row(row, active_proxy_ctx, cell_token, wait_s=2.0)

        # Persist the full agent transcript LOCAL-ONLY (prefer the untruncated
        # full_output; fall back to the ~2000-char output_tail). Never let a
        # transcript-write failure break the benchmark loop.
        if transcripts_dir:
            body = full_output
            if body is None:
                body = row["output_tail"]
            try:
                write_transcript(
                    transcript_path(transcripts_dir, results_stem, run_id),
                    row, body,
                )
            except Exception:  # noqa: BLE001 - transcript IO must not fail a cell
                pass

        # The checker is the sole authority on task success (and score). Capture
        # the workspace just before it runs so unauditable rows can be replayed.
        try:
            current_workspace_files = capture_workspace_files(workdir)
            row["workspace_changed"] = current_workspace_files != initial_workspace_files
            row["checker_workspace_files"] = scrub_workspace_evidence_paths(
                current_workspace_files)
            checker_start = time.monotonic()
            try:
                checker_exit, raw_score, checker_stdout, checker_stderr = run_checker(
                    task_dir, workdir, checker_timeout_s)
            finally:
                row["t_checker_s"] = round(time.monotonic() - checker_start, 3)
        except Exception:  # noqa: BLE001
            row["checker_exit"] = None
            if row["error"] is None:
                row["error"] = traceback.format_exc(limit=4).strip()
            row["failure_class"] = classify_failure(row, classifier_output, timeout_s)
            return _populate_proxy_row(row, active_proxy_ctx, cell_token)
        row["checker_stdout"] = scrub_checker_output(checker_stdout)
        row["checker_stderr"] = scrub_checker_output(checker_stderr)
        row["checker_exit"] = checker_exit
        row["success"] = (checker_exit == 0)
        # exit 0 is a full pass (score 1.0) regardless of any SCORE line; a
        # nonzero exit takes the SCORE line for partial credit, else 0.0.
        row["score"] = 1.0 if checker_exit == 0 else (
            raw_score if raw_score is not None else 0.0)
        row["failure_class"] = classify_failure(row, classifier_output, timeout_s)
        row["failure_reason"] = classify_failure_reason(row, classifier_output)
        return _populate_proxy_row(row, active_proxy_ctx, cell_token)
    finally:
        if workspace_observer is not None:
            try:
                workspace_observer(workdir)
            except Exception:  # evidence collection must not alter the cell verdict
                pass
        shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Agent-harness comparison runner.")
    parser.add_argument("--task", required=True,
                        help="comma-separated task name(s)")
    parser.add_argument("--harness", default=None,
                        help="comma-separated stock harness or candidate names")
    parser.add_argument("--model", default=None,
                        help=f"canonical model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--trials", type=int, default=None,
                        help="trials per (task, harness) cell (default: 1)")
    parser.add_argument("--trial", type=int, default=None,
                        help="run only this trial number (for matrix wrappers; "
                             "default: run 1..--trials)")
    parser.add_argument("--timeout", type=int, default=2400,
                        help="per-cell adapter timeout in seconds (default: 2400)")
    parser.add_argument("--checker-timeout", type=int, default=120,
                        help="checker.sh timeout in seconds (default: 120); "
                             "on timeout the row records checker_exit='timeout'")
    parser.add_argument("--force", action="store_true",
                        help="re-run cells even if their run_id already exists")
    parser.add_argument("--results-path", default=None,
                        help="override the results.jsonl path "
                             "(default: <repo|cwd>/results/results.jsonl)")
    parser.add_argument("--adapters-dir", default=None,
                        help="override the adapters directory "
                             "(default: packaged obench/adapters)")
    parser.add_argument("--candidate", action="append", default=[], metavar="SPEC",
                        help="candidate TOML path or harness pack ref "
                             "org/name[@version][:manifest] (repeatable)")
    parser.add_argument("--tasks-dir", default=None,
                        help="override the tasks directory "
                             "(default: ./tasks or ./.openbench/tasks)")
    parser.add_argument("--transcripts-dir", default=None,
                        help="base dir for LOCAL-ONLY per-cell transcripts "
                             "(default: a 'transcripts/' sibling of the results "
                             "log). Transcripts are unscrubbed; review with "
                             "obench/scrub.py --check before sharing.")
    parser.add_argument("--exec", dest="exec_mode", default="local",
                        choices=("local", "docker"),
                        help="execution backend: 'local' (host, default) or "
                             "'docker' (one disposable container per cell)")
    parser.add_argument("--docker-image", default="openbench-harness:latest",
                        help="image for --exec docker "
                             "(default: openbench-harness:latest)")
    parser.set_defaults(docker_fallback=False)
    parser.add_argument("--docker-fallback", dest="docker_fallback",
                        action="store_true",
                        help="in --exec docker, allow falling back to local when "
                             "the daemon/image is unavailable (homogenizes the "
                             "whole run to local at preflight; aborts if a "
                             "mid-run per-cell fallback would mix exec lanes)")
    parser.add_argument("--no-docker-fallback", dest="docker_fallback",
                        action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--allow-version-drift", action="store_true",
                        help="run despite host/image CLI pin mismatch and mark every row")
    parser.add_argument("--max-consecutive-infra", type=int,
                        default=DEFAULT_MAX_CONSECUTIVE_INFRA, metavar="N",
                        help="abort after N consecutive near-zero-token infra/rate-limit "
                             "cells (default: 3; 0 disables)")
    parser.add_argument("--preflight-smoke", action="store_true",
                        help="run one smoke cell (make-it-run if present, else "
                             "first runnable task) into a sidecar before main cells")
    parser.add_argument("--allow-preflight-failure", action="store_true",
                        help="start main cells even if the requested preflight smoke fails")
    parser.add_argument("--proxy", action="store_true",
                        help="start one owned counting proxy and inject it into "
                             "supported harness/model cells (Cursor and Devin are unsupported)")
    args = parser.parse_args(argv)
    if args.max_consecutive_infra < 0:
        parser.error("--max-consecutive-infra must be >= 0")

    cfg = load_config()
    if args.results_path is None:
        args.results_path = cfg.results_path or default_results_path()
    if args.adapters_dir is None:
        args.adapters_dir = default_adapters_dir()
    if args.model is None:
        args.model = cfg.model or DEFAULT_MODEL
    if args.trials is None:
        args.trials = cfg.trials if cfg.trials is not None else 1
    if args.harness is None:
        args.harness = ",".join(cfg.harnesses) if cfg.harnesses else ""
    try:
        # Explicit --tasks-dir wins; else config tasks_dir; else discovery.
        tasks_override = args.tasks_dir or cfg.tasks_dir
        args.tasks_dir = resolve_tasks_dir(tasks_override)
    except TasksDirError as exc:
        parser.error(str(exc))

    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    harnesses = [h.strip() for h in args.harness.split(",") if h.strip()]
    from .candidates import load_candidates, candidate_proxy_capable
    try:
        candidates = load_candidates(args.candidate, args.adapters_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    collisions = sorted(set(harnesses) & set(candidates))
    if collisions:
        parser.error("candidate name collides with --harness: " + ",".join(collisions))
    for name in candidates:
        if name not in harnesses:
            harnesses.append(name)
    if not harnesses:
        parser.error("at least one --harness or --candidate is required")

    # Docker fallback can turn a nominal Docker invocation into a mixed run, so
    # it must satisfy the same host gate. Every Docker invocation also checks
    # its build-time pin labels with one inspect before any cell can execute.
    host_drift = []
    image_drift = []
    try:
        if args.exec_mode == "local" or args.docker_fallback:
            host_drift = host_version_drift(harnesses, candidates)
        if args.exec_mode == "docker":
            declared_images = []
            has_undeclared = False
            for task in tasks:
                declared, _ = task_docker_spec(
                    os.path.abspath(os.path.join(args.tasks_dir, task)))
                if declared:
                    declared_images.append(declared)
                else:
                    has_undeclared = True
            preflight_images = sorted(set(declared_images))
            if declared_images and args.docker_fallback:
                print(
                    "Version preflight failed: --docker-fallback is incompatible "
                    "with tasks that declare docker_image",
                    file=sys.stderr,
                )
                return 2
            if has_undeclared or not preflight_images:
                preflight_images.append(args.docker_image)
            image_available = True
            for image in preflight_images:
                drift, available = image_version_drift(image, harnesses, candidates)
                image_drift.extend(drift)
                if not available:
                    image_available = False
                    hint = f"docker build -t {image} obench/docker"
                    if not args.docker_fallback:
                        print(
                            f"Version preflight failed: cannot inspect Docker image "
                            f"{image!r}. Build it with: {hint}",
                            file=sys.stderr,
                        )
                        return 2
                    print(
                        f"WARN: cannot inspect Docker image {image!r}; "
                        f"falling back to the validated host lane. Build it with: {hint}",
                        file=sys.stderr,
                    )
                    # Do not let a later per-cell Docker retry bypass an inconclusive
                    # image gate. The host lane was validated above because fallback
                    # is enabled, so force this whole invocation onto that lane.
                    args.exec_mode = "local"
    except (OSError, VersionDriftError) as exc:
        print(f"Version preflight failed: {exc}", file=sys.stderr)
        return 2

    drift = host_drift or image_drift
    if (host_drift or image_drift) and not args.allow_version_drift:
        print(version_drift_refusal(
            host_drift, image_drift, args.docker_image), file=sys.stderr)
        return 2
    if host_drift or image_drift:
        print("WARN: version drift allowed; every emitted row will be marked", file=sys.stderr)

    transcripts_dir = args.transcripts_dir or default_transcripts_dir(args.results_path)
    results_stem = os.path.splitext(os.path.basename(args.results_path))[0]

    try:
        existing = set() if args.force else load_existing_run_ids(args.results_path)
    except ResultsLogError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    # Probe each harness's version at most once per invocation (a version()
    # probe may spawn a subprocess), then stamp the cached value into every row.
    versions = {h: probe_version(h, args.adapters_dir, candidates.get(h)) for h in harnesses}

    if args.trial is not None and args.trial < 1:
        parser.error("--trial must be >= 1")
    trial_numbers = [args.trial] if args.trial is not None else range(1, args.trials + 1)

    proxy_ctx = None
    proxy_server = None
    if args.proxy:
        from . import proxy as counting_proxy
        ledger_parent = os.environ.get("OPENBENCH_PROXY_LEDGER_DIR") or tempfile.mkdtemp(
            prefix="openbench_proxy_", dir=os.environ.get("OPENBENCH_DOCKER_TMPDIR") or None)
        listen_host = "0.0.0.0" if args.exec_mode == "docker" else "127.0.0.1"
        subbridge_origin = "http://127.0.0.1:8317"
        if "grokbuild" in harnesses and args.model == "gpt-5.6":
            # Meter before CLIProxyAPI on a dedicated route, so mixed-harness
            # runs cannot redirect unrelated OpenAI-compatible cells.
            bridge_base = os.environ.get("CLIPROXYAPI_BASE_URL") or "http://127.0.0.1:8317/v1"
            from urllib.parse import urlsplit, urlunsplit
            parsed_bridge = urlsplit(bridge_base)
            if parsed_bridge.scheme not in {"http", "https"} or not parsed_bridge.netloc:
                parser.error("CLIPROXYAPI_BASE_URL must be an absolute HTTP(S) URL")
            if parsed_bridge.username is not None or parsed_bridge.password is not None:
                parser.error("CLIPROXYAPI_BASE_URL must not contain URL-embedded credentials")
            if parsed_bridge.query or parsed_bridge.fragment:
                parser.error("CLIPROXYAPI_BASE_URL must not contain a query or fragment")
            subbridge_origin = urlunsplit((parsed_bridge.scheme, parsed_bridge.netloc, "", "", ""))
        proxy_server, _thread = counting_proxy.start_in_thread(
            listen_host, 0, ledger_parent, subbridge_upstream=subbridge_origin,
            require_registered_tokens=args.exec_mode == "docker")
        port = proxy_server.server_address[1]
        proxy_ctx = {
            "ledger_dir": ledger_parent,
            "local_base_url": f"http://127.0.0.1:{port}",
            "docker_base_url": f"http://host.docker.internal:{port}",
        }
        proxy_names = {h: (candidates[h].proxy_adapter if h in candidates else h)
                       for h in harnesses}
        manifest_proxy = {
            h for h, candidate in candidates.items()
            if candidate_proxy_capable(candidate)
        }
        unsupported = sorted(
            h for h, base in proxy_names.items()
            if h not in manifest_proxy and base not in PROXY_HARNESSES
        )
        unsupported_cells = [
            h for h, base in proxy_names.items()
            if h not in manifest_proxy and base in PROXY_HARNESSES
            and not proxy_supported_for_cell(base, args.model)
        ]
        if unsupported:
            print("WARN --proxy does not wire these harnesses yet: " + ",".join(unsupported))
        if unsupported_cells:
            print("WARN --proxy does not wire these harness/model cells yet: "
                  + ",".join(f"{h}:{args.model}" for h in unsupported_cells))
        print(f"PROXY listening={proxy_ctx['local_base_url']} ledger_dir={ledger_parent}")

    ran = 0
    skipped = 0
    infra_streak = 0
    try:
        if args.preflight_smoke:
            try:
                smoke_task = select_preflight_task(args.tasks_dir)
            except TasksDirError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            smoke_harness = harnesses[0]
            smoke_candidate = candidates.get(smoke_harness)
            smoke_row = run_cell(
                smoke_harness, smoke_task, args.model, 0, args.timeout,
                args.tasks_dir, args.adapters_dir,
                args.checker_timeout,
                exec_mode=args.exec_mode, docker_image=args.docker_image,
                docker_fallback=args.docker_fallback,
                harness_version=versions.get(smoke_harness),
                transcripts_dir=transcripts_dir, results_stem=results_stem,
                proxy_ctx=proxy_ctx,
                candidate=smoke_candidate,
                version_drift=bool(drift),
            )
            smoke_path = preflight_results_path(args.results_path)
            append_row(smoke_path, smoke_row)
            print(
                f"PREFLIGHT {smoke_row['run_id']} failure_class="
                f"{smoke_row.get('failure_class')} tokens={smoke_row.get('tokens')} "
                f"results={smoke_path}"
            )
            if is_near_zero_infra(smoke_row):
                message = preflight_refusal_message(smoke_row)
                if not args.allow_preflight_failure:
                    print(message, file=sys.stderr)
                    return 2
                print("WARN: " + message, file=sys.stderr)

        for harness in harnesses:
            for task in tasks:
                for trial in trial_numbers:
                    candidate = candidates.get(harness)
                    run_id = make_run_id(
                        harness, task, args.model, trial,
                        candidate.identity_digest if candidate is not None else None,
                    )
                    if run_id in existing:
                        skipped += 1
                        print(f"SKIP {run_id}")
                        continue
                    row = run_cell(
                        harness, task, args.model, trial, args.timeout,
                        args.tasks_dir, args.adapters_dir, args.checker_timeout,
                        exec_mode=args.exec_mode, docker_image=args.docker_image,
                        docker_fallback=args.docker_fallback,
                        harness_version=versions.get(harness),
                        transcripts_dir=transcripts_dir, results_stem=results_stem,
                        proxy_ctx=proxy_ctx,
                        candidate=candidate,
                        version_drift=bool(drift),
                    )
                    # Fail-closed on mixed lanes: a mid-run docker→local fallback
                    # must not land in the same results file as docker cells.
                    # (Whole-run preflight homogenization sets exec_mode=local.)
                    if (args.exec_mode == "docker"
                            and args.docker_fallback
                            and row.get("exec_mode") == "local"):
                        print(
                            f"FATAL: docker→local fallback on {run_id}; refusing "
                            "mixed exec_mode lanes (re-run with a working image, "
                            "or omit --docker-fallback for fail-closed docker).",
                            file=sys.stderr,
                        )
                        return 2
                    append_row(args.results_path, row)
                    existing.add(run_id)
                    ran += 1
                    status = "ok" if row["success"] else "fail"
                    print(f"RUN  {run_id} success={row['success']} score={row['score']} "
                          f"completed={row['completed']} checker_exit={row['checker_exit']} "
                          f"exec={row['exec_mode']} [{status}]")
                    if is_near_zero_infra(row):
                        infra_streak += 1
                        if (args.max_consecutive_infra
                                and infra_streak >= args.max_consecutive_infra):
                            print(circuit_breaker_message(infra_streak, row), file=sys.stderr)
                            return 2
                    else:
                        infra_streak = 0
    finally:
        if proxy_server is not None:
            proxy_server.shutdown()
            proxy_server.server_close()

    print(f"\nDone. ran={ran} skipped={skipped} results={args.results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
