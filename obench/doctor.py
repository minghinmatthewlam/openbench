#!/usr/bin/env python3
"""Preflight doctor for the agent-harness benchmark.

Run this BEFORE a matrix run to catch missing CLIs / auth / model pins without
spending any tokens (no live model calls). Per requested harness it verifies:

  1. CLI     - the harness binary is installed (which + --version captured)
  2. AUTH    - the adapter-specific credential is present, mirroring exactly
               what each bench/adapters/<name>.py expects at run time
  3. MODEL   - the adapter module imports and its MODELS maps the requested
               canonical model name

Docker daemon/image availability is informational. A successfully inspected
image whose labels drift from the Dockerfile pins fails the preflight.

    python3 -m obench.doctor [--harness codex,pi,...] [--model gpt-5.5-medium]
    python3 -m obench.doctor --docker-env

Exit status is nonzero if any requested harness fails any of CLI/AUTH/MODEL, or
any ``--docker-env`` check fails.

Auth expectations are mirrored from the adapters (read them, don't invent):
  codex     ~/.codex/auth.json exists (adapter uses ~/.codex login as-is)
  codex_v1/codex_v2 same auth as codex; adapters compose temp CODEX_HOME
  pi        ~/.pi/agent/auth.json exists AND has an "openai-codex" or
            "anthropic" entry (adapter's isolated-HOME route reads this file)
  opencode  `opencode auth list` shows an OpenAI oauth credential (adapter
            strips OPENAI_API_KEY to force the subscription OAuth route)
  cursor    `cursor-agent status` exits 0 (existing Cursor login)
  claude    no ~/.claude mount; API-key routes require provider env keys
  grokbuild no ~/.grok mount; open routes need vendor keys, while gpt-5.6
            uses host-side CLIProxyAPI subscription OAuth
  devin     ~/.config/devin exists (existing devin login)

Python3 stdlib only.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import http.client
import subprocess
import sys
import tomllib
from urllib.parse import urlsplit

from .bump_clis import (image_pin_mismatches, parse_image_pin_labels,
                        pinned_versions, reported_version, resolve_pin_key)
from .paths import PACKAGE_DIR, default_adapters_dir, ensure_package_path_on_sys_path

HERE = PACKAGE_DIR
ADAPTERS_DIR = default_adapters_dir()
DEFAULT_MODEL = "gpt-5.5-medium"
DEFAULT_IMAGE = "openbench-harness:latest"

CHECKS = ("CLI", "VERSION", "AUTH", "MODEL")
# Candidate-only checks appear in Details (and in the matrix when present).
CANDIDATE_EXTRA_CHECKS = ("CONFIG", "ENV")

# Docker-env check columns
DOCKER_ENV_CHECKS = ("BUILDX", "CPUS", "MEMORY", "IMAGES", "AUTH")

# M4 open canonical model -> the env key its provider needs. When --model is one
# of these, the AUTH check becomes "is this key exported?" instead of the
# harness's own subscription-login check. Mirrors the adapters' OPEN_MODELS.
OPEN_MODEL_ENV = {
    "glm-5.2": "ZAI_API_KEY",
    "glm-4.7-flash": "ZAI_API_KEY",
    "deepseek-v4-flash": "DEEPSEEK_API_KEY",
    "kimi-k2.7-code": "MOONSHOT_API_KEY",
    "laguna-s-2.1": "OPENROUTER_API_KEY",
    "inkling": "OPENROUTER_API_KEY",
}
FRONTIER_MODEL_ENV = {
    "claude-opus-4-8": "ANTHROPIC_API_KEY",
}
KEYS_ENV = "~/.openbench/keys.env"

# Default env-requirements path
ENV_REQUIREMENTS_PATH = ".openbench/env-requirements.toml"
# Default packs dir for discovering per-task pinned images
PACKS_DIR = os.path.join(".openbench", "packs")


# --------------------------------------------------------------------------- #
# Probes: every side effect goes through this object so tests can mock it all
# and never touch the real CLIs, filesystem, or network.
# --------------------------------------------------------------------------- #
class Probes:
    """Real-world probes: subprocess, filesystem, adapter import."""

    def which(self, cli):
        """Absolute path to ``cli`` on PATH, or None."""
        from shutil import which
        return which(cli)

    def run(self, argv, timeout=15):
        """Run ``argv`` headlessly; return (exit_code|None, combined_output).

        exit_code is None if the command is missing or times out.
        """
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None, ""
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def exists(self, path):
        """True if ``path`` (file or dir) exists, expanding ``~``."""
        return os.path.exists(os.path.expanduser(path))

    def http_get(self, url, headers=None, timeout=2.0):
        """Return (status, body) for a small readiness GET, or (None, '')."""
        parsed = urlsplit(url)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        try:
            conn = cls(parsed.hostname, parsed.port, timeout=timeout)
            conn.request("GET", parsed.path or "/", headers=headers or {})
            response = conn.getresponse()
            body = response.read(1024 * 1024).decode("utf-8", "replace")
            conn.close()
            return response.status, body
        except (OSError, http.client.HTTPException, ValueError):
            return None, ""

    def getenv(self, name):
        """Return the environment variable ``name`` (or None)."""
        return os.environ.get(name)

    def read_json(self, path):
        """Parse JSON at ``path`` (expanding ``~``); None if missing/invalid."""
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def read_text(self, path):
        """Read text at ``path`` (expanding ``~``); None if missing/unreadable."""
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            return None

    def read_toml(self, path):
        """Parse TOML at ``path``; None if missing/invalid."""
        text = self.read_text(path)
        if text is None:
            return None
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            return None

    def import_adapter(self, name):
        """Import ``obench/adapters/<name>.py`` and return the module."""
        ensure_package_path_on_sys_path()
        path = os.path.join(ADAPTERS_DIR, f"{name}.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"adapter not found: {path}")
        spec = importlib.util.spec_from_file_location(f"doctor_adapter_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def listdir(self, path):
        """List directory entries; return empty list on error."""
        try:
            return os.listdir(path)
        except OSError:
            return []

    def isdir(self, path):
        """True if path is a directory."""
        return os.path.isdir(path)

    def isfile(self, path):
        """True if path is a file."""
        return os.path.isfile(path)


# --------------------------------------------------------------------------- #
# Auth checks - one per harness, mirroring the adapter's own expectation.
# Each returns (ok: bool, detail: str).
# --------------------------------------------------------------------------- #
def _auth_codex(p):
    path = "~/.codex/auth.json"
    if p.exists(path):
        return True, os.path.expanduser(path)
    return False, f"missing {os.path.expanduser(path)}"


def _auth_pi_provider(p, provider):
    path = "~/.pi/agent/auth.json"
    if not p.exists(path):
        return False, f"missing {os.path.expanduser(path)}"
    data = p.read_json(path)
    if not isinstance(data, dict):
        return False, f"unreadable JSON at {os.path.expanduser(path)}"
    if provider in data:
        return True, f"entry: {provider}"
    return False, f"no {provider} entry in ~/.pi/agent/auth.json"


def _auth_pi(p):
    return _auth_pi_provider(p, "openai-codex")


def _auth_opencode(p):
    return _auth_opencode_provider(p, "openai")


def _auth_opencode_provider(p, provider):
    code, out = p.run(["opencode", "auth", "list"])
    if code is None:
        return False, "`opencode auth list` did not run"
    if code != 0:
        return False, f"`opencode auth list` exit {code}"
    # Subscription credentials print as lines mentioning provider + oauth; API
    # key env lines have no "oauth" and should not pass subscription checks.
    for line in out.splitlines():
        low = line.lower()
        if provider.lower() in low and "oauth" in low:
            return True, f"{provider} oauth credential present"
    return False, f"no {provider} oauth credential in `opencode auth list`"


def _auth_cursor(p):
    code, out = p.run(["cursor-agent", "status"])
    if code == 0:
        first = out.strip().splitlines()[0] if out.strip() else "logged in"
        return True, first
    return False, f"`cursor-agent status` exit {code}"


def _auth_devin(p):
    path = "~/.config/devin"
    if p.exists(path):
        return True, os.path.expanduser(path)
    return False, f"missing {os.path.expanduser(path)}"


# Stock fallback when an adapter does not export DOCTOR = {"cli", "auth"}.
# Adapters may declare DOCTOR themselves; load_harnesses() overlays those.
# The adapter module name equals the harness name (cursor's binary is
# cursor-agent but its adapter is cursor.py).
_STOCK_HARNESSES = {
    "codex":    {"cli": "codex",        "auth": _auth_codex},
    "codex_v1": {"cli": "codex",        "auth": _auth_codex},
    "codex_v2": {"cli": "codex",        "auth": _auth_codex},
    "pi":       {"cli": "pi",           "auth": _auth_pi},
    "opencode": {"cli": "opencode",     "auth": _auth_opencode},
    "cursor":   {"cli": "cursor-agent", "auth": _auth_cursor},
    "claude":   {"cli": "claude",       "auth": lambda p: (True, "API-key routes checked per model")},
    "grokbuild": {"cli": "grok",         "auth": lambda p: (True, "BYOK routes checked per model")},
    "devin":    {"cli": "devin",        "auth": _auth_devin},
}

# Adapters omitted from the default preflight matrix (opt-in via --harness).
_OPT_IN_HARNESSES = frozenset({"claude", "grokbuild"})

_CANDIDATE_HINT = (
    "pass --candidate path/to/spec.toml or an installed harness pack ref "
    "(org/name@version) for third-party harnesses"
)


def _import_adapter_module(adapters_dir, name):
    """Import ``adapters/<name>.py``; raise on missing file or load failure."""
    ensure_package_path_on_sys_path()
    path = os.path.join(adapters_dir, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(f"doctor_scan_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_adapter_doctor(adapters_dir, name):
    """Return ``{"cli", "auth"}`` from an adapter's optional ``DOCTOR`` export.

    Returns None when the module is missing, fails to import, or does not
    declare a usable DOCTOR dict.
    """
    try:
        module = _import_adapter_module(adapters_dir, name)
    except Exception:  # noqa: BLE001 - keep stock fallback for broken adapters
        return None
    doc = getattr(module, "DOCTOR", None)
    if not isinstance(doc, dict):
        return None
    cli, auth = doc.get("cli"), doc.get("auth")
    if not isinstance(cli, str) or not cli or not callable(auth):
        return None
    return {"cli": cli, "auth": auth}


def load_harnesses(adapters_dir=None):
    """Build harness -> {cli, auth} from stock fallback + adapter DOCTOR exports."""
    adapters_dir = adapters_dir or ADAPTERS_DIR
    harnesses = {name: dict(spec) for name, spec in _STOCK_HARNESSES.items()}
    try:
        names = sorted(
            os.path.splitext(entry)[0]
            for entry in os.listdir(adapters_dir)
            if entry.endswith(".py") and not entry.startswith("_")
        )
    except OSError:
        return harnesses
    for name in names:
        discovered = discover_adapter_doctor(adapters_dir, name)
        if discovered is not None:
            harnesses[name] = discovered
    return harnesses


HARNESSES = load_harnesses()
# Default doctor preflight keeps the historical matrix harnesses for the default
# gpt-5.5-medium model; claude/grokbuild are opt-in because they support
# API-key/open-model routes, not the default ChatGPT subscription model.
ALL_HARNESSES = [h for h in HARNESSES if h not in _OPT_IN_HARNESSES]


def known_harness_names():
    """Stock harness names shown in unknown-harness errors (stable order)."""
    return [h for h in HARNESSES if h not in _OPT_IN_HARNESSES] + sorted(
        h for h in HARNESSES if h in _OPT_IN_HARNESSES
    )


# --------------------------------------------------------------------------- #
# Individual check functions -> (ok, detail)
# --------------------------------------------------------------------------- #
def check_cli(p, cli):
    path = p.which(cli)
    if not path:
        return False, f"{cli} not found on PATH"
    _, out = p.run([cli, "--version"])
    ver = out.strip().splitlines()[0] if out.strip() else ""
    return True, f"{path} ({ver})" if ver else path


def check_version(p, harness, cli, pins):
    """Compare a host CLI's reported version with its Dockerfile pin."""
    base = "codex" if harness.startswith("codex_") else harness
    try:
        key = resolve_pin_key(base)
    except ValueError:
        return None, "no Dockerfile pin (n/a)"
    expected = pins.get(key)
    if expected is None:
        return False, f"host=unavailable pin=missing ({key}) [drift]"
    code, out = p.run([cli, "--version"])
    actual = reported_version(out) if code == 0 else None
    ok = actual == expected
    label = "ok" if ok else "drift"
    return ok, f"host={actual or 'unavailable'} pin={expected} [{label}]"


def check_model(p, harness, model):
    try:
        mod = p.import_adapter(harness)
    except Exception as exc:  # noqa: BLE001 - report any import failure as FAIL
        return False, f"adapter import failed: {exc}"
    models = getattr(mod, "MODELS", None)
    if not isinstance(models, dict):
        return False, "adapter exposes no MODELS dict"
    if model in models:
        return True, f"{model} -> {models[model]}"
    open_models = getattr(mod, "OPEN_MODELS", None)
    if isinstance(open_models, dict) and model in open_models:
        origin = "open via registry" if getattr(mod, "OPEN_MODELS_SOURCE", None) else "open"
        return True, f"{model} -> {open_models[model]['model_id']} ({origin})"
    known = list(models) + (list(open_models) if isinstance(open_models, dict) else [])
    return False, f"{model} not in MODELS/OPEN_MODELS {known}"


def open_models_registry():
    """Path of the open-model registry in effect, or None.

    Returns the string ``"<error>: ..."`` rather than raising so preflight can
    report a broken registry instead of dying on it.
    """
    path = os.path.join(ADAPTERS_DIR, "_open_models.py")
    try:
        spec = importlib.util.spec_from_file_location("doctor_open_models", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.find_registry()
    except Exception as exc:  # noqa: BLE001 - surface any registry problem
        return f"<error>: {exc}"


def _keys_env_has(p, env_key):
    text = p.read_text(KEYS_ENV)
    if text is None:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        try:
            parts = shlex.split(line, comments=True, posix=True)
            first = parts[1] if parts and parts[0] == "export" and len(parts) > 1 else parts[0]
            key, val = first.split("=", 1)
        except (ValueError, IndexError):
            key, val = line.split("=", 1)[0].strip(), line.split("=", 1)[1].strip()
        if key == env_key and val.strip():
            return True
    return False


def check_open_key(p, env_key, *, keys_env_ok=False):
    """AUTH check for API-key routes: env key exported, or keys.env if allowed."""
    if p.getenv(env_key):
        return True, f"{env_key} present"
    if keys_env_ok and _keys_env_has(p, env_key):
        return True, f"{env_key} present in {os.path.expanduser(KEYS_ENV)}"
    if keys_env_ok:
        return False, f"SETUP-NEEDED: export {env_key} or add it to {os.path.expanduser(KEYS_ENV)}"
    return False, f"SETUP-NEEDED: export {env_key}"


def check_subbridge(p):
    """Verify CLIProxyAPI is installed and its configured ingress is reachable."""
    configured_base = p.getenv("CLIPROXYAPI_BASE_URL")
    if not configured_base and not p.which("cliproxyapi"):
        return False, "SETUP-NEEDED: brew install cliproxyapi"
    base = configured_base or "http://127.0.0.1:8317/v1"
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False, "SETUP-NEEDED: CLIPROXYAPI_BASE_URL must be an absolute HTTP(S) URL"
    local_hosts = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
    if parsed.scheme == "http" and parsed.hostname not in local_hosts:
        return False, "SETUP-NEEDED: remote CLIPROXYAPI_BASE_URL must use HTTPS"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False, "SETUP-NEEDED: CLIPROXYAPI_BASE_URL has an invalid port"
    models_url = base.rstrip("/") + "/models"
    ingress_key = p.getenv("CLIPROXYAPI_API_KEY") or "openbench-local-ingress"
    status, body = p.http_get(models_url, {"Authorization": f"Bearer {ingress_key}"})
    try:
        payload = json.loads(body)
        model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    except (ValueError, AttributeError):
        model_ids = set()
    if status != 200 or "gpt-5.6" not in model_ids:
        return False, f"SETUP-NEEDED: CLIProxyAPI gpt-5.6 subscription route unavailable at {parsed.hostname}:{port}"
    return True, f"CLIProxyAPI gpt-5.6 route ready at {parsed.hostname}:{port}"


def _auth_cursor_container(p):
    path = "~/.openbench/cursor-container-auth/.config/cursor/auth.json"
    if p.getenv("CURSOR_API_KEY"):
        return True, "CURSOR_API_KEY present"
    if p.exists(path):
        return True, os.path.expanduser(path)
    return False, ("SETUP-NEEDED: run bench/cursor_container_login.sh "
                   f"or export CURSOR_API_KEY (missing {os.path.expanduser(path)})")


def _auth_frontier(p, harness, model):
    env_key = FRONTIER_MODEL_ENV[model]
    if harness == "pi":
        return _auth_pi_provider(p, "anthropic")
    if harness == "opencode":
        return _auth_opencode_provider(p, "anthropic")
    if harness == "cursor":
        if p.getenv("BENCH_IN_CONTAINER"):
            return _auth_cursor_container(p)
        return _auth_cursor(p)
    if harness in {"codex", "codex_v1", "codex_v2"}:
        return check_open_key(p, env_key, keys_env_ok=True)
    if harness == "claude":
        return check_open_key(p, env_key)
    return HARNESSES[harness]["auth"](p)


def check_docker(p):
    """Informational Docker daemon probe -> (ok|None, detail)."""
    if not p.which("docker"):
        return None, "docker not on PATH (informational)"
    code, out = p.run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code == 0 and out.strip():
        return True, f"daemon up (server {out.strip().splitlines()[0]})"
    return False, "docker installed but daemon not responding"


def check_image_versions(p, harnesses, pins, image=DEFAULT_IMAGE):
    """Compare requested harness pins with one Docker image-label inspect."""
    code, out = p.run([
        "docker", "inspect", "--format", "{{json .Config.Labels}}", image,
    ])
    if code != 0:
        return None, (f"{image} unavailable; build with: "
                      f"docker build -t {image} obench/docker")
    labels = parse_image_pin_labels(out)
    keys = []
    for harness in harnesses:
        base = "codex" if harness.startswith("codex_") else harness
        try:
            key = resolve_pin_key(base)
        except ValueError:
            continue
        if key not in keys:
            keys.append(key)
    mismatches = image_pin_mismatches(pins, labels, keys)
    if mismatches:
        detail = "; ".join(
            f"{item['harness']}: image={item['actual']} pin={item['expected']}"
            for item in mismatches)
        return False, "drift: " + detail
    return True, f"{image} matches Dockerfile pins"


# --------------------------------------------------------------------------- #
# Docker-environment checks (--docker-env mode)
# --------------------------------------------------------------------------- #
_ENV_REQUIREMENTS_DEFAULT = {"cpus": 4, "memory_gib": 12}
_AUTH_LANES = (
    ("codex (subscription)", lambda p: _auth_codex(p)),
    ("pi (subscription)", lambda p: _auth_pi_provider(p, "openai-codex")),
    ("opencode (subscription)", lambda p: _auth_opencode_provider(p, "openai")),
    ("cursor (subscription)", lambda p: _auth_cursor(p)),
    ("devin (subscription)", lambda p: _auth_devin(p)),
    ("ANTHROPIC_API_KEY", lambda p: check_open_key(p, "ANTHROPIC_API_KEY")),
    ("OPENAI_API_KEY", lambda p: check_open_key(p, "OPENAI_API_KEY")),
    ("DEEPSEEK_API_KEY", lambda p: check_open_key(p, "DEEPSEEK_API_KEY")),
)


def load_env_requirements(p, path=ENV_REQUIREMENTS_PATH):
    """Load resource requirements from TOML; return dict with defaults."""
    data = p.read_toml(path)
    if data is None:
        return dict(_ENV_REQUIREMENTS_DEFAULT)
    return {
        "cpus": int(data.get("cpus", _ENV_REQUIREMENTS_DEFAULT["cpus"])),
        "memory_gib": int(data.get("memory_gib", _ENV_REQUIREMENTS_DEFAULT["memory_gib"])),
    }


def check_buildx(p):
    """Check BuildKit/buildx plugin is installed and functional."""
    if not p.which("docker"):
        return False, "docker not on PATH"
    code, out = p.run(["docker", "buildx", "version"])
    if code == 0 and out.strip():
        ver = out.strip().splitlines()[0]
        return True, f"buildx plugin present ({ver})"
    return False, "buildx plugin not found — install: docker buildx install or docker/buildx-bin"


def check_docker_resources(p, requirements):
    """Check docker daemon CPU and memory meet requirements from env-requirements.toml."""
    if not p.which("docker"):
        return False, "docker not on PATH"

    # Check CPUs
    code_cpu, out_cpu = p.run(["docker", "info", "--format", "{{.NCPU}}"])
    if code_cpu != 0 or not out_cpu.strip():
        return False, "docker daemon not responding"
    try:
        actual_cpus = float(out_cpu.strip())
    except (ValueError, TypeError):
        return False, f"unparseable CPU count: {out_cpu.strip()!r}"
    req_cpus = requirements.get("cpus", 4)
    cpu_ok = actual_cpus >= req_cpus

    # Check memory (in GiB)
    code_mem, out_mem = p.run(["docker", "info", "--format", "{{.MemTotal}}"])
    if code_mem != 0 or not out_mem.strip():
        return False, "docker daemon not responding"
    try:
        mem_bytes = int(out_mem.strip())
        actual_mem_gib = mem_bytes / (1024 ** 3)
    except (ValueError, TypeError):
        return False, f"unparseable memory: {out_mem.strip()!r}"
    req_mem = requirements.get("memory_gib", 12)
    mem_ok = actual_mem_gib >= req_mem

    if cpu_ok and mem_ok:
        return True, (f"CPUs={actual_cpus:.1f} (req >= {req_cpus})  "
                      f"Memory={actual_mem_gib:.1f} GiB (req >= {req_mem})")
    parts = []
    if not cpu_ok:
        parts.append(f"CPUs={actual_cpus:.1f} < {req_cpus} (required)")
    if not mem_ok:
        parts.append(f"Memory={actual_mem_gib:.1f} GiB < {req_mem} GiB (required)")
    return False, "; ".join(parts)


def discover_task_images(p, packs_dir=PACKS_DIR):
    """Discover per-task pinned Docker images from installed task packs.

    Returns a dict mapping image_ref (with digest) -> [(pack, task_name)].
    Images are collected from task.toml files under installed packs.
    """
    images = {}
    if not p.isdir(packs_dir):
        return images

    for org_name in p.listdir(packs_dir):
        org_path = os.path.join(packs_dir, org_name)
        if not p.isdir(org_path):
            continue
        for pack_name in p.listdir(org_path):
            pack_path = os.path.join(org_path, pack_name)
            if not p.isdir(pack_path):
                continue
            for version in p.listdir(pack_path):
                version_path = os.path.join(pack_path, version)
                pack_toml = os.path.join(version_path, "pack.toml")
                pack_data = p.read_toml(pack_toml)
                pack_kind = (pack_data or {}).get("kind", "")
                if pack_kind != "tasks":
                    continue
                # Scan each task subdirectory for task.toml with docker_image.
                for task_name in p.listdir(version_path):
                    task_path = os.path.join(version_path, task_name)
                    task_toml_path = os.path.join(task_path, "task.toml")
                    if not (p.isdir(task_path) and p.isfile(task_toml_path)):
                        continue
                    task_data = p.read_toml(task_toml_path)
                    if task_data is None:
                        continue
                    image_ref = (task_data.get("docker_image") or "").strip()
                    if not image_ref:
                        continue
                    image_key = image_ref.split("@")[0] if "@" in image_ref else image_ref
                    source = f"{org_name}/{pack_name}:{version}/{task_name}"
                    images.setdefault(image_key, []).append({
                        "ref": image_ref,
                        "source": source,
                    })
    return images


def check_task_images(p, images):
    """Check per-task pinned images: present locally AND functionally probed.

    Each image is first inspected for local presence (docker image inspect).
    If present, a short exec probe validates it can actually run:
      docker run --rm <image> python3 -c "print('ok')"
    This catches corrupt or empty images that pass inspect but fail at runtime
    (a failure mode observed in image save/load corruption across runtimes).
    """
    if not p.which("docker"):
        return False, "docker not on PATH"

    if not images:
        return None, "no per-task pinned images found in installed packs (n/a)"

    inspected = {}
    errors = []

    for image_key, sources in images.items():
        ref = sources[0]["ref"]
        # Check image exists via inspect (fast).
        code_inspect, _ = p.run([
            "docker", "image", "inspect", ref,
        ])
        if code_inspect != 0:
            errors.append(f"{image_key}: not found locally ({ref})")
            inspected[image_key] = False
            continue

        # Functional probe: run the image and exec python3 -c print('ok').
        code_probe, out_probe = p.run([
            "docker", "run", "--rm", ref, "python3", "-c", "print('ok')",
        ], timeout=30)
        if code_probe != 0:
            errors.append(f"{image_key}: functional probe FAILED (exit {code_probe})")
            inspected[image_key] = False
            continue
        if "ok" not in (out_probe or "").strip():
            errors.append(f"{image_key}: functional probe produced unexpected output")
            inspected[image_key] = False
            continue
        inspected[image_key] = True

    if not inspected:
        return None, "no per-task pinned images found (n/a)"

    all_ok = all(inspected.values())
    n_ok = sum(1 for v in inspected.values() if v)
    n_total = len(inspected)
    if all_ok:
        return True, f"{n_ok}/{n_total} images present and functional"
    detail = f"{n_ok}/{n_total} images OK; " + "; ".join(errors[:5])
    if len(errors) > 5:
        detail += f" (+{len(errors)-5} more)"
    return False, detail


def check_auth_lanes(p, lanes=_AUTH_LANES):
    """Check auth freshness per configured lane.

    Each lane is (label: str, check_fn: callable). Returns (ok_all, details)
    where details is a list of (label, ok, detail).
    """
    results = []
    for label, check_fn in lanes:
        try:
            ok, detail = check_fn(p)
        except Exception as exc:  # noqa: BLE001 - never crash on a single lane
            ok, detail = False, f"probe errored: {exc}"
        results.append((label, ok, detail))
    return results


def evaluate_docker_env(p, requirements=None, task_images=None):
    """Run all --docker-env checks; return (rows, ok)."""
    if requirements is None:
        requirements = load_env_requirements(p)
    if task_images is None:
        task_images = discover_task_images(p)

    rows = []
    all_ok = True

    # BUILDX
    buildx_ok, buildx_detail = check_buildx(p)
    rows.append({"check": "BUILDX", "ok": buildx_ok, "detail": buildx_detail})
    if buildx_ok is False:
        all_ok = False

    # CPUS/MEMORY
    resource_ok, resource_detail = check_docker_resources(p, requirements)
    rows.append({"check": "CPUS", "ok": resource_ok, "detail": resource_detail})
    if resource_ok is False:
        all_ok = False

    # IMAGES (per-task pinned images)
    images_ok, images_detail = check_task_images(p, task_images)
    rows.append({"check": "IMAGES", "ok": images_ok, "detail": images_detail})
    if images_ok is False:
        all_ok = False

    # AUTH lanes
    lane_results = check_auth_lanes(p)
    n_auth_ok = sum(1 for _, ok, _ in lane_results if ok)
    n_auth_fail = sum(1 for _, ok, _ in lane_results if not ok)
    n_auth_info = sum(1 for _, ok, _ in lane_results if ok is None)
    if n_auth_fail == 0:
        auth_ok = True
        auth_detail = f"{n_auth_ok}/{n_auth_ok + n_auth_fail + n_auth_info} lanes fresh"
    else:
        auth_ok = False
        failed = [label for label, ok, _ in lane_results if ok is False]
        auth_detail = f"{n_auth_ok}/{n_auth_ok + n_auth_fail + n_auth_info} lanes fresh; stale: {', '.join(failed)}"
    rows.append({"check": "AUTH", "ok": auth_ok, "detail": auth_detail})
    if auth_ok is False:
        all_ok = False

    return rows, all_ok, lane_results


def format_docker_env_report(rows, lane_results, requirements):
    """Render the PASS/FAIL table for --docker-env mode."""
    lines = []
    lines.append("Docker Environment Preflight")
    lines.append(f"Requirements: CPUs >= {requirements['cpus']}, "
                 f"Memory >= {requirements['memory_gib']} GiB")
    lines.append("")

    # Status table.
    headers = ["check", "status", "detail"]
    table = []
    for row in rows:
        status = "OK" if row["ok"] is True else ("FAIL" if row["ok"] is False else "INFO")
        table.append([row["check"], status, row["detail"]])

    widths = [len(h) for h in headers]
    for cells in table:
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(str(c)))

    def fmt(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines.append(fmt(headers))
    lines.append(fmt(["-" * w for w in widths]))
    for cells in table:
        lines.append(fmt(cells))

    # Auth lane details.
    lines.append("")
    lines.append("Auth lanes:")
    for label, ok, detail in lane_results:
        status = "OK" if ok is True else ("FAIL" if ok is False else "INFO")
        lines.append(f"  [{status:>4}] {label:<30} {detail}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Candidate preflight (manifest / config-variant)
# --------------------------------------------------------------------------- #
def check_manifest_version(p, candidate):
    """VERSION for manifests: declared version_command must return non-empty."""
    if not candidate.version_command:
        return False, "manifest declares no version_command"
    code, out = p.run(list(candidate.version_command))
    if code is None:
        return False, "version_command did not run"
    if code != 0:
        return False, f"version_command exit {code}"
    text = (out or "").strip()
    if not text:
        return False, "version_command returned empty"
    return True, text.splitlines()[0]


def check_manifest_auth(p, candidate):
    """AUTH for manifests: every declared auth_files source must exist."""
    if not candidate.auth_files:
        return True, "no auth_files declared"
    missing = []
    for auth in candidate.auth_files:
        source = auth.get("source", "")
        if not p.exists(source):
            missing.append(source)
    if missing:
        return False, "missing " + ", ".join(missing)
    return True, f"{len(candidate.auth_files)} auth file(s) present"


def check_manifest_pass_env(p, candidate):
    """ENV for manifests: warn (INFO) when declared pass_env names are unset."""
    if not candidate.pass_env:
        return None, "no pass_env declared"
    missing = [name for name in candidate.pass_env if not p.getenv(name)]
    if missing:
        return None, f"WARN unset: {', '.join(missing)}"
    return True, f"all {len(candidate.pass_env)} pass_env set"


def check_manifest_model(candidate, model):
    """MODEL for manifests: resolve via [models], or accept any when empty."""
    if not candidate.models:
        return True, f"{model} accepted (no [models] pin map)"
    if model in candidate.models:
        return True, f"{model} -> {candidate.models[model]}"
    return False, f"{model} not in [models] {list(candidate.models)}"


def check_config_variant_files(p, candidate):
    """CONFIG for config-variants: config_dir and each config_files source exist."""
    if not p.exists(candidate.config_dir):
        return False, f"missing config_dir {candidate.config_dir}"
    missing = []
    for entry in candidate.config_files:
        source = entry if isinstance(entry, str) else entry.get("source", "")
        path = os.path.join(candidate.config_dir, source)
        if not p.exists(path):
            missing.append(source)
    if missing:
        return False, "missing config_files: " + ", ".join(missing)
    return True, f"{candidate.config_dir} ({len(candidate.config_files)} file(s))"


def evaluate_candidate(candidate, model, probes, pins=None):
    """Preflight one loaded candidate; return ``(rows, ok)`` like ``evaluate``."""
    pins = pinned_versions() if pins is None else pins
    if candidate.kind == "config-variant":
        return _evaluate_config_variant(candidate, model, probes, pins)
    return _evaluate_manifest(candidate, model, probes)


def _evaluate_manifest(candidate, model, probes):
    name = candidate.name
    rows = []
    all_ok = True
    cli = candidate.command[0]
    cli_ok, cli_detail = check_cli(probes, cli)
    version_ok, version_detail = check_manifest_version(probes, candidate)
    auth_ok, auth_detail = check_manifest_auth(probes, candidate)
    env_ok, env_detail = check_manifest_pass_env(probes, candidate)
    model_ok, model_detail = check_manifest_model(candidate, model)
    for check, ok, detail in (
        ("CLI", cli_ok, cli_detail),
        ("VERSION", version_ok, version_detail),
        ("AUTH", auth_ok, auth_detail),
        ("ENV", env_ok, env_detail),
        ("MODEL", model_ok, model_detail),
    ):
        rows.append({"harness": name, "check": check, "ok": ok, "detail": detail})
        if ok is False:
            all_ok = False
    return rows, all_ok


def _evaluate_config_variant(candidate, model, probes, pins):
    """Stock checks for the base adapter, plus config_dir/config_files existence."""
    name = candidate.name
    base = candidate.base_adapter
    spec = HARNESSES.get(base)
    rows = []
    all_ok = True
    if spec is None:
        rows.append({
            "harness": name, "check": "KNOWN", "ok": False,
            "detail": (f"unknown base_adapter {base!r} "
                       f"(have {known_harness_names()}); {_CANDIDATE_HINT}"),
        })
        return rows, False

    cli_ok, cli_detail = check_cli(probes, spec["cli"])
    version_ok, version_detail = check_version(probes, base, spec["cli"], pins)
    if base == "grokbuild" and model == "gpt-5.6":
        auth_ok, auth_detail = check_subbridge(probes)
    elif model in FRONTIER_MODEL_ENV:
        auth_ok, auth_detail = _auth_frontier(probes, base, model)
    elif model in OPEN_MODEL_ENV:
        keys_env_ok = base in {"codex", "codex_v1", "codex_v2"}
        auth_ok, auth_detail = check_open_key(
            probes, OPEN_MODEL_ENV[model], keys_env_ok=keys_env_ok)
    else:
        auth_ok, auth_detail = spec["auth"](probes)
    model_ok, model_detail = check_model(probes, base, model)
    config_ok, config_detail = check_config_variant_files(probes, candidate)
    for check, ok, detail in (
        ("CLI", cli_ok, cli_detail),
        ("VERSION", version_ok, version_detail),
        ("AUTH", auth_ok, auth_detail),
        ("MODEL", model_ok, model_detail),
        ("CONFIG", config_ok, config_detail),
    ):
        rows.append({"harness": name, "check": check, "ok": ok, "detail": detail})
        if ok is False:
            all_ok = False
    return rows, all_ok


# --------------------------------------------------------------------------- #
# Evaluation + rendering
# --------------------------------------------------------------------------- #
def evaluate(harnesses, model, probes, pins=None, candidates=None):
    """Return ``(rows, ok)`` for the requested harnesses.

    ``rows`` is a list of dicts ``{harness, check, ok, detail}`` covering the
    CLI/AUTH/MODEL checks. ``ok`` (the second return value) is True iff every
    such check passed. Unknown harness names produce a single failing row.
    ``candidates`` maps harness label -> loaded candidate for --candidate specs.
    """
    rows = []
    all_ok = True
    pins = pinned_versions() if pins is None else pins
    candidates = candidates or {}
    for name in harnesses:
        if name in candidates:
            cand_rows, cand_ok = evaluate_candidate(
                candidates[name], model, probes, pins=pins)
            rows.extend(cand_rows)
            if not cand_ok:
                all_ok = False
            continue

        spec = HARNESSES.get(name)
        if spec is None:
            rows.append({
                "harness": name, "check": "KNOWN", "ok": False,
                "detail": (f"unknown harness (have {known_harness_names()}); "
                           f"{_CANDIDATE_HINT}"),
            })
            all_ok = False
            continue

        cli_ok, cli_detail = check_cli(probes, spec["cli"])
        version_ok, version_detail = check_version(probes, name, spec["cli"], pins)
        if name == "grokbuild" and model == "gpt-5.6":
            auth_ok, auth_detail = check_subbridge(probes)
        elif model in FRONTIER_MODEL_ENV:
            auth_ok, auth_detail = _auth_frontier(probes, name, model)
        elif model in OPEN_MODEL_ENV:
            # Open model: AUTH = provider env key present (harness login is moot).
            keys_env_ok = name in {"codex", "codex_v1", "codex_v2"}
            auth_ok, auth_detail = check_open_key(
                probes, OPEN_MODEL_ENV[model], keys_env_ok=keys_env_ok)
        else:
            auth_ok, auth_detail = spec["auth"](probes)
        model_ok, model_detail = check_model(probes, name, model)

        for check, ok, detail in (
            ("CLI", cli_ok, cli_detail),
            ("VERSION", version_ok, version_detail),
            ("AUTH", auth_ok, auth_detail),
            ("MODEL", model_ok, model_detail),
        ):
            rows.append({"harness": name, "check": check, "ok": ok,
                         "detail": detail})
            if ok is False:
                all_ok = False
    return rows, all_ok


def _status(ok, check=None):
    if ok is None:
        return "INFO"
    if not ok and check == "VERSION":
        return "DRIFT"
    return "OK" if ok else "FAIL"


def format_report(rows, harnesses, docker_row, image_row=None):
    """Render the status matrix + details + Docker/image status lines."""
    lines = []

    # Status matrix: one row per harness, one column per check.
    by_harness = {}
    for row in rows:
        by_harness.setdefault(row["harness"], {})[row["check"]] = row["ok"]

    checks = list(CHECKS)
    for row in rows:
        if row["check"] in CANDIDATE_EXTRA_CHECKS and row["check"] not in checks:
            checks.append(row["check"])

    headers = ["harness"] + checks
    table = []
    for name in harnesses:
        cells = [name]
        harness_checks = by_harness.get(name, {})
        if "KNOWN" in harness_checks:  # unknown harness -> collapse across columns
            cells += ["FAIL"] * len(checks)
        else:
            cells += [_status(harness_checks.get(c), c) for c in checks]
        table.append(cells)

    widths = [len(h) for h in headers]
    for cells in table:
        for i, c in enumerate(cells):
            widths[i] = max(widths[i], len(c))

    def fmt(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines.append(fmt(headers))
    lines.append(fmt(["-" * w for w in widths]))
    lines.extend(fmt(cells) for cells in table)

    # Details block (every check, so passes are auditable too).
    lines.append("")
    lines.append("Details:")
    for row in rows:
        lines.append(f"  [{_status(row['ok'], row['check']):>5}] {row['harness']:<9} "
                     f"{row['check']:<6} {row['detail']}")

    # Docker (informational).
    ok, detail = docker_row
    lines.append("")
    lines.append(f"Docker (informational): [{_status(ok)}] {detail}")
    if image_row is not None:
        image_ok, image_detail = image_row
        image_status = "DRIFT" if image_ok is False else _status(image_ok)
        lines.append(f"Image pins: [{image_status}] {image_detail}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark preflight doctor.")
    parser.add_argument("--harness", default=None,
                        help="comma-separated harness names to check "
                             f"(default: all {ALL_HARNESSES})")
    parser.add_argument("--candidate", action="append", default=[], metavar="SPEC",
                        help="candidate TOML path or harness pack ref "
                             "org/name[@version][:manifest] (repeatable); "
                             "preflight without editing stock harness lists")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"canonical model to resolve (default: {DEFAULT_MODEL})")
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE,
                        help=f"image to compare with pins (default: {DEFAULT_IMAGE})")
    parser.add_argument("--adapters-dir", default=ADAPTERS_DIR,
                        help="adapters directory for stock DOCTOR discovery / "
                             "config-variant base checks")
    parser.add_argument("--docker-env", action="store_true",
                        help="preflight Docker environment: buildx, daemon resources, "
                             "per-task image probe, auth freshness. Independent of "
                             "--harness checks.")
    parser.add_argument("--env-requirements", default=ENV_REQUIREMENTS_PATH,
                        help=f"path to env-requirements.toml (default: {ENV_REQUIREMENTS_PATH})")
    args = parser.parse_args(argv)

    # --docker-env mode: standalone docker environment preflight.
    if args.docker_env:
        probes = Probes()
        requirements = load_env_requirements(probes, args.env_requirements)
        task_images = discover_task_images(probes)
        rows, ok, lane_results = evaluate_docker_env(
            probes, requirements, task_images)
        report = format_docker_env_report(rows, lane_results, requirements)
        print(report)
        print()
        print(f"Docker environment: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    candidates = {}
    if args.candidate:
        from .candidates import load_candidates
        try:
            candidates = load_candidates(args.candidate, args.adapters_dir)
        except (OSError, ValueError) as exc:
            print(f"doctor: failed to load --candidate: {exc}", file=sys.stderr)
            return 2

    if args.harness is None:
        harnesses = list(ALL_HARNESSES) if not candidates else []
    else:
        harnesses = [h.strip() for h in args.harness.split(",") if h.strip()]
    for name in candidates:
        if name not in harnesses:
            harnesses.append(name)
    if not harnesses:
        print("doctor: at least one --harness or --candidate is required",
              file=sys.stderr)
        return 2

    probes = Probes()

    pins = pinned_versions()
    rows, ok = evaluate(harnesses, args.model, probes, pins=pins, candidates=candidates)
    docker_row = check_docker(probes)
    stock_for_image = [h for h in harnesses if h not in candidates]
    image_row = check_image_versions(probes, stock_for_image, pins, args.docker_image)
    ok = ok and image_row[0] is not False
    print(format_report(rows, harnesses, docker_row, image_row))
    print()
    print(f"Preflight: {'PASS' if ok else 'FAIL'} "
          f"({len(harnesses)} harness(es), model={args.model})")
    registry = open_models_registry()
    if registry:
        print(f"Open-model registry: {registry}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
