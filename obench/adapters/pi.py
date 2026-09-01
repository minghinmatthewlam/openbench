"""Adapter for the `pi` CLI (ChatGPT-subscription "openai-codex" route).

Headless invocation:
    HOME=<isolated tmp with only .pi/agent/auth.json>
    pi -p --provider openai-codex --model gpt-5.5 \
       --thinking medium <instruction>

Notes / quirks:
- pi loads the user's personal extensions from the real ~/.pi. One of them
  (pi-goal) crashes `-p` non-interactive mode. To avoid this WITHOUT touching
  the user's config, we run pi under an ISOLATED HOME: a fresh temp dir that
  contains ONLY `.pi/agent/auth.json` copied from the real one. No settings.json
  means no personal extensions are registered; built-in factory behavior remains.
- Subscription route: provider `openai-codex` exposes `gpt-5.5`
  (verified via `pi --list-models`). The API-key `openai` provider also has
  gpt-5.5 but we prefer the subscription credential.
- Reasoning effort via `--thinking medium`.
- The real ~/.pi/agent/auth.json is only READ (copied), never modified.
- `--mode json` emits a JSONL event stream. The final `agent_end` event carries
  `messages[]`, each assistant message holding
  `usage={input,output,cacheRead,cacheWrite,totalTokens}`; `turn_end` events
  mark model rounds. Token accounting (see ``_parse_json``):
    tokens = sum of input+output over assistant messages (fresh tokens; cache
             re-reads excluded, matching the other adapters' definition).
    turns  = number of `turn_end` events (model rounds).
  Parsing is defensive: shape drift yields tokens=None/turns=None + raw tail.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

try:
    from obench.auth_persist import try_persist_auth_file
    from obench import gateway_spec as _gateway_spec
except ImportError:  # file-path / Docker mount layout
    from auth_persist import try_persist_auth_file
    import gateway_spec as _gateway_spec

NAME = "pi"
ADAPTER_API_VERSION = 2
ROUTED_CAPABILITIES = {
    "protocols": ["openai_chat", "openai_responses"],
    "execution_lanes": ["local", "docker"],
    "streaming": True,
    "dynamic_model_ids": True,
    "route_plan_transport": "sanitized_file",
}


def _doctor_auth(probes):
    """Doctor AUTH probe: isolated-HOME route needs ~/.pi/agent/auth.json + openai-codex."""
    path = "~/.pi/agent/auth.json"
    if not probes.exists(path):
        return False, f"missing {os.path.expanduser(path)}"
    data = probes.read_json(path)
    if not isinstance(data, dict):
        return False, f"unreadable JSON at {os.path.expanduser(path)}"
    if "openai-codex" in data:
        return True, "entry: openai-codex"
    return False, "no openai-codex entry in ~/.pi/agent/auth.json"


# Optional doctor metadata: scanned by obench.doctor to build the harness
# preflight table without hard-coding every adapter in doctor.py.
DOCTOR = {"cli": "pi", "auth": _doctor_auth}

# canonical model name -> pi provider/model pair. Both routes use pi's
# subscription/OAuth credentials under ~/.pi; no API key is required here.
# Thinking parity for the opus frontier lane: Anthropic Claude Opus 4.8 is run
# with pi's `--thinking medium`, matching the benchmark's medium-reasoning tier.
MODELS = {
    "gpt-5.5-medium": {"provider": "openai-codex", "model_id": "gpt-5.5", "thinking": "medium"},
    "gpt-5.6-sol": {"provider": "openai-codex", "model_id": "gpt-5.6-sol", "thinking": "medium"},
    "gpt-5.6-terra": {"provider": "openai-codex", "model_id": "gpt-5.6-terra", "thinking": "medium"},
    "gpt-5.6-luna": {"provider": "openai-codex", "model_id": "gpt-5.6-luna", "thinking": "medium"},
    "claude-opus-4-8": {"provider": "anthropic", "model_id": "claude-opus-4-8", "thinking": "medium"},
    "grok-4.5": {"provider": "xai", "model_id": "grok-4.5", "thinking": "medium"},
}
_REAL_AUTH = os.path.expanduser("~/.pi/agent/auth.json")
_EXE = "pi"
_ROUTED_PROVIDER = "openbench-routed"
_SYNTHETIC_API_KEY = "openbench-routed-synthetic"
_ROUTE_PLAN_FIELDS = {
    "schema_version", "experiment_digest", "arm_digest", "arm_id",
    "route_kind", "endpoint", "protocol", "canonical_model", "requested_model",
    "requested_provider", "allowed_models", "allowed_providers",
    "fallback_enabled", "retry_count", "cache_enabled", "auth_env",
    "sampling", "allow_private_endpoint", "private_host_allowlist",
    "private_cidr_allowlist",
}
_SAMPLING_FIELDS = {"temperature", "top_p", "seed"}
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_CELL_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ROUTED_ENV_ALLOWLIST = {
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP",
    "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS",
    "NO_PROXY", "no_proxy",
}


def _empty_token_usage():
    return {
        "tokens_input_uncached": None,
        "tokens_cache_read": None,
        "tokens_cache_write": None,
        "tokens_output": None,
        "tokens_reasoning": None,
        "usage_raw": None,
        "token_basis": None,
        "replies_ok": None,
        "replies_throttled": None,
    }


def _legacy_tokens(token_usage):
    # Delegated TOKEN_PARITY contract: keep the legacy scalar as
    # uncached_input + output. Cache reads and cache writes remain available in
    # split fields but are intentionally not folded into this compatibility
    # value.
    inp = token_usage.get("tokens_input_uncached")
    out = token_usage.get("tokens_output")
    if isinstance(inp, int) and isinstance(out, int):
        return inp + out
    return None


def _num(value):
    return int(value) if isinstance(value, (int, float)) else None


def _proxy_cell_url(*parts):
    base = os.environ.get("OPENBENCH_PROXY_BASE_URL")
    token = os.environ.get("OPENBENCH_PROXY_CELL_TOKEN")
    if not os.environ.get("OPENBENCH_PROXY") or not base or not token:
        return None
    path = "/".join(str(p).strip("/") for p in ("cell", token, *parts) if str(p).strip("/"))
    return base.rstrip("/") + "/" + path


def _proxied_base_url(route, original_url=None):
    if not os.environ.get("OPENBENCH_PROXY"):
        return original_url
    if route == "codex":
        return _proxy_cell_url("codex", "backend-api")
    parsed = urlsplit(original_url or "")
    tail = (parsed.path or "").strip("/")
    vendor = route
    return _proxy_cell_url("chat", vendor, tail)


def version():
    """Return the CLI version string (with binary path), or None on failure.

    Cheap `pi --version` (short-circuits before extensions load, so no isolated
    HOME needed); never raises (the runner calls this defensively).
    """
    try:
        proc = subprocess.run(
            [_EXE, "--version"],
            capture_output=True, text=True, timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - version probing must never raise
        return None
    out = (proc.stdout or proc.stderr or "").strip()
    if not out:
        return None
    path = shutil.which(_EXE)
    return f"{out} ({path})" if path else out


_DELTA_MARKER = '"type":"message_update"'


def _run_streaming(cmd, cwd, timeout_s, env):
    """Run pi consuming stdout line-by-line, dropping per-token delta events.

    ``--mode json`` re-emits the FULL accumulated partial message inside every
    ``message_update`` delta event, so a single long reasoning turn produces
    output quadratic in its token count (observed: GBs on 32k-token turns,
    OOM-killing the container). The parser only needs ``turn_end``/``agent_end``
    events, which carry final content and usage — so delta lines are discarded
    at read time instead of buffered.

    Returns (stdout_text, stderr_text, returncode, timed_out).
    """
    import threading

    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, stdin=subprocess.DEVNULL, env=env,
    )
    out_lines, err_chunks = [], []

    def _drain_stdout():
        for line in proc.stdout:
            if _DELTA_MARKER not in line:
                out_lines.append(line)
        proc.stdout.close()

    def _drain_stderr():
        for chunk in proc.stderr:
            err_chunks.append(chunk)
        proc.stderr.close()

    t_out = threading.Thread(target=_drain_stdout, daemon=True)
    t_err = threading.Thread(target=_drain_stderr, daemon=True)
    t_out.start()
    t_err.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    t_out.join(timeout=10)
    t_err.join(timeout=10)
    return "".join(out_lines), "".join(err_chunks), proc.returncode, timed_out


def _err_tail(exc, limit=2000):
    """Last `limit` chars of a TimeoutExpired's captured output, decoding safely.

    On TimeoutExpired, `.stdout`/`.stderr` may be bytes (even under text=True),
    str, or None. Concatenating bytes with the ``""`` fallback raises TypeError,
    so decode each part first — the handler must always yield a clean tail.
    """
    def _dec(x):
        if x is None:
            return ""
        return x.decode("utf-8", "replace") if isinstance(x, bytes) else x
    text = _dec(exc.stdout) + _dec(exc.stderr)
    return text if limit is None else text[-limit:]


# --- M4 open models (first-party pay-per-token, OpenAI-compatible) ----------
# Wired via a pi provider EXTENSION written into the isolated temp HOME and
# loaded with `-e` (works even under --no-extensions). Nothing touches the
# user's ~/.pi. apiKey uses pi's "$ENV_KEY" env resolution. Base URLs verified
# from official docs 2026-07. Key-gated in run().
#
# Thinking parity: every open model is registered as reasoning-capable and run
# with `--thinking medium`. Per-model compat maps that to the closest vendor
# thinking-on behavior: GLM-5.2 medium -> Z.ai `reasoning_effort=high`; DeepSeek,
# Kimi, and GLM-4.7 Flash use the vendor's thinking-on default (no medium level).
# (Duplicated across pi/opencode/codex so each adapter stays self-contained.)
OPEN_MODELS = {
    "glm-5.2":           {"provider": "zai",      "model_id": "glm-5.2",           "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True, "thinkingFormat": "zai"},      "thinkingLevelMap": {"minimal": None, "low": "high", "medium": "high", "high": "high", "xhigh": "max"}},
    "glm-4.7-flash":     {"provider": "zai",      "model_id": "glm-4.7-flash",     "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False, "thinkingFormat": "zai"},     "thinkingLevelMap": {"off": None}},
    "deepseek-v4-flash": {"provider": "openrouter", "model_id": "deepseek/deepseek-v4-flash", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter DeepSeek V4 Flash",      "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False, "thinkingFormat": "deepseek", "requiresReasoningContentOnAssistantMessages": True}, "thinkingLevelMap": {"off": None}},
    "kimi-k2.7-code":    {"provider": "moonshot", "model_id": "kimi-k2.7-code",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi", "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False, "maxTokensField": "max_tokens", "supportsStrictMode": False, "thinkingFormat": "deepseek"}, "thinkingLevelMap": {"off": None}},
    "kimi-k3":    {"provider": "moonshot", "model_id": "kimi-k3",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi K3", "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": False, "maxTokensField": "max_tokens", "supportsStrictMode": False, "thinkingFormat": "deepseek"}, "thinkingLevelMap": {"off": None}},
    "laguna-s-2.1": {"provider": "openrouter", "context_window": 262144, "max_tokens": 32768, "model_id": "poolside/laguna-s-2.1", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Poolside Laguna S 2.1", "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True, "supportsStrictMode": False}, "thinkingLevelMap": {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}},
    "inkling": {"provider": "openrouter", "context_window": 524288, "max_tokens": 32768, "model_id": "thinkingmachines/inkling", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Thinking Machines Inkling", "thinking": "medium", "compat": {"supportsStore": False, "supportsDeveloperRole": False, "supportsReasoningEffort": True, "supportsStrictMode": False}, "thinkingLevelMap": {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "high"}},
}


def _load_open_models():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_open_models.py")
    spec = importlib.util.spec_from_file_location("openbench_open_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The dict above is the default. An optional TOML registry
# (``.openbench/open_models.toml`` or ``~/.openbench/open_models.toml``, see
# ``_open_models.py``) may add entries or retune existing ones, so a user can
# carry a BYO route without a diff against the adapter. ``OPEN_MODELS_SOURCE``
# is the file it came from, or None when the built-ins are untouched.
OPEN_MODELS, OPEN_MODELS_SOURCE = _load_open_models().load(
    NAME, OPEN_MODELS,
    required=("provider", "thinking"),
    defaults={
        "thinking": "medium",
        "compat": {"supportsStore": False, "supportsDeveloperRole": False,
                   "supportsReasoningEffort": False},
        "thinkingLevelMap": {"off": None},
    },
)


def _unsupported(model):
    known = list(MODELS) + list(OPEN_MODELS)
    return {"completed": False, "error": f"unsupported-model: {model!r} (have {known})",
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_empty_token_usage()}


def _setup_needed(env_key, model):
    return {"completed": False,
            "error": f"SETUP-NEEDED: export {env_key} to use {model}",
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_empty_token_usage()}


def _subscription_setup_needed(provider, model):
    return {"completed": False,
            "error": (f"SETUP-NEEDED: login to pi provider {provider!r} for {model} "
                      f"(missing provider credential in {_REAL_AUTH})"),
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_empty_token_usage()}


def _has_subscription_auth(provider):
    try:
        with open(_REAL_AUTH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and provider in data


def _pi_models_override(base_url):
    return json.dumps({"providers": {"openai-codex": {"baseUrl": base_url}}}, indent=2)


_MODEL_LIMITS_CACHE = {}
# Where the pinned limits live. In the container the adapter is file-loaded from
# /bench/adapters with the JSON mounted alongside entry.py; in a checkout it sits
# under data/. Both are checked, nearest first.
_MODEL_LIMITS_PATHS = (
    "/bench/model_limits.json",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "model_limits.json"),
)


def _model_limits(model_key):
    """Pinned context window and output cap for an open model.

    Raises rather than defaulting. The previous code did
    ``spec.get('context_window', 128000)`` / ``spec.get('max_tokens', 8192)``,
    and five of seven open models silently inherited those invented numbers --
    deepseek-v4-flash ran at 128000/8192 against a real 1048576/393216. It went
    unnoticed for weeks precisely because a plausible default never complains.

    It is not a cosmetic mismatch: pi clamps each request to
    ``min(maxTokens, max(1, contextWindow - promptTokens - 4096))``, so an
    understated context window drives the reply budget to 1 token as the
    conversation grows and the model is then scored as answering wrong.

    Regenerate with ``python3 -m obench.fetch_model_limits --write``; never
    hand-write values here.
    """
    if not _MODEL_LIMITS_CACHE:
        for path in _MODEL_LIMITS_PATHS:
            try:
                with open(path, encoding="utf-8") as fh:
                    _MODEL_LIMITS_CACHE.update(json.load(fh))
                break
            except (OSError, json.JSONDecodeError):
                continue
    entry = _MODEL_LIMITS_CACHE.get(model_key)
    if not entry:
        raise RuntimeError(
            f"no pinned context/output limits for {model_key!r} (looked in "
            f"{', '.join(_MODEL_LIMITS_PATHS)}). Run "
            f"`python3 -m obench.fetch_model_limits --fetched-at <date> --write`. "
            f"There is deliberately no default: the old 128000/8192 fallback "
            f"understated deepseek by 8x on context and 48x on output.")
    return entry


def _limits_provenance(model_key):
    """The declared limits for a cell, for the results row.

    Native (subscription) models carry None: pi supplies their limits from its
    own catalog and sends no cap, which is itself the asymmetry worth seeing in
    the data -- gpt-5.6-sol ran uncapped while every open model was capped.
    """
    if model_key not in OPEN_MODELS:
        return {"model_context_window": None, "model_max_tokens": None}
    limits = _model_limits(model_key)
    return {"model_context_window": limits["context_window"],
            "model_max_tokens": limits["max_tokens"]}


def _pi_provider_ext(spec, model_key):
    """JS extension source registering the open provider (loaded via -e).

    pi resolves "$ENV_KEY" in apiKey from the environment; api
    "openai-completions" appends /chat/completions to baseUrl. The model
    metadata advertises reasoning plus vendor-specific thinking controls so the
    CLI's `--thinking medium` becomes a real thinking-on request.
    """
    return (
        "export default function (pi) {\n"
        f'  pi.registerProvider("{spec["provider"]}", {{\n'
        f'    name: "{spec["display"]}",\n'
        f'    baseUrl: "{_proxied_base_url(spec["provider"], spec["base_url"])}",\n'
        f'    apiKey: "${spec["env_key"]}",\n'
        '    api: "openai-completions",\n'
        "    models: [{\n"
        f'      id: "{spec["model_id"]}", name: "{spec["model_id"]}",\n'
        "      reasoning: true, input: [\"text\"],\n"
        f'      compat: {json.dumps(spec["compat"])},\n'
        f'      thinkingLevelMap: {json.dumps(spec["thinkingLevelMap"])},\n'
        "      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },\n"
        f"      contextWindow: {_model_limits(model_key)['context_window']}, "
        f"maxTokens: {_model_limits(model_key)['max_tokens']}\n"
        "    }]\n"
        "  });\n"
        "}\n"
    )


def _route_error(message):
    raise _gateway_spec.GatewaySpecError(f"invalid sanitized RoutePlan: {message}")


def _strict_keys(value, expected, path):
    if not isinstance(value, dict):
        _route_error(f"{path} must be an object")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        _route_error(f"{path} has unknown fields: {', '.join(unknown)}")
    if missing:
        _route_error(f"{path} missing fields: {', '.join(missing)}")
    return value


def _route_string(value, path, pattern=None):
    if not isinstance(value, str) or not value.strip():
        _route_error(f"{path} must be a non-empty string")
    result = value.strip()
    if pattern is not None and not pattern.fullmatch(result):
        _route_error(f"{path} has invalid format")
    return result


def _route_string_tuple(value, path, pattern=None):
    if not isinstance(value, list) or not value:
        _route_error(f"{path} must be a non-empty array")
    result = tuple(
        _route_string(item, f"{path}[{index}]", pattern)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        _route_error(f"{path} must not contain duplicates")
    return result


def _route_bool(value, path):
    if not isinstance(value, bool):
        _route_error(f"{path} must be a boolean")
    return value


def _route_int(value, path, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _route_error(f"{path} must be an integer of at least {minimum}")
    return value


def _route_number(value, path, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _route_error(f"{path} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        _route_error(f"{path} must be between {minimum} and {maximum}")
    return result


def _load_route_plan(route_plan_path):
    path = Path(route_plan_path)
    try:
        if path.stat().st_size > 1024 * 1024:
            _route_error("file exceeds 1 MiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
    except _gateway_spec.GatewaySpecError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _route_error(f"cannot load {path}: {exc}")

    data = _strict_keys(raw, _ROUTE_PLAN_FIELDS, "route plan")
    schema_version = _route_int(data["schema_version"], "schema_version", 1)
    if schema_version != _gateway_spec.SCHEMA_VERSION:
        _route_error(f"schema_version must be {_gateway_spec.SCHEMA_VERSION}")
    protocol = _route_string(data["protocol"], "protocol")
    if protocol not in _gateway_spec.PROTOCOLS:
        _route_error(
            "protocol must be one of: "
            + ", ".join(sorted(_gateway_spec.PROTOCOLS))
        )
    route_kind = _route_string(data["route_kind"], "route_kind")
    if route_kind not in {"direct", "gateway"}:
        _route_error("route_kind must be 'direct' or 'gateway'")

    allowed_models = _route_string_tuple(data["allowed_models"], "allowed_models")
    allowed_providers = _route_string_tuple(
        data["allowed_providers"], "allowed_providers", _ID_RE)
    requested_model = _route_string(data["requested_model"], "requested_model")
    canonical_model = _route_string(data["canonical_model"], "canonical_model")
    requested_provider = _route_string(
        data["requested_provider"], "requested_provider", _ID_RE)

    fallback_enabled = _route_bool(data["fallback_enabled"], "fallback_enabled")
    retry_count = _route_int(data["retry_count"], "retry_count")
    cache_enabled = _route_bool(data["cache_enabled"], "cache_enabled")
    if fallback_enabled:
        _route_error("fallback must be disabled")
    if requested_model not in allowed_models:
        _route_error("allowed_models must contain requested_model")
    if requested_provider not in allowed_providers:
        _route_error("allowed_providers must contain requested_provider")
    if retry_count != 0 or cache_enabled:
        _route_error("retries and cache must be disabled")

    sampling_data = _strict_keys(data["sampling"], _SAMPLING_FIELDS, "sampling")
    sampling = _gateway_spec.Sampling(
        temperature=_route_number(
            sampling_data["temperature"], "sampling.temperature", 0.0, 2.0),
        top_p=_route_number(sampling_data["top_p"], "sampling.top_p", 0.0, 1.0),
        seed=_route_int(sampling_data["seed"], "sampling.seed"),
    )

    allow_private_endpoint = _route_bool(data["allow_private_endpoint"], "allow_private_endpoint")
    hosts_raw = data["private_host_allowlist"]
    cidrs_raw = data["private_cidr_allowlist"]
    try:
        hosts, cidrs = _gateway_spec._parse_allowlists(  # noqa: SLF001
            {
                "private_host_allowlist": hosts_raw,
                "private_cidr_allowlist": cidrs_raw,
            },
            allow_private_endpoint,
        )
        endpoint = _gateway_spec._validate_endpoint(  # noqa: SLF001
            data["endpoint"], "endpoint", allow_private_endpoint, hosts, cidrs)
    except _gateway_spec.GatewaySpecError as exc:
        _route_error(str(exc))
    suffix = _gateway_spec.PROTOCOL_ENDPOINT_SUFFIXES[protocol]
    if not urlsplit(endpoint).path.rstrip("/").endswith(suffix):
        _route_error(f"endpoint path must end with {suffix}")

    return _gateway_spec.RoutePlan(
        schema_version=schema_version,
        experiment_digest=_route_string(
            data["experiment_digest"], "experiment_digest", _DIGEST_RE),
        arm_digest=_route_string(data["arm_digest"], "arm_digest", _DIGEST_RE),
        arm_id=_route_string(data["arm_id"], "arm_id", _ID_RE),
        route_kind=route_kind,
        endpoint=endpoint,
        protocol=protocol,
        canonical_model=canonical_model,
        requested_model=requested_model,
        requested_provider=requested_provider,
        allowed_models=allowed_models,
        allowed_providers=allowed_providers,
        fallback_enabled=fallback_enabled,
        retry_count=retry_count,
        cache_enabled=cache_enabled,
        auth_env=_route_string(data["auth_env"], "auth_env", _ENV_RE),
        sampling=sampling,
        allow_private_endpoint=allow_private_endpoint,
        private_host_allowlist=hosts,
        private_cidr_allowlist=cidrs,
    )


def _routed_proxy_url(plan):
    if os.environ.get("OPENBENCH_PROXY") != "1":
        _route_error("OPENBENCH_PROXY=1 is required")
    base = os.environ.get("OPENBENCH_PROXY_BASE_URL", "")
    token = os.environ.get("OPENBENCH_PROXY_CELL_TOKEN", "")
    parsed = urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        _route_error("OPENBENCH_PROXY_BASE_URL must be an absolute HTTP(S) URL")
    if not _CELL_TOKEN_RE.fullmatch(token):
        _route_error("OPENBENCH_PROXY_CELL_TOKEN has invalid format")

    return (
        f"{base.rstrip('/')}/cell/{token}/route/{plan.arm_digest}"
    )


def _routed_provider_ext(plan, proxy_url):
    api = (
        "openai-responses"
        if plan.protocol == "openai_responses"
        else "openai-completions"
    )
    compat = (
        "supportsDeveloperRole: true, "
        'sessionAffinityFormat: "openai-nosession", '
        "supportsLongCacheRetention: false, supportsToolSearch: false"
        if plan.protocol == "openai_responses"
        else (
            "supportsStore: false, supportsDeveloperRole: false, "
            'supportsUsageInStreaming: true, maxTokensField: "max_tokens"'
        )
    )
    return (
        "export default function (pi) {\n"
        f"  pi.registerProvider({json.dumps(_ROUTED_PROVIDER)}, {{\n"
        '    name: "OpenBench routed proxy",\n'
        f"    baseUrl: {json.dumps(proxy_url)},\n"
        f"    apiKey: {json.dumps(_SYNTHETIC_API_KEY)},\n"
        f"    api: {json.dumps(api)},\n"
        "    models: [{\n"
        f"      id: {json.dumps(plan.requested_model)},\n"
        f"      name: {json.dumps(plan.requested_model)},\n"
        '      reasoning: false, input: ["text"],\n'
        f"      compat: {{ {compat} }},\n"
        "      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },\n"
        "      contextWindow: 128000, maxTokens: 16384\n"
        "    }]\n"
        "  });\n"
        "}\n"
    )


def _routed_child_env(iso_home):
    env = {
        key: value for key, value in os.environ.items()
        if key in _ROUTED_ENV_ALLOWLIST
    }
    env["HOME"] = iso_home
    env["PI_CODING_AGENT_DIR"] = os.path.join(iso_home, ".pi", "agent")
    env["PI_TELEMETRY"] = "0"
    return env


def _parse_json_with_usage(stdout):
    """Parse pi's JSONL event stream into (tokens, turns, tail, token_usage).

    Split usage is summed from per-turn ``turn_end.message.usage`` records as
    verified in TOKEN_PARITY.md. ``tokens`` remains the legacy scalar:
    uncached input + output, with cache reads excluded.
    """
    events = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not events:
        return None, None, "", _empty_token_usage()

    turns = sum(1 for e in events if e.get("type") == "turn_end") or None

    def _split_from_usages(usages):
        usage_raw = []
        invariant_ok = True
        totals = {
            "tokens_input_uncached": 0,
            "tokens_cache_read": 0,
            "tokens_cache_write": 0,
            "tokens_output": 0,
            "tokens_reasoning": 0,
        }
        for usage in usages:
            if not isinstance(usage, dict):
                continue
            inp = _num(usage.get("input"))
            cache_read = _num(usage.get("cacheRead"))
            cache_write = _num(usage.get("cacheWrite"))
            out = _num(usage.get("output"))
            reasoning = _num(usage.get("reasoning"))
            total = _num(usage.get("totalTokens"))
            if None in (inp, cache_read, cache_write, out):
                invariant_ok = False
                continue
            if total is None or inp + cache_read + cache_write + out != total:
                invariant_ok = False
            if reasoning is None or reasoning > out:
                invariant_ok = False
                reasoning = 0 if reasoning is None else reasoning
            usage_raw.append(usage)
            totals["tokens_input_uncached"] += inp
            totals["tokens_cache_read"] += cache_read
            totals["tokens_cache_write"] += cache_write
            totals["tokens_output"] += out
            totals["tokens_reasoning"] += reasoning
        if not usage_raw:
            return _empty_token_usage()
        out = _empty_token_usage()
        out.update(totals)
        out["usage_raw"] = usage_raw
        out["token_basis"] = "vendor_split" if invariant_ok else "estimated"
        return out

    token_usage = _split_from_usages(
        (ev.get("message") or {}).get("usage")
        for ev in events
        if ev.get("type") == "turn_end"
    )

    # Prefer the final agent_end message list; fall back to message_end events.
    messages = None
    for e in events:
        if e.get("type") == "agent_end" and isinstance(e.get("messages"), list):
            messages = e["messages"]
    if messages is None:
        messages = [e.get("message") for e in events
                    if e.get("type") == "message_end"]

    if token_usage.get("token_basis") is None:
        # Older/documented pi JSON shapes put usage on assistant messages in
        # agent_end/message_end rather than on turn_end.message. Keep that
        # surface as a fallback, but prefer turn_end to avoid double-counting
        # when both are present.
        token_usage = _split_from_usages(
            msg.get("usage")
            for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "assistant"
        )

    transcript = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for part in (msg.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                transcript.append(part["text"])

    tail = "\n".join(transcript)[-2000:]

    # Reply health: how many assistant replies actually arrived vs died on a
    # provider throttle. This is the signal that made a 429 storm invisible at
    # the row level -- a storm cell completes, produces SOME tokens, and gets a
    # real checker verdict, so failure_class/error/turns/tokens all look like an
    # ordinary wrong answer. wide25 laguna: 38 of 50 cells had more 429-killed
    # replies than delivered ones (370 messages ending in "429: ... temporarily
    # rate-limited") while deepseek ran 0/50 -- and the row for such a cell read
    # `wrong_answer, error=None, turns=16`. Rows must carry what the verdict
    # depends on, or every row-level tool inherits the blindness.
    ok = throttled = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if msg.get("stopReason") == "error":
            err = str(msg.get("errorMessage") or "")
            if "429" in err or "rate limit" in err.lower() or "rate-limit" in err.lower():
                throttled += 1
        else:
            ok += 1
    token_usage["replies_ok"] = ok
    token_usage["replies_throttled"] = throttled
    return _legacy_tokens(token_usage), turns, tail, token_usage



def _parse_json(stdout):
    """Backward-compatible parser returning legacy fields only."""
    tokens, turns, tail, token_usage = _parse_json_with_usage(stdout)
    return tokens, turns, tail


def run_routed(
    instruction: str,
    workdir: str,
    route_plan_path: str,
    timeout_s: int,
) -> dict:
    plan = _load_route_plan(route_plan_path)
    proxy_url = _routed_proxy_url(plan)
    iso_home = tempfile.mkdtemp(prefix="pi_routed_home_")
    try:
        ext_path = os.path.join(iso_home, "routed-provider.mjs")
        with open(ext_path, "w", encoding="utf-8") as fh:
            fh.write(_routed_provider_ext(plan, proxy_url))
        cmd = [
            _EXE, "-p", "--no-approve", "--no-extensions",
            "-e", ext_path,
            "--provider", _ROUTED_PROVIDER,
            "--model", plan.requested_model,
            "--mode", "json",
            instruction,
        ]
        stdout_text, stderr_text, returncode, timed_out = _run_streaming(
            cmd, workdir, timeout_s, _routed_child_env(iso_home))
        combined = stdout_text + stderr_text
        if timed_out:
            return {
                "completed": False, "error": f"timeout after {timeout_s}s",
                "output_tail": combined[-2000:], "full_output": combined,
                "tokens": None, "turns": None, "cmd": cmd,
                **_empty_token_usage(),
            }
        try:
            tokens, turns, tail, token_usage = _parse_json_with_usage(stdout_text)
        except Exception:  # noqa: BLE001
            tokens, turns, tail, token_usage = None, None, "", _empty_token_usage()
        return {
            "completed": returncode == 0,
            "error": None if returncode == 0 else f"exit {returncode}",
            "output_tail": tail or combined[-2000:],
            "full_output": combined,
            "tokens": tokens,
            "turns": turns,
            "cmd": cmd,
            # Routed runs get their limits from the route plan's provider
            # config, not from OPEN_MODELS, so there is nothing of ours to
            # stamp here; canonical_model records which model it was.
            **_limits_provenance(plan.canonical_model),
            **token_usage,
        }
    finally:
        shutil.rmtree(iso_home, ignore_errors=True)


def run(instruction: str, workdir: str, model: str, timeout_s: int) -> dict:
    if model in MODELS:
        provider = MODELS[model]["provider"]
        if not _has_subscription_auth(provider):
            return _subscription_setup_needed(provider, model)
    elif model in OPEN_MODELS:
        spec = OPEN_MODELS[model]
        if not os.environ.get(spec["env_key"]):
            return _setup_needed(spec["env_key"], model)
    else:
        return _unsupported(model)

    iso_home = tempfile.mkdtemp(prefix="pi_home_")
    isolated_auth = None
    try:
        env = dict(os.environ)
        env["HOME"] = iso_home
        # PI_CODING_AGENT_DIR overrides HOME; always replace an inherited owner
        # value so settings/resources/auth cannot escape the isolated tree.
        env["PI_CODING_AGENT_DIR"] = os.path.join(iso_home, ".pi", "agent")
        env.pop("PI_CODING_AGENT_SESSION_DIR", None)
        env.pop("PI_PACKAGE_DIR", None)

        if model in MODELS:
            # Subscription route: isolate HOME with only the copied auth.json.
            spec = MODELS[model]
            agent_dir = os.path.join(iso_home, ".pi", "agent")
            os.makedirs(agent_dir, exist_ok=True)
            isolated_auth = os.path.join(agent_dir, "auth.json")
            shutil.copy2(_REAL_AUTH, isolated_auth)
            proxy_url = _proxied_base_url("codex")
            if proxy_url:
                with open(os.path.join(agent_dir, "models.json"), "w", encoding="utf-8") as fh:
                    fh.write(_pi_models_override(proxy_url))
            cmd = [
                "pi", "-p",
                # Benchmark workspaces are data, not executable configuration.
                # This preserves Pi's built-in factory tools while preventing a
                # task's .pi extensions/packages from running in the harness.
                "--no-approve",
                "--provider", spec["provider"],
                "--model", spec["model_id"],
                "--thinking", spec["thinking"],
                "--mode", "json",
                instruction,
            ]
        else:
            # Open model: register the provider via a temp extension (env key
            # supplies auth). No subscription auth.json needed.
            spec = OPEN_MODELS[model]
            # Retry tuning for storm-prone providers, from forensics on the
            # laguna 429 storm: pi's default first retry lands at 2s, squarely
            # in the 82%-failure spacing band, and the SDK-layer retry defaults
            # to ZERO attempts. Base 6s puts retries in the >=5s band that
            # succeeded ~95% mid-storm. Written into the ISOLATED home so the
            # user's real settings.json is never touched.
            settings_dir = os.path.join(iso_home, ".pi", "agent")
            os.makedirs(settings_dir, exist_ok=True)
            settings_path = os.path.join(settings_dir, "settings.json")
            if not os.path.exists(settings_path):
                with open(settings_path, "w", encoding="utf-8") as fh:
                    json.dump({"retry": {"enabled": True, "maxRetries": 5,
                                         "baseDelayMs": 6000,
                                         "provider": {"maxRetries": 1}}}, fh)
            ext_path = os.path.join(iso_home, "open-provider.mjs")
            with open(ext_path, "w", encoding="utf-8") as fh:
                fh.write(_pi_provider_ext(spec, model))
            cmd = [
                "pi", "-p",
                "--no-approve",
                "-e", ext_path,
                "--provider", spec["provider"],
                "--model", spec["model_id"],
                "--thinking", spec["thinking"],
                "--mode", "json",
                instruction,
            ]

        stdout_text, stderr_text, returncode, timed_out = _run_streaming(
            cmd, workdir, timeout_s, env)
        if timed_out:
            full_output = stdout_text + stderr_text
            return {
                "completed": False,
                "error": f"timeout after {timeout_s}s",
                "output_tail": full_output[-2000:],
                "full_output": full_output,
                "tokens": None,
                "turns": None,
                "cmd": cmd,
                **_empty_token_usage(),
            }

        combined = stdout_text + stderr_text
        try:
            tokens, turns, tail, token_usage = _parse_json_with_usage(stdout_text)
        except Exception:  # noqa: BLE001 - never let usage parsing break a run
            tokens, turns, tail, token_usage = None, None, "", _empty_token_usage()
        if not tail:
            tail = combined[-2000:]

        return {
            "completed": returncode == 0,
            "error": None if returncode == 0 else f"exit {returncode}",
            "output_tail": tail,
            # Optional (ADAPTER_SPEC v1): full untruncated stdout+stderr for the
            # runner's local transcript. LOCAL-ONLY; never published unscrubbed.
            "full_output": combined,
            "tokens": tokens,
            "turns": turns,
            "cmd": cmd,
            # Stamp the limits this cell actually ran under. Without them, an
            # arm handicapped by an understated context window looks in the data
            # exactly like one that simply performed worse -- which is how
            # deepseek running at 128000/8192 went unnoticed for weeks.
            **_limits_provenance(model),
            **token_usage,
        }
    finally:
        if isolated_auth is not None:
            try_persist_auth_file(isolated_auth, _REAL_AUTH)
        shutil.rmtree(iso_home, ignore_errors=True)
