"""Adapter for xAI's `grok` (Grok Build CLI) using BYOK model routes.

Headless invocation (benchmark policy is pinned here, like Codex's flags):
    HOME=<isolated tmp with generated ~/.grok/config.toml> GROK_SUBAGENTS=0
    grok --no-auto-update -p <instruction> --model <canonical-model> \
         --output-format streaming-json --always-approve --cwd <workdir>

Custom models are declared in the isolated HOME only; the user's real ~/.grok is
never read, copied, or mounted. The current Grok source parses model catalog
entries from `[model.<catalog-key>]` (``ModelEntryConfig``), while `[models]` is
the separate defaults/role-pin table. Each route explicitly sets ``base_url``,
``api_backend``, ``env_key``, and ``auth_scheme``. Subagents are disabled twice:
``GROK_SUBAGENTS=0`` in the child environment and ``[subagents] enabled=false``
in generated config, so every invocation has a stable parity guard.

In counting-proxy mode the same model entry replaces `base_url` with its
per-cell proxy URL. Open vendors use `/chat/<vendor>/<upstream-path>`; OpenAI
uses `/openai/<upstream-path>`. The proxy then forwards to the vendor unchanged.

Probe result (2026-07-07): this BYOK custom-model path works without xAI login.
`--output-format streaming-json` emits JSONL events like
``thought``/``text``/``end``.  Those events carried no token-usage fields in the
observed stream, so token accounting also reads Grok's local
``logs/unified.jsonl`` ``shell.turn.inference_done`` counters from the isolated
HOME.  Turns are counted from terminal ``end`` events (single `-p` run => 1).
"""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile

try:
    from obench.auth_persist import try_persist_auth_file
except ImportError:  # file-path / Docker mount layout
    from auth_persist import try_persist_auth_file

NAME = "grokbuild"
_EXE = "grok"

# Subscription lane: native xAI models billed to the user's Grok subscription.
# Auth is the host login at ~/.grok/auth.json, staged into the disposable HOME.
# No custom catalog is written, so Grok routes to xAI's own endpoint; the
# counting proxy cannot meter this lane (CLI-reported tokens only).
# GPT-5.6 is instead routed through the local CLIProxyAPI subscription bridge
# (see OPEN_MODELS below).
MODELS = {
    "grok-4.5": {"model_id": "grok-4.5"},
}
_REAL_GROK_AUTH = os.path.expanduser("~/.grok/auth.json")

# Grok has built-in auxiliary roles (session summaries/titles/image descriptions)
# that can otherwise fall back to the built-in `grok-build` model id. Override
# that alias to the selected BYOK endpoint and point every documented role key at
# the selected custom model so no internal request is routed to an xAI model.
_AUX_MODEL_ALIASES = ("grok-build",)

# Long-thinking coding tasks can exceed the docs' example 8192 cap, and a live
# GLM scheme-evaluator cell still truncated at 32768. Grok treats provider
# max-token truncation as fatal, so use a high per-request cap that leaves room
# for reasoning-heavy GLM/Kimi/DeepSeek runs while staying within these models'
# advertised large context windows.
_MAX_COMPLETION_TOKENS = 65536

# Match the benchmark's medium-equivalent thinking convention. Grok Build exposes
# both `--effort` and `--reasoning-effort` for this control.
_EFFORT = "medium"
# Exact OpenAI-compatible endpoint data copied from bench/adapters/pi.py.
OPEN_MODELS = {
    "glm-5.2":           {"model_id": "glm-5.2",           "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM", "proxy_route": "chat/zai"},
    "deepseek-v4-flash": {"model_id": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",     "env_key": "DEEPSEEK_API_KEY", "display": "DeepSeek", "proxy_route": "chat/deepseek"},
    "kimi-k2.7-code":    {"model_id": "kimi-k2.7-code",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi", "proxy_route": "chat/moonshot"},
    "kimi-k3":    {"model_id": "kimi-k3",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi K3", "proxy_route": "chat/moonshot"},
    "laguna-s-2.1": {"model_id": "poolside/laguna-s-2.1", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Poolside Laguna S 2.1", "proxy_route": "chat/openrouter"},
    "inkling": {"model_id": "thinkingmachines/inkling", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Thinking Machines Inkling", "proxy_route": "chat/openrouter"},
    # CLIProxyAPI owns Codex/ChatGPT subscription OAuth and refresh. The value
    # sent to its local ingress is either its optional ingress key or a harmless
    # placeholder; it is never an OpenAI API key.
    "gpt-5.6-sol":       {"model_id": "gpt-5.6-sol",       "base_url": "http://127.0.0.1:8317/v1",     "base_url_env": "CLIPROXYAPI_BASE_URL", "env_key": "CLIPROXYAPI_API_KEY", "display": "GPT-5.6 Sol via CLIProxyAPI", "proxy_route": "subbridge", "subscription_bridge": True},
}


def _grokbuild_route(entry):
    """Fill the proxy route from the provider, as the built-ins do."""
    if not entry.get("proxy_route") and entry.get("provider"):
        entry = dict(entry, proxy_route="chat/%s" % entry["provider"])
    return entry


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
    required=("proxy_route",),
    derive=_grokbuild_route,
)

_SUBBRIDGE_PLACEHOLDER = "openbench-local-ingress"


def _token_fields_none():
    return {
        "tokens_input_uncached": None,
        "tokens_cache_read": None,
        "tokens_output": None,
        "tokens_reasoning": None,
    }


def _unsupported(model):
    return {"completed": False,
            "error": f"unsupported-model: {model!r} (have {list(OPEN_MODELS)})",
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_token_fields_none()}


def _setup_needed(msg):
    return {"completed": False, "error": f"SETUP-NEEDED: {msg}",
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_token_fields_none()}


def _resolve_exe():
    return shutil.which(_EXE)


def version():
    """Return the Grok CLI version string (with binary path), or None."""
    exe = _resolve_exe()
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=5,
            stdin=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001 - version probing must never raise
        return None
    out = (proc.stdout or proc.stderr or "").strip()
    return f"{out} ({exe})" if out else None


def _proxy_cell_url(*parts):
    base = os.environ.get("OPENBENCH_PROXY_BASE_URL")
    token = os.environ.get("OPENBENCH_PROXY_CELL_TOKEN")
    if not os.environ.get("OPENBENCH_PROXY") or not base or not token:
        return None
    path = "/".join(str(p).strip("/") for p in ("cell", token, *parts) if str(p).strip("/"))
    return base.rstrip("/") + "/" + path


def _resolved_spec(spec):
    """Resolve only the route URL; credential values never enter generated config."""
    resolved = dict(spec)
    if spec.get("subscription_bridge") and os.environ.get("BENCH_IN_CONTAINER"):
        resolved["base_url"] = "http://host.docker.internal:8317/v1"
    if spec.get("base_url_env") and os.environ.get(spec["base_url_env"]):
        resolved["base_url"] = os.environ[spec["base_url_env"]]
    from urllib.parse import urlsplit, urlunsplit
    parsed = urlsplit(resolved["base_url"])
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("model base URL must not contain URL-embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("model base URL must not contain a query or fragment")
    local_hosts = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
    if parsed.scheme == "http" and parsed.hostname not in local_hosts:
        raise ValueError("remote model base URL must use HTTPS")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("model base URL has an invalid port") from exc
    if (spec.get("subscription_bridge") and os.environ.get("BENCH_IN_CONTAINER")
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
        netloc = "host.docker.internal" + (f":{port}" if port is not None else "")
        resolved["base_url"] = urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    return resolved


def _proxied_spec(spec):
    spec = _resolved_spec(spec)
    if not os.environ.get("OPENBENCH_PROXY"):
        return spec
    from urllib.parse import urlsplit
    tail = (urlsplit(spec["base_url"]).path or "").strip("/")
    url = _proxy_cell_url(*spec["proxy_route"].split("/"), tail)
    return dict(spec, base_url=url) if url else spec


def _toml_str(value):
    # JSON string syntax is valid TOML basic-string syntax for these values.
    return json.dumps(str(value))


def _model_section(alias, spec):
    return (
        f"[model.{_toml_str(alias)}]\n"
        f"model = {_toml_str(spec['model_id'])}\n"
        f"base_url = {_toml_str(spec['base_url'])}\n"
        f"name = {_toml_str(spec['display'])}\n"
        f"env_key = {_toml_str(spec['env_key'])}\n"
        'api_backend = "chat_completions"\n'
        'auth_scheme = "bearer"\n'
        "stream_tool_calls = false\n"
        "context_window = 128000\n"
        f"max_completion_tokens = {_MAX_COMPLETION_TOKENS}\n\n"
    )


def _config_toml(model, spec):
    sections = [
        "[cli]\n"
        "auto_update = false\n\n"
        "[models]\n"
        f"default = {_toml_str(model)}\n"
        f"web_search = {_toml_str(model)}\n"
        f"session_summary = {_toml_str(model)}\n"
        f"image_description = {_toml_str(model)}\n"
        f"max_completion_tokens = {_MAX_COMPLETION_TOKENS}\n"
        f"default_reasoning_effort = {_toml_str(_EFFORT)}\n\n"
        "[ui]\n"
        f"fork_secondary_model = {_toml_str(model)}\n\n"
        "[compaction.memory_flush]\n"
        f"flush_model = {_toml_str(model)}\n\n"
        "[goal]\n"
        f"planner_model = {_toml_str(model)}\n"
        f"strategist_model = {_toml_str(model)}\n"
        f"skeptic_models = [{_toml_str(model)}]\n\n"
        "[subagents]\n"
        "enabled = false\n\n"
        "[subagents.models]\n"
        f"explore = {_toml_str(model)}\n"
        f"plan = {_toml_str(model)}\n\n"
    ]
    for alias in (model, *_AUX_MODEL_ALIASES):
        sections.append(_model_section(alias, spec))
    return "".join(sections)


def _write_config(iso_home, model, spec):
    grok_dir = os.path.join(iso_home, ".grok")
    os.makedirs(grok_dir, exist_ok=True)
    path = os.path.join(grok_dir, "config.toml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_config_toml(model, spec))
    return path


def _usage_tokens(obj):
    """Best-effort usage parser for future Grok stream shape drift."""
    if not isinstance(obj, dict):
        return None
    usage = obj.get("usage") or obj.get("tokenUsage") or obj.get("tokens")
    if not isinstance(usage, dict):
        return None
    # Known/common OpenAI/Anthropic-ish field names.  The 2026-07 probe did not
    # emit any of these, but keeping this defensive parser makes future CLI
    # additions useful without changing the adapter contract.
    total = usage.get("total_tokens") or usage.get("totalTokens")
    if isinstance(total, (int, float)):
        return int(total)
    pairs = [
        ("input_tokens", "output_tokens"),
        ("prompt_tokens", "completion_tokens"),
        ("inputTokens", "outputTokens"),
    ]
    for a, b in pairs:
        if isinstance(usage.get(a), (int, float)) and isinstance(usage.get(b), (int, float)):
            return int(usage[a]) + int(usage[b])
    return None


def _parse_log_usage(grok_dir):
    """Return token usage totals from Grok's local run log, if present.

    The observed streaming-json events do not carry usage, but Grok writes one
    ``shell.turn.inference_done`` log record per model call. Counters are
    PER-CALL, not cumulative, so sum every event in the fresh isolated HOME.
    Assumption: every adapter run creates a fresh HOME, so the log contains only
    this run's events; if Grok ever reuses a HOME, filter events by session id.
    """
    log_path = os.path.join(grok_dir, "logs", "unified.jsonl")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None

    totals = {
        "tokens_input_uncached": 0,
        "tokens_cache_read": 0,
        "tokens_output": 0,
        "tokens_reasoning": 0,
    }
    found = False
    calls = 0
    # These counters come straight from Grok's own per-call records, so the
    # split is vendor-reported -- but the adapter never labelled it, leaving
    # token_basis None on 180 of 214 solved grokbuild cells and making them
    # uncostable even though every field was present.
    #
    # Unlike pi, the log carries NO total_tokens, so the total-consistency
    # invariant pi uses cannot be checked here. Only the containment checks the
    # data supports are applied; anything violating them falls back to
    # "estimated" rather than claiming vendor fidelity we did not verify.
    invariant_ok = True
    for raw in lines:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("msg") != "shell.turn.inference_done":
            continue
        ctx = obj.get("ctx") if isinstance(obj.get("ctx"), dict) else {}
        prompt = ctx.get("prompt_tokens")
        cached = ctx.get("cached_prompt_tokens") or 0
        completion = ctx.get("completion_tokens")
        reasoning = ctx.get("reasoning_tokens") or 0
        if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
            cached_i = int(cached) if isinstance(cached, (int, float)) else 0
            if cached_i > int(prompt):
                # cache-read cannot exceed the prompt it was read for; the
                # max(0, ...) below would silently absorb it into a 0.
                invariant_ok = False
            totals["tokens_input_uncached"] += max(0, int(prompt) - cached_i)
            totals["tokens_cache_read"] += cached_i
            totals["tokens_output"] += int(completion)
            if isinstance(reasoning, (int, float)):
                if int(reasoning) > int(completion):
                    invariant_ok = False
                totals["tokens_reasoning"] += int(reasoning)
            found = True
            calls += 1
        else:
            # A record we cannot parse means the summed totals are incomplete.
            invariant_ok = False
    if not found:
        return None
    totals["tokens"] = totals["tokens_input_uncached"] + totals["tokens_output"]
    totals["turns"] = calls
    totals["token_basis"] = "vendor_split" if invariant_ok else "estimated"
    return totals


def _parse_stream(stdout):
    """Parse Grok `streaming-json` stdout into (tokens, turns, tail)."""
    text_parts = []
    tokens_total = 0
    found_tokens = False
    end_events = 0
    parsed_any = False

    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        parsed_any = True
        typ = obj.get("type")
        if typ == "text" and obj.get("data"):
            text_parts.append(str(obj["data"]))
        elif typ in {"message", "assistant"} and obj.get("text"):
            text_parts.append(str(obj["text"]))
        if typ == "end":
            end_events += 1
        tok = _usage_tokens(obj)
        if tok is not None:
            tokens_total += tok
            found_tokens = True

    tail = "".join(text_parts)[-2000:]
    tokens = tokens_total if found_tokens else None
    turns = end_events or (1 if parsed_any else None)
    return tokens, turns, tail


def run(instruction: str, workdir: str, model: str, timeout_s: int) -> dict:
    subscription = model in MODELS
    if not subscription and model not in OPEN_MODELS:
        return _unsupported(model)
    route_spec = None
    spec = None
    if not subscription:
        spec = OPEN_MODELS[model]
        try:
            route_spec = _proxied_spec(spec)
        except ValueError as exc:
            return _setup_needed(str(exc))
        if not spec.get("subscription_bridge") and not os.environ.get(spec["env_key"]):
            return _setup_needed(f"export {spec['env_key']} to use {model}")
    elif not os.path.isfile(_REAL_GROK_AUTH):
        return _setup_needed(f"log in to Grok (`grok` interactive) so {_REAL_GROK_AUTH} exists")
    exe = _resolve_exe()
    if not exe:
        return _setup_needed("install Grok Build CLI (`npm install -g @xai-official/grok`) and ensure `grok` is on PATH")

    iso_home = tempfile.mkdtemp(prefix="grokbuild_home_")
    isolated_auth = None
    try:
        if subscription:
            grok_dir = os.path.join(iso_home, ".grok")
            os.makedirs(grok_dir, exist_ok=True)
            isolated_auth = os.path.join(grok_dir, "auth.json")
            shutil.copy2(_REAL_GROK_AUTH, isolated_auth)
            model = MODELS[model]["model_id"]
            env = dict(os.environ)
        else:
            grok_dir = os.path.dirname(_write_config(iso_home, model, route_spec))
            if spec.get("subscription_bridge"):
                # Filter by name before retrieving values, so a pay-per-token key
                # is neither read nor copied on this path. Grok sees only
                # CLIProxyAPI ingress auth (or a local placeholder when ingress
                # auth is off).
                env = {name: os.environ[name] for name in os.environ
                       if name != "OPENAI_API_KEY"}
                env[spec["env_key"]] = os.environ.get(spec["env_key"], _SUBBRIDGE_PLACEHOLDER)
            else:
                env = dict(os.environ)
        env["HOME"] = iso_home
        env["GROK_SUBAGENTS"] = "0"
        # Keep Grok's generated state within the disposable home and suppress
        # non-essential network work where the CLI exposes a switch.
        cmd = [
            exe,
            "--no-auto-update",
            "-p", instruction,
            "--model", model,
            "--output-format", "streaming-json",
            "--effort", _EFFORT,
            "--reasoning-effort", _EFFORT,
            "--always-approve",
            "--cwd", workdir,
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=workdir, capture_output=True, text=True,
                timeout=timeout_s, stdin=subprocess.DEVNULL, env=env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            full_output = stdout + stderr
            token_fields = _token_fields_none()
            try:
                tokens, turns, tail = _parse_stream(stdout)
                usage = _parse_log_usage(grok_dir)
                if usage is not None:
                    token_fields = {k: usage.get(k) for k in token_fields}
                    if tokens is None:
                        tokens = usage.get("tokens")
                    if usage.get("turns"):
                        turns = usage["turns"]
            except Exception:  # noqa: BLE001 - parsing must not break a run
                tokens, turns, tail = None, None, ""
            if not tail:
                tail = full_output[-2000:]
            return {"completed": False, "error": f"timeout after {timeout_s}s",
                    "output_tail": tail, "full_output": full_output,
                    "tokens": tokens, "turns": turns, "cmd": cmd,
                    **token_fields}

        combined = (proc.stdout or "") + (proc.stderr or "")
        token_fields = _token_fields_none()
        try:
            tokens, turns, tail = _parse_stream(proc.stdout or "")
            usage = _parse_log_usage(grok_dir)
            if usage is not None:
                token_fields = {k: usage.get(k) for k in token_fields}
                if tokens is None:
                    tokens = usage.get("tokens")
                if usage.get("turns"):
                    turns = usage["turns"]
        except Exception:  # noqa: BLE001 - parsing must not break a run
            tokens, turns, tail = None, None, ""
        if not tail:
            tail = combined[-2000:]
        return {"completed": proc.returncode == 0,
                "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
                "output_tail": tail, "full_output": combined,
                "tokens": tokens, "turns": turns, "cmd": cmd,
                **token_fields}
    finally:
        if isolated_auth is not None:
            try_persist_auth_file(isolated_auth, _REAL_GROK_AUTH)
        shutil.rmtree(iso_home, ignore_errors=True)
