"""Adapter for the `opencode` CLI (ChatGPT-subscription OAuth route).

Headless invocation:
    OPENAI_API_KEY unset in child env
    opencode run --dir <workdir> -m openai/gpt-5.5 --variant medium \
        --auto --format json <instruction>

Notes / quirks:
- OPENAI_API_KEY MUST be stripped from the child env. If it is present,
  opencode uses the API-key provider instead of the stored subscription
  OAuth credential (~/.local/share/opencode/auth.json); stripping it forces
  the subscription route (verified).
- `--variant medium` selects the reasoning effort for the model.
- `--auto` auto-approves tool permissions so file edits happen unattended.
  opencode `run` is non-interactive, but write/edit permission is otherwise
  gated; --auto is required for the agent to modify files headlessly.
- `--dir` sets the working directory the agent operates in.
- `--format json` emits a JSONL event stream. Each ``step_finish`` event is one
  model round and carries ``part.tokens={input,output,reasoning,cache{...}}``.
  Token accounting (see ``_parse_json``):
    tokens_input_uncached/cache/output/reasoning are summed across
    ``step_finish`` records. opencode reports visible output and reasoning as
    separate fields, so tokens_output is normalized to output+reasoning. The
    legacy tokens scalar is tokens_input_uncached + tokens_output, with cache
    reads excluded.
    turns  = number of step_finish events (model rounds; one assistant message
             each).
  Parsing is defensive: on any shape drift it yields tokens=None/turns=None and
  the raw output as the tail, never raising.
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

NAME = "opencode"
_EXE = "opencode"


def _empty_token_usage():
    return {
        "tokens_input_uncached": None,
        "tokens_cache_read": None,
        "tokens_cache_write": None,
        "tokens_output": None,
        "tokens_reasoning": None,
        "usage_raw": None,
        "token_basis": None,
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


def _proxied_base_url(spec):
    if not os.environ.get("OPENBENCH_PROXY"):
        return spec["base_url"]
    from urllib.parse import urlsplit
    tail = (urlsplit(spec["base_url"]).path or "").strip("/")
    return _proxy_cell_url("chat", spec["provider"], tail)

# canonical model name -> opencode `-m` model string (provider/model)
MODELS = {
    "gpt-5.5-medium": "openai/gpt-5.5",
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    # Thinking parity for the opus frontier lane: Anthropic's opencode provider
    # gets the same medium-equivalent request via `--variant medium`.
    "claude-opus-4-8": "anthropic/claude-opus-4-8",
    "grok-4.5": "xai/grok-4.5",
}

# canonical model name -> `--variant` reasoning effort
_VARIANT = {
    "gpt-5.5-medium": "medium",
    "gpt-5.6-sol": "medium",
    "gpt-5.6-terra": "medium",
    "gpt-5.6-luna": "medium",
    "claude-opus-4-8": "medium",
    "grok-4.5": None,  # xai serves grok-4.5 without an effort selector
}

# opencode's Anthropic OAuth login (`opencode auth login -p anthropic`) writes
# here on current releases. Older OpenAI subscription paths are still mounted by
# docker_exec; this guard is only for the new Anthropic frontier route.
_AUTH_CANDIDATES = (
    os.path.expanduser("~/.local/share/opencode/auth.json"),
    os.path.expanduser("~/.opencode/data/auth.json"),
)
_ANTHROPIC_AUTH = next((path for path in _AUTH_CANDIDATES if os.path.isfile(path)),
                       _AUTH_CANDIDATES[-1])


def _has_anthropic_oauth():
    try:
        with open(_ANTHROPIC_AUTH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    text = json.dumps(data).lower()
    return "anthropic" in text and "oauth" in text

# --- M4 open models (first-party pay-per-token, OpenAI-compatible) ----------
# Wired via a custom provider passed through OPENCODE_CONFIG_CONTENT (inline
# JSON env var) so nothing touches the user's opencode config and the temp
# workspace stays clean. apiKey uses opencode's {env:VAR} interpolation. Base
# URLs verified from official docs 2026-07. Key-gated in run().
#
# Thinking parity: run every open model with opencode's `--variant` selector.
# GLM-5.2 maps medium-equivalent to Z.ai's `high`; the other open models use
# `medium` as the portable thinking-on request and rely on the provider/default
# to clamp or ignore unsupported effort levels.
# (Duplicated across pi/opencode/codex so each adapter stays self-contained.)
OPEN_MODELS = {
    "glm-5.2":           {"provider": "zai",      "model_id": "glm-5.2",           "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "variant": "high"},
    "glm-4.7-flash":     {"provider": "zai",      "model_id": "glm-4.7-flash",     "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "variant": "medium"},
    "deepseek-v4-flash": {"provider": "deepseek", "model_id": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",     "env_key": "DEEPSEEK_API_KEY", "display": "DeepSeek",      "variant": "medium"},
    "kimi-k2.7-code":    {"provider": "moonshot", "model_id": "kimi-k2.7-code",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi", "variant": "medium"},
    "kimi-k3":    {"provider": "moonshot", "model_id": "kimi-k3",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi K3", "variant": "medium"},
    "laguna-s-2.1": {"provider": "openrouter", "model_id": "poolside/laguna-s-2.1", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Poolside Laguna S 2.1", "variant": "medium"},
    "inkling": {"provider": "openrouter", "model_id": "thinkingmachines/inkling", "base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY", "display": "OpenRouter Thinking Machines Inkling", "variant": "medium"},
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
    required=("provider", "variant"),
    defaults={"variant": "medium"},
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


def _open_config_content(spec):
    """Inline OPENCODE_CONFIG_CONTENT JSON registering the open provider."""
    prov = spec["provider"]
    return json.dumps({
        "provider": {
            prov: {
                "npm": "@ai-sdk/openai-compatible",
                "name": spec["display"],
                "options": {
                    "baseURL": _proxied_base_url(spec),
                    "apiKey": "{env:" + spec["env_key"] + "}",
                },
                "models": {spec["model_id"]: {}},
            }
        }
    })


def version():
    """Return the CLI version string (with binary path), or None on failure.

    Cheap `opencode --version`; never raises (the runner calls this defensively).
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


def _parse_json_with_usage(stdout):
    """Parse opencode's JSONL event stream into (tokens, turns, tail, usage).

    opencode reports visible output and reasoning separately; TOKEN_PARITY.md
    normalizes ``tokens_output`` to vendor completion tokens by adding them.
    A vendor-side hidden title/background call is not present in CLI JSONL, so
    this parser intentionally accounts only for reported ``step_finish`` events.
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

    turns = 0
    transcript = []
    token_usage = _empty_token_usage()
    usage_raw = []
    invariant_ok = True
    totals = {
        "tokens_input_uncached": 0,
        "tokens_cache_read": 0,
        "tokens_cache_write": 0,
        "tokens_output": 0,
        "tokens_reasoning": 0,
    }
    for ev in events:
        etype = ev.get("type")
        part = ev.get("part") or {}
        if etype == "step_finish":
            turns += 1
            tok = part.get("tokens") or {}
            cache = tok.get("cache") or {}
            inp = _num(tok.get("input"))
            visible_out = _num(tok.get("output"))
            reasoning = _num(tok.get("reasoning"))
            cache_read = _num(cache.get("read"))
            cache_write = _num(cache.get("write"))
            total = _num(tok.get("total"))
            if None in (inp, visible_out, reasoning, cache_read, cache_write):
                invariant_ok = False
                continue
            if total is None or inp + cache_read + cache_write + visible_out + reasoning != total:
                invariant_ok = False
            usage_raw.append(tok)
            totals["tokens_input_uncached"] += inp
            totals["tokens_cache_read"] += cache_read
            totals["tokens_cache_write"] += cache_write
            totals["tokens_output"] += visible_out + reasoning
            totals["tokens_reasoning"] += reasoning
        elif etype == "text":
            text = part.get("text")
            if text:
                transcript.append(text)
        elif etype == "tool_use":
            tool = part.get("tool")
            if tool:
                transcript.append(f"[tool: {tool}]")

    if usage_raw:
        token_usage.update(totals)
        token_usage["usage_raw"] = usage_raw
        token_usage["token_basis"] = "vendor_split" if invariant_ok else "estimated"

    tail = "\n".join(transcript)[-2000:]
    return _legacy_tokens(token_usage), (turns or None), tail, token_usage



def _parse_json(stdout):
    """Backward-compatible parser returning legacy fields only."""
    tokens, turns, tail, token_usage = _parse_json_with_usage(stdout)
    return tokens, turns, tail

def _isolated_env():
    """Return (environment, temp HOME) containing only opencode auth files."""
    iso_home = tempfile.mkdtemp(prefix="opencode_home_")
    env = dict(os.environ)
    env["HOME"] = iso_home
    env["XDG_CONFIG_HOME"] = os.path.join(iso_home, ".config")
    env["XDG_DATA_HOME"] = os.path.join(iso_home, ".local", "share")
    env["XDG_STATE_HOME"] = os.path.join(iso_home, ".local", "state")
    env["XDG_CACHE_HOME"] = os.path.join(iso_home, ".cache")
    # These variables can point directly at an owner's config outside HOME.
    for name in ("OPENCODE_CONFIG", "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG_CONTENT"):
        env.pop(name, None)
    for source in _AUTH_CANDIDATES:
        if not os.path.isfile(source):
            continue
        # Current releases use XDG_DATA_HOME/opencode/auth.json. Normalize old
        # auth locations there too; never copy adjacent config/state files.
        dest = os.path.join(env["XDG_DATA_HOME"], "opencode", "auth.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(source, dest)
        break
    return env, iso_home


def run(instruction: str, workdir: str, model: str, timeout_s: int) -> dict:
    auth_source = next((path for path in _AUTH_CANDIDATES if os.path.isfile(path)), None)
    env, iso_home = _isolated_env()
    if model in MODELS:
        if model == "claude-opus-4-8" and not _has_anthropic_oauth():
            shutil.rmtree(iso_home, ignore_errors=True)
            return {"completed": False,
                    "error": f"SETUP-NEEDED: run `opencode auth login -p anthropic` (missing {_ANTHROPIC_AUTH})",
                    "output_tail": "", "tokens": None, "turns": None, "cmd": None,
                    **_empty_token_usage()}
        cmd = [
            "opencode", "run",
            "--dir", workdir,
            "-m", MODELS[model],
            # xai rejects --variant with a server error (grok-4.5 has no
            # selectable effort); omit the flag when the map holds None.
            *(["--variant", _VARIANT[model]] if _VARIANT.get(model) else []),
            "--auto",
            "--format", "json",
            "--title", "openbench",
            instruction,
        ]
        env.pop("OPENAI_API_KEY", None)  # force subscription OAuth route
        if model == "claude-opus-4-8":
            env.pop("ANTHROPIC_API_KEY", None)  # force Anthropic OAuth route
    elif model in OPEN_MODELS:
        spec = OPEN_MODELS[model]
        if not os.environ.get(spec["env_key"]):
            shutil.rmtree(iso_home, ignore_errors=True)
            return _setup_needed(spec["env_key"], model)
        cmd = [
            "opencode", "run",
            "--dir", workdir,
            "-m", f'{spec["provider"]}/{spec["model_id"]}',
            "--variant", spec["variant"],
            "--auto",
            "--format", "json",
            "--title", "openbench",
            instruction,
        ]
        env["OPENCODE_CONFIG_CONTENT"] = _open_config_content(spec)
    else:
        shutil.rmtree(iso_home, ignore_errors=True)
        return _unsupported(model)

    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            full_output = _err_tail(e, limit=None)
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
    finally:
        if model in MODELS and auth_source is not None:
            isolated_auth = os.path.join(env["XDG_DATA_HOME"], "opencode", "auth.json")
            try_persist_auth_file(isolated_auth, auth_source)
        shutil.rmtree(iso_home, ignore_errors=True)

    combined = (proc.stdout or "") + (proc.stderr or "")
    try:
        tokens, turns, tail, token_usage = _parse_json_with_usage(proc.stdout or "")
    except Exception:  # noqa: BLE001 - never let usage parsing break a run
        tokens, turns, tail, token_usage = None, None, "", _empty_token_usage()
    if not tail:
        tail = combined[-2000:]

    return {
        "completed": proc.returncode == 0,
        "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
        "output_tail": tail,
        # Optional (ADAPTER_SPEC v1): full untruncated stdout+stderr for the
        # runner's local transcript. LOCAL-ONLY; never published unscrubbed.
        "full_output": combined,
        "tokens": tokens,
        "turns": turns,
        "cmd": cmd,
        **token_usage,
    }
