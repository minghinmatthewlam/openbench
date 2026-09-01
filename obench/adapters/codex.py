"""Adapter for the `codex` CLI (OpenAI Codex, ChatGPT-subscription login).

Headless invocation:
    CODEX_HOME=<isolated tmp with auth.json only> codex exec --json \
        --disable apps --disable plugins --disable multi_agent \
        --skip-git-repo-check -C <workdir> -s workspace-write \
        -m gpt-5.5 -c model_reasoning_effort="medium" <instruction>

Notes / quirks:
- `codex exec` is fully non-interactive; there are no approval prompts to
  suppress. `-s workspace-write` is the least-privileged sandbox that still
  lets the agent edit files inside the workspace root. In the docker lane
  (BENCH_IN_CONTAINER=1) it is replaced by
  `--dangerously-bypass-approvals-and-sandbox`: bwrap cannot nest inside the
  bench container, and the disposable container is the external sandbox.
- The runner hands us a disposable temp dir that is usually NOT a git repo,
  so `--skip-git-repo-check` is required or codex refuses to start.
- Reasoning effort is set via a config override, not the model string. The
  canonical "-medium" suffix is mapped to model_reasoning_effort.
- Copies only runtime `auth.json` into a fresh `CODEX_HOME`; personal config,
  instructions, skills, plugins, MCPs, rules, memories, and sessions are absent.
- `--json` emits a JSONL event stream. The final `turn.completed` event carries
  `usage={input_tokens,cached_input_tokens,output_tokens,reasoning_output_tokens}`.
  Token accounting emits TOKEN_PARITY.md split fields from the final aggregate:
    tokens_input_uncached = input_tokens - cached_input_tokens
    tokens_cache_read     = cached_input_tokens
    tokens_cache_write    = 0
    tokens_output         = output_tokens  # already reasoning-inclusive
    tokens_reasoning      = reasoning_output_tokens
    tokens                = tokens_input_uncached + tokens_output
    turns  = number of `turn.completed` events (model rounds).
  Human tail is synthesized from `agent_message`/`file_change` items.
  NOTE: `codex exec --json` only flushes/exits cleanly when driven as a plain
  captured subprocess (stdin=DEVNULL); piping it through a shell can stall it.
  Parsing is defensive: shape drift yields tokens=None/turns=None + raw tail.

Open models (M4): DeepSeek / Z.ai / Moonshot are chat-only, but codex 0.142
requires the Responses API, so they run through a host-side LiteLLM bridge (see
OPEN_MODELS and bench/openmodel_bridge.sh). Token accounting is UNCHANGED in
mechanism (same fresh-basis parse of `turn.completed.usage`), but the basis now
transits the bridge: the counts codex reports are the bridge's Responses-shaped
usage, remapped from the upstream chat-completions `usage` (prompt_tokens ->
input_tokens, completion_tokens -> output_tokens, split reasoning_tokens). This
matches the vendor's own billed usage; there is no codex-native usage for these
models to cross-check against.
"""

import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile

try:
    from obench.auth_persist import auth_file_lease, auth_lease_proves_path
except ImportError:  # file-path / Docker mount layout
    from auth_persist import auth_file_lease, auth_lease_proves_path

NAME = "codex"
_EXE = "codex"
_MULTI_AGENT_ENV = "OPENBENCH_CODEX_MULTI_AGENT"


def _feature_flags(env_override=None):
    """Keep stock runs OFF; only the checked-in candidate explicitly opts ON."""
    flags = ["--disable", "apps", "--disable", "plugins"]
    if env_override and env_override.get(_MULTI_AGENT_ENV) == "enabled":
        flags += ["--enable", "multi_agent"]
    else:
        flags += ["--disable", "multi_agent"]
    return flags


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

# canonical model name -> codex `-m` model string
MODELS = {
    "gpt-5.5-medium": "gpt-5.5",
    "gpt-5.6-sol": "gpt-5.6-sol",
    "gpt-5.6-terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
}

# canonical model name -> reasoning effort passed via `-c model_reasoning_effort`
_EFFORT = {
    "gpt-5.5-medium": "medium",
    "gpt-5.6-sol": "medium",
    "gpt-5.6-terra": "medium",
    "gpt-5.6-luna": "medium",
}

# canonical model name -> service tier override. GPT-5.6 Sol must stay on the
# normal/non-fast lane even if the operator's Codex config defaults to priority.
_SERVICE_TIER = {
    "gpt-5.6-sol": "default",
    "gpt-5.6-terra": "default",
    "gpt-5.6-luna": "default",
}

# --- M4 open models (first-party pay-per-token, chat-only vendors) -----------
# codex-cli >=0.142 REMOVED wire_api="chat"; custom providers must speak the
# Responses API. DeepSeek / Z.ai / Moonshot only serve /chat/completions, so
# codex cannot talk to them directly. We route through a host-side LiteLLM proxy
# (the "bridge", see bench/openmodel_bridge.sh) that accepts /v1/responses
# ingress and translates each call to /chat/completions upstream. codex is thus
# pointed at the bridge (wire_api stays "responses"); ``base_url`` is the bridge,
# NOT the vendor. The bridge maps model_id -> the vendor route + real key.
#
# BRIDGE REQUIREMENT: a human/orchestrator must start the bridge BEFORE any
# open-model codex run (`bench/openmodel_bridge.sh`, foreground). The runner does
# NOT manage it. run() does a cheap TCP probe first and returns SETUP-NEEDED if
# the bridge port is unreachable.
#
# env_key is still required (SETUP-NEEDED if unset): codex refuses to start a
# custom provider whose env_key names an unset variable, and it sends that value
# as the ingress bearer (the bridge ignores it and injects the vendor key from
# its own environment).
#
# Base URLs below are documentation/provenance only; the adapter talks to the
# bridge, which is configured with these same vendor endpoints.
#
# Thinking parity: the adapter requests `model_reasoning_effort="medium"` for
# every open model. The LiteLLM bridge hook normalizes that to the closest
# vendor thinking-on setting (GLM-5.2 medium -> Z.ai high; otherwise the
# vendor's thinking-on default when levels are not exposed on the bridge route).
# (Duplicated across the pi/opencode/codex adapters so each stays self-contained
#  under the runner's isolated importer.)
OPEN_MODELS = {
    # Thinking parity for the opus frontier lane: codex requests
    # `model_reasoning_effort="medium"`; the LiteLLM bridge preserves that as
    # Anthropic medium reasoning while injecting ANTHROPIC_API_KEY upstream.
    "claude-opus-4-8":   {"provider": "anthropic", "model_id": "claude-opus-4-8",   "base_url": "https://api.anthropic.com",     "env_key": "ANTHROPIC_API_KEY", "display": "Anthropic Claude", "effort": "medium"},
    "glm-5.2":           {"provider": "zai",      "model_id": "glm-5.2",           "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "effort": "medium"},
    "glm-4.7-flash":     {"provider": "zai",      "model_id": "glm-4.7-flash",     "base_url": "https://api.z.ai/api/paas/v4", "env_key": "ZAI_API_KEY",      "display": "Z.ai GLM",      "effort": "medium"},
    "deepseek-v4-flash": {"provider": "deepseek", "model_id": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",     "env_key": "DEEPSEEK_API_KEY", "display": "DeepSeek",      "effort": "medium"},
    "kimi-k2.7-code":    {"provider": "moonshot", "model_id": "kimi-k2.7-code",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi", "effort": "medium"},
    "kimi-k3":    {"provider": "moonshot", "model_id": "kimi-k3",    "base_url": "https://api.moonshot.ai/v1",   "env_key": "MOONSHOT_API_KEY", "display": "Moonshot Kimi K3", "effort": "medium"},
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
    required=("provider", "effort"),
    defaults={"effort": "medium"},
)

# Host-side bridge (LiteLLM proxy). Port must match bench/openmodel_bridge.sh
# (both default to 4141; override in lockstep via BENCH_BRIDGE_PORT).
_BRIDGE_DEFAULT_PORT = 4141
_KEYS_ENV = os.path.expanduser("~/.openbench/keys.env")


def _proxy_cell_url(*parts):
    base = os.environ.get("OPENBENCH_PROXY_BASE_URL")
    token = os.environ.get("OPENBENCH_PROXY_CELL_TOKEN")
    if not os.environ.get("OPENBENCH_PROXY") or not base or not token:
        return None
    path = "/".join(str(p).strip("/") for p in ("cell", token, *parts) if str(p).strip("/"))
    return base.rstrip("/") + "/" + path


def _bridge_host():
    """Hostname the bridge is reachable at from the current lane.

    In the docker lane entry.py sets BENCH_IN_CONTAINER; Docker Desktop resolves
    ``host.docker.internal`` to the host running the bridge. On the host lane it
    is plain ``localhost``.
    """
    return "host.docker.internal" if os.environ.get("BENCH_IN_CONTAINER") else "localhost"


def _bridge_port():
    return int(os.environ.get("BENCH_BRIDGE_PORT", _BRIDGE_DEFAULT_PORT))


def _bridge_base_url():
    """codex ``base_url`` for the bridge; codex appends ``/responses`` to it.

    When the counting proxy is active, route through its ``bridge`` prefix
    (proxy -> LiteLLM -> vendor) so open-model cells get proxy-metered usage.
    """
    proxied = _proxy_cell_url("bridge", "v1")
    if proxied:
        return proxied
    return f"http://{_bridge_host()}:{_bridge_port()}/v1"


def _bridge_reachable(timeout=3.0):
    """Cheap TCP connect probe to the bridge port. True iff something accepts."""
    try:
        with socket.create_connection((_bridge_host(), _bridge_port()), timeout=timeout):
            return True
    except OSError:
        return False


def _bridge_down(model):
    return {"completed": False,
            "error": (f"SETUP-NEEDED: open-model bridge unreachable at "
                      f"{_bridge_host()}:{_bridge_port()} for {model} "
                      f"(start it: bench/openmodel_bridge.sh)"),
            "output_tail": "", "tokens": None, "turns": None, "cmd": None,
            **_empty_token_usage()}


def _keys_env_has(env_key):
    try:
        with open(_KEYS_ENV, encoding="utf-8") as fh:
            for raw in fh:
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
    except OSError:
        return False
    return False


def _host_has_key(env_key):
    return bool(os.environ.get(env_key) or _keys_env_has(env_key))


def _codex_env_for_bridge(env_key):
    # The bridge injects the real upstream key from its host process. Give codex
    # only a non-secret placeholder so its shell-capable agent cannot read API
    # credentials from the environment.
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env[env_key] = "openbench-bridge-placeholder"
    return env


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


def version():
    """Return the CLI version string (with binary path), or None on failure.

    Cheap `codex --version`; never raises (the runner calls this defensively).
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
    """Parse codex's JSONL event stream into (tokens, turns, tail, usage).

    Codex's final aggregate ``input_tokens`` is cache-inclusive and
    ``output_tokens`` is reasoning-inclusive. Reasoning is recorded separately
    as a subset and must not be added to the legacy scalar.
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

    token_usage = _empty_token_usage()
    usage_raw = []
    last_usage = None
    turns = 0
    transcript = []
    for ev in events:
        etype = ev.get("type")
        if etype == "turn.completed":
            turns += 1
            usage = ev.get("usage") or {}
            if isinstance(usage, dict):
                usage_raw.append(usage)
                last_usage = usage
        elif etype == "item.completed":
            item = ev.get("item") or {}
            itype = item.get("type")
            if itype == "agent_message":
                text = item.get("text")
                if text:
                    transcript.append(text)
            elif itype == "file_change":
                names = [os.path.basename(c.get("path", ""))
                         for c in (item.get("changes") or [])
                         if isinstance(c, dict) and c.get("path")]
                transcript.append(f"[file_change: {', '.join(n for n in names if n)}]")
            elif itype == "command_execution":
                transcript.append("[command]")

    if last_usage is not None:
        inp = _num(last_usage.get("input_tokens"))
        cached = _num(last_usage.get("cached_input_tokens"))
        cache_write = _num(
            last_usage.get("cache_write_tokens")
            or last_usage.get("cache_creation_input_tokens")
            or last_usage.get("cache_creation_tokens")
            or 0
        )
        out = _num(last_usage.get("output_tokens"))
        reasoning = _num(last_usage.get("reasoning_output_tokens"))
        invariant_ok = None not in (inp, cached, cache_write, out, reasoning)
        if invariant_ok and (cached + cache_write > inp or reasoning > out):
            invariant_ok = False
        if invariant_ok:
            token_usage.update({
                "tokens_input_uncached": inp - cached - cache_write,
                "tokens_cache_read": cached,
                "tokens_cache_write": cache_write,
                "tokens_output": out,
                "tokens_reasoning": reasoning,
            })
        token_usage["usage_raw"] = last_usage
        token_usage["token_basis"] = "vendor_split" if invariant_ok else "estimated"

    tail = "\n".join(transcript)[-2000:]
    return _legacy_tokens(token_usage), (turns or None), tail, token_usage



def _parse_json(stdout):
    """Backward-compatible parser returning legacy fields only."""
    tokens, turns, tail, token_usage = _parse_json_with_usage(stdout)
    return tokens, turns, tail

def run(
    instruction: str,
    workdir: str,
    model: str,
    timeout_s: int,
    env_override=None,
    auth_lease_proofs=(),
) -> dict:
    if os.environ.get("BENCH_IN_CONTAINER"):
        # codex's own sandbox (bwrap) needs user namespaces and cannot nest
        # inside the bench container; the disposable container IS the external
        # sandbox, which is the documented intent of this flag.
        sandbox = ["--dangerously-bypass-approvals-and-sandbox"]
    else:
        sandbox = ["-s", "workspace-write"]
    base = [
        "codex", "exec",
        "--json",
    ] + _feature_flags(env_override) + [
        "--skip-git-repo-check",
        "-C", workdir,
    ] + sandbox
    if model in MODELS:
        cmd = base + [
            "-m", MODELS[model],
            "-c", f'model_reasoning_effort="{_EFFORT[model]}"',
        ]
        if model in _SERVICE_TIER:
            cmd += ["-c", f'service_tier="{_SERVICE_TIER[model]}"']
        proxy_url = _proxy_cell_url("codex", "backend-api", "codex")
        if proxy_url:
            cmd += ["-c", f'openai_base_url="{proxy_url}"']
        cmd += [instruction]
    elif model in OPEN_MODELS:
        spec = OPEN_MODELS[model]
        if not _host_has_key(spec["env_key"]):
            return _setup_needed(spec["env_key"], model)
        # Route through the host-side Responses<->Chat bridge (see the OPEN_MODELS
        # docstring and bench/openmodel_bridge.sh). Fail fast with a clear
        # SETUP-NEEDED error if the bridge isn't up, rather than letting codex
        # spend a turn's timeout failing to connect.
        if not _bridge_reachable():
            return _bridge_down(model)
        prov = spec["provider"]
        cmd = base + [
            "-c", f'model_providers.{prov}.name="{spec["display"]}"',
            "-c", f'model_providers.{prov}.base_url="{_bridge_base_url()}"',
            "-c", f'model_providers.{prov}.env_key="{spec["env_key"]}"',
            "-c", f'model_providers.{prov}.wire_api="responses"',
            "-c", f'model_provider="{prov}"',
            "-c", f'model_reasoning_effort="{spec["effort"]}"',
            "-m", spec["model_id"],
            instruction,
        ]
    else:
        return _unsupported(model)

    child_env = _codex_env_for_bridge(spec["env_key"]) if model in OPEN_MODELS else os.environ.copy()
    if env_override:
        child_env.update(env_override)
        # This is an adapter control, not child-process configuration.
        child_env.pop(_MULTI_AGENT_ENV, None)

    # Stock runs get a fresh CODEX_HOME containing authentication only.  In
    # particular, never copy config.toml, AGENTS.md, skills, MCP definitions,
    # rules, memories, sessions, or plugins from the machine owner. Ablation
    # adapters supply their own already-composed CODEX_HOME via env_override.
    isolated_home = None
    auth_src = None
    auth_copy = None
    auth_lease = None
    provided_codex_home = (
        env_override.get("CODEX_HOME") if env_override else None
    )
    if provided_codex_home:
        auth_src = os.path.join(
            os.path.expanduser(provided_codex_home), "auth.json"
        )
        if (os.path.isfile(auth_src)
                and not auth_lease_proves_path(
                    auth_lease_proofs, auth_src
                )):
            auth_lease = auth_file_lease(auth_src).__enter__()
    else:
        isolated_home = tempfile.mkdtemp(prefix="codex_home_")
        auth_root = os.path.expanduser(os.environ.get("CODEX_HOME") or "~/.codex")
        auth_src = os.path.join(auth_root, "auth.json")
        if os.path.isfile(auth_src):
            try:
                auth_lease = auth_file_lease(auth_src).__enter__()
                auth_copy = os.path.join(isolated_home, "auth.json")
                auth_lease.stage(auth_copy)
            except BaseException:
                if auth_lease is not None:
                    auth_lease.__exit__(None, None, None)
                shutil.rmtree(isolated_home, ignore_errors=True)
                raise
        child_env["CODEX_HOME"] = isolated_home

    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                env=child_env,
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
        try:
            if auth_copy and auth_lease:
                auth_lease.try_persist(auth_copy)
        finally:
            if auth_lease:
                auth_lease.__exit__(None, None, None)
            if isolated_home:
                shutil.rmtree(isolated_home, ignore_errors=True)

    combined = (proc.stdout or "") + (proc.stderr or "")
    try:
        tokens, turns, tail, token_usage = _parse_json_with_usage(proc.stdout or "")
    except Exception:  # noqa: BLE001 - never let usage parsing break a run
        tokens, turns, tail, token_usage = None, None, "", _empty_token_usage()
    if not tail:
        tail = combined[-2000:]

    if model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna") and token_usage.get("token_basis") == "vendor_split":
        raw = token_usage.get("usage_raw") or {}
        if not any(k in raw for k in ("cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens")):
            # GPT-5.6 may expose billable cache writes on newer Codex event
            # schemas. If this CLI omits the field, keep the legacy fresh-ish
            # scalar usable for the smoke contract but do not assert complete
            # split parity: cache writes are unknown and the uncached lane may
            # include writes depending on Codex's aggregate input semantics.
            token_usage["tokens_cache_write"] = None
            token_usage["token_basis"] = "estimated"

    return {
        "completed": proc.returncode == 0,
        "error": None if proc.returncode == 0 else f"exit {proc.returncode}",
        "output_tail": tail,
        # Optional (ADAPTER_SPEC v1): full untruncated stdout+stderr so the
        # runner can persist a complete local transcript. Cheap here (already
        # concatenated). LOCAL-ONLY: transcripts are never published unscrubbed.
        "full_output": combined,
        "tokens": tokens,
        "turns": turns,
        "cmd": cmd,
        **token_usage,
    }
