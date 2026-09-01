"""Optional file-backed registry for adapter ``OPEN_MODELS`` entries.

Each adapter keeps its in-code ``OPEN_MODELS`` dict as the default. This helper
merges an optional TOML registry over that default so a user can add or retune
an open/BYO route without editing adapter code and carrying a diff against
upstream.

Search order, first hit wins:

1. ``$OPENBENCH_OPEN_MODELS`` (explicit path; set it empty to disable lookup)
2. ``.openbench/open_models.toml``, walking up from the working directory
3. ``~/.openbench/open_models.toml``

Schema. Shared fields sit on the model table, adapter-specific ones in a
subtable named for the adapter:

    [models.qwen3-coder]
    provider  = "openrouter"
    model_id  = "qwen/qwen3-coder"
    base_url  = "https://openrouter.ai/api/v1"
    env_key   = "OPENROUTER_API_KEY"
    display   = "OpenRouter Qwen3 Coder"

    [models.qwen3-coder.grokbuild]
    proxy_route = "chat/openrouter"

A model name containing a dot must be quoted, or TOML reads it as a dotted
key: ``[models."glm-5.2"]``, not ``[models.glm-5.2]``. Most of the built-in
names have a dot in them.

A model table with no subtable for the current adapter still applies; the
adapter's ``defaults`` fill in its own fields. Overriding one field of a
built-in model leaves the rest of that entry intact, nested dicts included.

Loaded by ``spec_from_file_location`` from each adapter, so this module is
stdlib-only and imports nothing from ``obench``.
"""

import os
import tomllib

ENV_VAR = "OPENBENCH_OPEN_MODELS"
CONFIG_DIRNAME = ".openbench"
CONFIG_FILENAME = "open_models.toml"

# Fields every entry needs regardless of which adapter reads it.
BASE_REQUIRED = ("model_id", "base_url", "env_key", "display")


class RegistryError(Exception):
    """Malformed registry file. Raised at import time, never swallowed."""


def find_registry(start=None, environ=None):
    """Return the registry path to use, or ``None`` when there is none."""
    env = os.environ if environ is None else environ
    if ENV_VAR in env:
        explicit = env[ENV_VAR].strip()
        if not explicit:
            return None
        if not os.path.isfile(explicit):
            raise RegistryError(f"{ENV_VAR}={explicit!r} is not a file")
        return os.path.abspath(explicit)

    cur = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(cur, CONFIG_DIRNAME, CONFIG_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    home = os.path.join(os.path.expanduser("~"), CONFIG_DIRNAME, CONFIG_FILENAME)
    return home if os.path.isfile(home) else None


def _read(path):
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except OSError as exc:
        raise RegistryError(f"{path}: cannot read ({exc})") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"{path}: invalid TOML ({exc})") from exc

    unknown = sorted(set(data) - {"models"})
    if unknown:
        raise RegistryError(f"{path}: unknown top-level table(s) {unknown}; expected [models]")
    models = data.get("models", {})
    if not isinstance(models, dict):
        raise RegistryError(f"{path}: [models] must be a table")
    return models


def _dotted_key_hint(name, table):
    """Explain the ``[models.glm-5.2]`` trap when a name got split on its dot."""
    if not isinstance(table, dict):
        return ""
    for key, value in table.items():
        if isinstance(value, dict) and key[:1].isdigit():
            return (f'; a model name containing "." must be quoted, e.g. '
                    f'[models."{name}.{key}"]')
    return ""


def _merge(base, overlay):
    """Overlay onto a copy of base, recursing into nested tables."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _entry_for(adapter, name, table, path):
    """Split a model table into shared fields plus this adapter's subtable."""
    if not isinstance(table, dict):
        raise RegistryError(f"{path}: [models.{name}] must be a table")

    shared = {}
    mine = {}
    for key, value in table.items():
        # A subtable named for some other adapter is not ours to read. Only
        # treat a nested table as adapter-scoped when its key has no meaning as
        # a shared field, which is why known nested fields (compat,
        # thinkingLevelMap) are passed through as shared.
        if key == adapter:
            if not isinstance(value, dict):
                raise RegistryError(
                    f"{path}: [models.{name}.{adapter}] must be a table")
            mine = value
        elif isinstance(value, dict) and key in _KNOWN_ADAPTERS and key != adapter:
            continue
        else:
            shared[key] = value
    return _merge(shared, mine)


# Adapter names whose subtables must be ignored when loading a different
# adapter. Kept explicit so an unrecognised nested key is surfaced as a shared
# field rather than silently dropped.
_KNOWN_ADAPTERS = frozenset({
    "codex", "codex_v1", "codex_v2", "opencode", "pi", "grokbuild",
    "claude", "cursor", "devin",
})


def load(adapter, builtin, required=(), defaults=None, derive=None,
         start=None, environ=None):
    """Return ``(models, source)`` for ``adapter``.

    ``builtin`` is returned unchanged when no registry file is found, so the
    in-code dict stays the default. ``required`` names adapter-specific fields
    that a brand-new model must end up with; ``defaults`` supplies them when the
    registry does not, and ``derive`` gets a last pass to fill a field that
    follows from the others.
    """
    path = find_registry(start=start, environ=environ)
    if path is None:
        return dict(builtin), None

    tables = _read(path)
    merged = {name: dict(spec) for name, spec in builtin.items()}
    needed = tuple(BASE_REQUIRED) + tuple(required)

    for name, table in tables.items():
        entry = _entry_for(adapter, name, table, path)
        if name in merged:
            entry = _merge(merged[name], entry)
        else:
            entry = _merge(dict(defaults or {}), entry)
        if derive is not None:
            entry = derive(entry)
        missing = [key for key in needed if not entry.get(key)]
        if missing:
            raise RegistryError(
                f"{path}: [models.{name}] is missing {missing} for adapter "
                f"{adapter!r}{_dotted_key_hint(name, table)}")
        merged[name] = entry

    return merged, path
