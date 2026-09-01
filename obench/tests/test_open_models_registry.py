#!/usr/bin/env python3
"""Tests for the optional file-backed OPEN_MODELS registry."""

import importlib.util
import os
import shutil
import tempfile
import unittest

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTERS_DIR = os.path.join(BENCH_DIR, "adapters")


def _load_helper():
    path = os.path.join(ADAPTERS_DIR, "_open_models.py")
    spec = importlib.util.spec_from_file_location("test_open_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


om = _load_helper()

BUILTIN = {
    "glm-5.2": {
        "provider": "zai",
        "model_id": "glm-5.2",
        "base_url": "https://api.z.ai/api/paas/v4",
        "env_key": "ZAI_API_KEY",
        "display": "Z.ai GLM",
        "effort": "medium",
        "compat": {"supportsStore": False, "thinkingFormat": "zai"},
    },
}

SHARED = (
    '[models.qwen3-coder]\n'
    'provider = "openrouter"\n'
    'model_id = "qwen/qwen3-coder"\n'
    'base_url = "https://openrouter.ai/api/v1"\n'
    'env_key = "OPENROUTER_API_KEY"\n'
    'display = "OpenRouter Qwen3 Coder"\n'
)


class RegistryFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="obench_open_models_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write(self, text, name="open_models.toml"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def load(self, path, adapter="codex", **kw):
        kw.setdefault("required", ("provider", "effort"))
        kw.setdefault("defaults", {"effort": "medium"})
        return om.load(adapter, BUILTIN, environ={om.ENV_VAR: path}, **kw)

    def test_no_registry_returns_builtin_unchanged(self):
        models, source = om.load("codex", BUILTIN, environ={om.ENV_VAR: ""})
        self.assertIsNone(source)
        self.assertEqual(models, BUILTIN)
        self.assertIsNot(models, BUILTIN)

    def test_adds_model_and_fills_adapter_default(self):
        path = self.write(SHARED)
        models, source = self.load(path)
        self.assertEqual(source, path)
        self.assertEqual(models["qwen3-coder"]["model_id"], "qwen/qwen3-coder")
        self.assertEqual(models["qwen3-coder"]["effort"], "medium")
        self.assertIn("glm-5.2", models)

    def test_adapter_subtable_wins_over_shared(self):
        path = self.write(SHARED + '\n[models.qwen3-coder.codex]\neffort = "high"\n')
        models, _ = self.load(path)
        self.assertEqual(models["qwen3-coder"]["effort"], "high")

    def test_other_adapters_subtable_is_ignored(self):
        path = self.write(SHARED + '\n[models.qwen3-coder.opencode]\nvariant = "high"\n')
        models, _ = self.load(path)
        self.assertNotIn("variant", models["qwen3-coder"])
        self.assertEqual(models["qwen3-coder"]["effort"], "medium")

    def test_partial_override_keeps_rest_of_builtin_entry(self):
        path = self.write('[models."glm-5.2"]\nmodel_id = "glm-5.2-airx"\n')
        models, _ = self.load(path)
        entry = models["glm-5.2"]
        self.assertEqual(entry["model_id"], "glm-5.2-airx")
        self.assertEqual(entry["env_key"], "ZAI_API_KEY")
        self.assertEqual(entry["compat"]["thinkingFormat"], "zai")

    def test_nested_table_override_merges_not_replaces(self):
        path = self.write(
            '[models."glm-5.2".compat]\nsupportsStore = true\n')
        models, _ = self.load(path)
        compat = models["glm-5.2"]["compat"]
        self.assertTrue(compat["supportsStore"])
        self.assertEqual(compat["thinkingFormat"], "zai")

    def test_builtin_dict_is_not_mutated(self):
        path = self.write('[models."glm-5.2"]\nmodel_id = "glm-5.2-airx"\n')
        self.load(path)
        self.assertEqual(BUILTIN["glm-5.2"]["model_id"], "glm-5.2")

    def test_derive_fills_field_from_another(self):
        path = self.write(SHARED)
        models, _ = self.load(
            path, adapter="grokbuild", required=("proxy_route",), defaults={},
            derive=lambda e: dict(e, proxy_route="chat/%s" % e["provider"])
            if not e.get("proxy_route") and e.get("provider") else e)
        self.assertEqual(models["qwen3-coder"]["proxy_route"], "chat/openrouter")

    def test_missing_required_field_names_the_model(self):
        path = self.write('[models.broken]\nmodel_id = "x"\n')
        with self.assertRaises(om.RegistryError) as ctx:
            self.load(path)
        msg = str(ctx.exception)
        self.assertIn("[models.broken]", msg)
        self.assertIn("base_url", msg)

    def test_unquoted_dotted_name_explains_the_quoting_rule(self):
        path = self.write('[models.glm-5.2]\nmodel_id = "x"\n')
        with self.assertRaises(om.RegistryError) as ctx:
            self.load(path)
        self.assertIn('[models."glm-5.2"]', str(ctx.exception))

    def test_unknown_top_level_table_is_rejected(self):
        path = self.write('[routes]\nx = 1\n')
        with self.assertRaises(om.RegistryError) as ctx:
            self.load(path)
        self.assertIn("unknown top-level table", str(ctx.exception))

    def test_invalid_toml_is_rejected(self):
        path = self.write("[models.x\n")
        with self.assertRaises(om.RegistryError) as ctx:
            self.load(path)
        self.assertIn("invalid TOML", str(ctx.exception))

    def test_env_var_pointing_at_missing_file_is_an_error(self):
        with self.assertRaises(om.RegistryError):
            om.find_registry(environ={om.ENV_VAR: os.path.join(self.tmp, "nope.toml")})

    def test_project_dir_beats_home_and_walks_up(self):
        root = os.path.join(self.tmp, "proj")
        nested = os.path.join(root, "src", "app")
        os.makedirs(nested)
        os.makedirs(os.path.join(root, ".openbench"))
        want = os.path.join(root, ".openbench", "open_models.toml")
        with open(want, "w", encoding="utf-8") as fh:
            fh.write(SHARED)
        self.assertEqual(om.find_registry(start=nested, environ={}), want)

    def test_no_registry_anywhere_returns_none(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        found = om.find_registry(start=empty, environ={})
        home = os.path.join(os.path.expanduser("~"), ".openbench", "open_models.toml")
        self.assertIn(found, (None, home))


class AdapterWiringTests(unittest.TestCase):
    """Every adapter with an OPEN_MODELS dict exposes its registry source."""

    ADAPTERS = ("codex", "opencode", "pi", "grokbuild")

    def _import(self, name):
        path = os.path.join(ADAPTERS_DIR, f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"wiring_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_adapters_expose_source_and_builtin_defaults(self):
        for name in self.ADAPTERS:
            with self.subTest(adapter=name):
                mod = self._import(name)
                self.assertTrue(hasattr(mod, "OPEN_MODELS_SOURCE"))
                self.assertIsInstance(mod.OPEN_MODELS, dict)
                self.assertIn("kimi-k3", mod.OPEN_MODELS)

    def test_every_builtin_entry_has_the_base_fields(self):
        for name in self.ADAPTERS:
            mod = self._import(name)
            for model, spec in mod.OPEN_MODELS.items():
                with self.subTest(adapter=name, model=model):
                    for field in om.BASE_REQUIRED:
                        self.assertTrue(spec.get(field), f"{field} missing")


if __name__ == "__main__":
    unittest.main()
