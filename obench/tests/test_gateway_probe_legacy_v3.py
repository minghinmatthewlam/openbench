import json
import shutil
import tempfile
import unittest
from pathlib import Path

from obench import gateway_probe_legacy_v3, gateway_probe_publish
from obench.gateway_probe_models import GatewayProbeRunError


GPT4O_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "gateway-probe"
    / "2026-07-27-gpt4o-mini-managed-30"
)
KIMI_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "gateway-probe"
    / "2026-07-28-kimi-k3-managed-100"
)


class LegacyGatewayProbeV3Tests(unittest.TestCase):
    def test_only_pinned_v3_archive_is_admitted(self):
        manifest = gateway_probe_legacy_v3.verify_bundle(
            GPT4O_BUNDLE,
        )

        self.assertEqual(manifest["result_schema_version"], 3)
        self.assertEqual(manifest["result_count"], 300)
        with self.assertRaisesRegex(
            GatewayProbeRunError,
            "manifest does not match schema",
        ):
            gateway_probe_publish.verify_bundle(GPT4O_BUNDLE)
        with tempfile.TemporaryDirectory() as tmp:
            unapproved = Path(tmp) / "other-v3-bundle"
            shutil.copytree(GPT4O_BUNDLE, unapproved)
            with self.assertRaisesRegex(
                GatewayProbeRunError,
                "identity is not approved",
            ):
                gateway_probe_legacy_v3.verify_bundle(unapproved)

    def test_current_v4_verifier_remains_independent(self):
        manifest = gateway_probe_publish.verify_bundle(KIMI_BUNDLE)
        self.assertEqual(manifest["result_schema_version"], 4)
        self.assertEqual(manifest["result_count"], 500)

    def test_pinned_archive_rejects_file_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "2026-07-27-gpt4o-mini-managed-30"
            shutil.copytree(GPT4O_BUNDLE, bundle)
            report_path = bundle / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["result_count"] = 0
            report_path.write_text(
                json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                GatewayProbeRunError,
                "artifact digest mismatch",
            ):
                gateway_probe_legacy_v3.verify_bundle(
                    bundle,
                )


if __name__ == "__main__":
    unittest.main()
