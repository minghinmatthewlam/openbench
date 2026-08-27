import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from obench import (
    gateway_cli,
    gateway_probe_cli,
    gateway_probe_results,
    gateway_probe_run,
    gateway_probe_spec,
)
from obench.gateway_probe_models import RunSummary
from obench.tests.test_gateway_probe_report import row
from obench.tests.test_gateway_probe_publish import (
    TEST_COMMIT,
    build_private_run,
)
from obench.tests.test_gateway_probe_spec import manifest


class GatewayProbeCliTests(unittest.TestCase):
    def test_gateway_dispatches_probe_publish_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = build_private_run(tmp)
            bundle = Path(tmp, "public")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                publish_code = gateway_cli.main([
                    "probe",
                    "publish",
                    str(run_dir),
                    str(bundle),
                    "--verified-with-commit",
                    TEST_COMMIT,
                ])
                verify_code = gateway_cli.main([
                    "probe",
                    "verify",
                    str(bundle),
                ])
        self.assertEqual(publish_code, 0)
        self.assertEqual(verify_code, 0)
        self.assertIn("blocks=cold:2/2,warm:2/2", stdout.getvalue())
        self.assertNotIn("exploratory", stdout.getvalue())
        self.assertNotIn("confirmatory", stdout.getvalue())

    def test_checked_in_examples_validate_through_cli(self):
        examples = Path(__file__).parents[1] / "examples"
        cases = (
            ("gateway-probe-responses.toml", "arms=2"),
            ("gateway-probe-five-way-responses.toml", "arms=5"),
            ("gateway-probe-kimi-k3-five-way-chat.toml", "arms=5"),
            ("gateway-probe-glm-5.3-flash-four-way-chat.toml", "arms=4"),
        )
        for filename, expected in cases:
            with self.subTest(filename=filename):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = gateway_probe_cli.main([
                        "validate",
                        str(examples / filename),
                    ])
                self.assertEqual(code, 0)
                self.assertIn(expected, stdout.getvalue())

    def test_gateway_dispatches_nested_probe_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            spec.write_text(manifest(), encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = gateway_cli.main(["probe", "validate", str(spec)])
        self.assertEqual(code, 0)
        self.assertIn("valid probe=probe-test", stdout.getvalue())

    def test_doctor_is_offline_and_fails_closed_for_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            spec.write_text(manifest(), encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = gateway_probe_cli.main(["doctor", str(spec)])
        self.assertEqual(code, 2)
        self.assertIn('"live_requests": false', stdout.getvalue())
        self.assertIn("OPENAI_API_KEY", stdout.getvalue())

    def test_run_dispatches_without_live_request_when_runner_is_mocked(self):
        summary = RunSummary(Path("out.jsonl"), 4, 2, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            spec.write_text(manifest(), encoding="utf-8")
            with mock.patch.object(
                gateway_probe_run, "run_experiment", return_value=summary
            ) as run:
                code = gateway_probe_cli.main([
                    "run",
                    str(spec),
                    "--results",
                    "out.jsonl",
                    "--max-blocks",
                    "3",
                    "--allow-cost-unavailable-block-recovery",
                ])
        self.assertEqual(code, 0)
        run.assert_called_once_with(
            str(spec),
            results_path="out.jsonl",
            force=False,
            allow_cost_unavailable_block_recovery=True,
            max_blocks=3,
        )

    def test_run_and_benchmark_reject_invalid_max_blocks(self):
        for command in ("run", "benchmark"):
            for value in ("0", "-1", "not-an-integer"):
                with self.subTest(command=command, value=value):
                    stderr = io.StringIO()
                    with (
                        contextlib.redirect_stderr(stderr),
                        self.assertRaises(SystemExit) as raised,
                    ):
                        gateway_probe_cli.main([
                            command,
                            "probe.toml",
                            "--max-blocks",
                            value,
                        ])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(
                        "must be a positive integer",
                        stderr.getvalue(),
                    )

    def test_malformed_prices_and_missing_credentials_exit_two_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            spec.write_text(manifest(), encoding="utf-8")
            cases = [
                (
                    ["doctor", str(spec)],
                    {gateway_probe_run.FROZEN_PRICES_ENV: "{bad-json"},
                    "not valid JSON",
                ),
                (
                    ["run", str(spec), "--results", str(Path(tmp, "out.jsonl"))],
                    {
                        gateway_probe_run.FROZEN_PRICES_ENV: json.dumps({
                            "openai/gpt-4o-mini": {
                                "input_per_million": "1",
                                "output_per_million": "2",
                                "effective_at": "2026-07-25T00:00:00Z",
                            }
                        })
                    },
                    "missing or empty",
                ),
            ]
            for argv, environ, expected in cases:
                with self.subTest(command=argv[0]):
                    stderr = io.StringIO()
                    with mock.patch.dict("os.environ", environ, clear=True):
                        with contextlib.redirect_stderr(stderr):
                            code = gateway_probe_cli.main(argv)
                    self.assertEqual(code, 2)
                    self.assertIn(expected, stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())

    def test_report_cli_renders_publishable_tables(self):
        rows = [
            row("direct", "cold", 1, baseline=True),
            row("gateway", "cold", 1, total=1.5),
            row("direct", "warm", 1, baseline=True),
            row("gateway", "warm", 1, total=1.5),
        ]
        for item in rows:
            item["scheduled_blocks_per_condition"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp, "probe.jsonl")
            results.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = gateway_probe_cli.main(["report", str(results)])
        self.assertEqual(code, 0)
        self.assertIn("Gateway Probe\nblocks cold=1/1 warm=1/1", stdout.getvalue())
        self.assertNotIn("exploratory", stdout.getvalue())
        self.assertNotIn("confirmatory", stdout.getvalue())
        self.assertIn("| Arm | Condition |", stdout.getvalue())
        self.assertIn("| Gateway | Condition | Phase metric |", stdout.getvalue())

    def test_benchmark_writes_self_contained_resumable_bundle(self):
        prices = json.dumps({
            "openai/gpt-4o-mini": {
                "input_per_million": "1",
                "output_per_million": "2",
                "effective_at": "2026-07-25T00:00:00Z",
            }
        })
        environ = {
            "OPENAI_API_KEY": "direct-secret",
            "OPENROUTER_API_KEY": "gateway-secret",
            gateway_probe_run.FROZEN_PRICES_ENV: prices,
        }

        def fake_run(_experiment, *, results_path, **_kwargs):
            experiment = gateway_probe_spec.load_experiment(_experiment)
            arms = {arm.arm_id: arm for arm in experiment.arms}
            rows = [
                row("direct", "cold", 1, baseline=True),
                row("gateway", "cold", 1),
                row("direct", "warm", 1, baseline=True),
                row("gateway", "warm", 1),
            ]
            for item in rows:
                item["scheduled_blocks_per_condition"] = 1
                item["identity"]["experiment"] = {
                    "id": experiment.experiment_id,
                    "digest": experiment.digest,
                }
                arm_id = item["identity"]["arm"]["id"]
                item["identity"]["arm"]["digest"] = arms[arm_id].digest
                item["model_match"] = experiment.model_match
                item["cell_id"] = gateway_probe_results.cell_id(
                    item["identity"]
                )
            path = Path(results_path)
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
                encoding="utf-8",
            )
            return RunSummary(path, 4, 2, 0, 0)

        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            output = Path(tmp, "bundle")
            source = manifest()
            spec.write_text(source, encoding="utf-8")
            with mock.patch.object(
                gateway_probe_run,
                "run_experiment",
                side_effect=fake_run,
            ) as run:
                with mock.patch.dict(os.environ, environ, clear=True):
                    code = gateway_probe_cli.main([
                        "benchmark",
                        str(spec),
                        "--output-dir",
                        str(output),
                    ])
                resume_environ = {
                    "OPENAI_API_KEY": "direct-secret",
                    "OPENROUTER_API_KEY": "gateway-secret",
                }
                with mock.patch.dict(
                    os.environ,
                    resume_environ,
                    clear=True,
                ):
                    resumed_code = gateway_probe_cli.main([
                        "benchmark",
                        str(spec),
                        "--output-dir",
                        str(output),
                        "--max-blocks",
                        "1",
                        "--allow-cost-unavailable-block-recovery",
                    ])
            self.assertEqual(code, 0)
            self.assertEqual(resumed_code, 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "experiment.toml",
                    "prices.json",
                    "results.jsonl",
                    "report.md",
                    "report.json",
                    "manifest.json",
                },
            )
            self.assertEqual((output / "experiment.toml").read_text(), source)
            bundle_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.iterdir()
            )
            self.assertNotIn("direct-secret", bundle_text)
            self.assertNotIn("gateway-secret", bundle_text)
            self.assertIn(
                "Request to response headers",
                (output / "report.md").read_text(),
            )
            manifest_data = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                manifest_data["result_schema_version"],
                gateway_probe_results.RESULT_SCHEMA_VERSION,
            )
            self.assertEqual(set(manifest_data["files"]), {
                "experiment.toml",
                "prices.json",
                "results.jsonl",
                "report.md",
                "report.json",
            })
            self.assertTrue(
                Path(run.call_args.args[0]).samefile(
                    output / "experiment.toml"
                )
            )
            self.assertEqual(run.call_count, 2)
            self.assertFalse(
                run.call_args_list[0].kwargs[
                    "allow_cost_unavailable_block_recovery"
                ]
            )
            self.assertTrue(
                run.call_args_list[1].kwargs[
                    "allow_cost_unavailable_block_recovery"
                ]
            )
            self.assertIsNone(run.call_args_list[0].kwargs["max_blocks"])
            self.assertEqual(run.call_args_list[1].kwargs["max_blocks"], 1)
            self.assertIn(
                gateway_probe_run.FROZEN_PRICES_ENV,
                run.call_args.kwargs["environ"],
            )
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, environ, clear=True),
                mock.patch.object(
                    gateway_probe_run,
                    "run_experiment",
                    side_effect=OSError("simulated run failure"),
                ),
                contextlib.redirect_stderr(stderr),
            ):
                failed_code = gateway_probe_cli.main([
                    "benchmark",
                    str(spec),
                    "--output-dir",
                    str(output),
                ])
            self.assertEqual(failed_code, 2)
            self.assertTrue((output / "results.jsonl").exists())
            self.assertTrue((output / "experiment.toml").exists())
            self.assertTrue((output / "prices.json").exists())
            self.assertFalse((output / "report.md").exists())
            self.assertFalse((output / "report.json").exists())
            self.assertFalse((output / "manifest.json").exists())

    def test_benchmark_doctor_failure_creates_no_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            output = Path(tmp, "bundle")
            spec.write_text(manifest(), encoding="utf-8")
            stderr = io.StringIO()
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stderr(stderr),
            ):
                code = gateway_probe_cli.main([
                    "benchmark",
                    str(spec),
                    "--output-dir",
                    str(output),
                ])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertIn("doctor failed", stderr.getvalue())

    def test_benchmark_reports_a_valid_first_block_partial_stop(self):
        environ = {
            "OPENAI_API_KEY": "direct-secret",
            "OPENROUTER_API_KEY": "gateway-secret",
            gateway_probe_run.FROZEN_PRICES_ENV: json.dumps({
                "openai/gpt-4o-mini": {
                    "input_per_million": "1",
                    "output_per_million": "2",
                    "effective_at": "2026-07-25T00:00:00Z",
                }
            }),
        }

        def partial_run(experiment_path, *, results_path, **_kwargs):
            experiment = gateway_probe_spec.load_experiment(experiment_path)
            direct = next(
                arm for arm in experiment.arms if arm.arm_id == "direct"
            )
            item = row("direct", "cold", 1, baseline=True)
            item["identity"]["experiment"] = {
                "id": experiment.experiment_id,
                "digest": experiment.digest,
            }
            item["identity"]["arm"]["digest"] = direct.digest
            item["model_match"] = experiment.model_match
            item["billing"]["stop_required"] = True
            item["outcome"]["budget_exhausted_reason"] = "usd_cap_reached"
            item["cell_id"] = gateway_probe_results.cell_id(item["identity"])
            path = Path(results_path)
            path.write_text(
                json.dumps(item, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return RunSummary(path, 1, 0, 0, 0)

        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp, "probe.toml")
            output = Path(tmp, "bundle")
            spec.write_text(manifest(), encoding="utf-8")
            with (
                mock.patch.dict(os.environ, environ, clear=True),
                mock.patch.object(
                    gateway_probe_run,
                    "run_experiment",
                    side_effect=partial_run,
                ) as run,
            ):
                code = gateway_probe_cli.main([
                    "benchmark",
                    str(spec),
                    "--output-dir",
                    str(output),
                    "--max-blocks",
                    "1",
                ])
            self.assertEqual(code, 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(
                report["arms"]["gateway"]["conditions"]["cold"][
                    "denominators"
                ]["attempted"],
                0,
            )
            self.assertEqual(
                report["paired_contrasts"]["gateway"]["cold"][
                    "request_stream_total_s"
                ]["coverage"],
                {"covered": 0, "ratio": 0.0, "total": 0},
            )
            self.assertTrue(all(
                count < report["scheduled_blocks_per_condition"]
                for count in report["complete_blocks"].values()
            ))
            self.assertEqual(run.call_args.kwargs["max_blocks"], 1)
            manifest_data = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("complete", manifest_data)


if __name__ == "__main__":
    unittest.main()
