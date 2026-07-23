#!/usr/bin/env python3
"""Tests for the stdlib counting proxy and proxy ledger row mapping."""

import gzip
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


from obench import proxy  # noqa: E402
from obench import report  # noqa: E402
from obench import run  # noqa: E402
from obench.adapters import pi  # noqa: E402

SECRET = "TEST_SECRET_VALUE_MUST_NOT_APPEAR"


class FixtureUpstream(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        self.server.requests.append({
            "path": self.path,
            "auth": self.headers.get("authorization"),
            "host": self.headers.get("host"),
            "body": self.rfile.read(length).decode("utf-8", "replace"),
        })
        if self.path.endswith("/gateway-chat"):
            payload = json.dumps({
                "model": "anthropic/claude-sonnet-4.5",
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 8,
                    "total_tokens": 38,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                    "cost": 0.0123,
                    "cost_details": {"upstream_inference_cost": 0.0100},
                },
            }).encode()
            ctype = "application/json"
        elif self.path.endswith("/gateway-sse"):
            payload = (
                "data: {\"model\":\"openai/gpt-5.6\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
                "data: {\"model\":\"openai/gpt-5.6\",\"usage\":{\"prompt_tokens\":12,"
                "\"completion_tokens\":3,\"total_tokens\":15,\"cost\":0.0009}}\n\n"
                "data: [DONE]\n\n"
            ).encode()
            ctype = "text/event-stream"
        elif self.path.endswith("/sse"):
            payload = (
                "event: response.completed\n"
                "data: {\"response\":{\"usage\":{\"input_tokens\":10,"
                "\"input_tokens_details\":{\"cached_tokens\":3},"
                "\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n"
            ).encode()
            ctype = "text/event-stream"
        elif self.path.endswith("/messages"):
            payload = json.dumps({
                "usage": {
                    "input_tokens": 11,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 7,
                },
            }).encode()
            ctype = "application/json"
        else:
            payload = json.dumps({
                "model": "fixture-model",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            }).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="proxy_test_")
        self.upstream = FixtureUpstream(("127.0.0.1", 0), FixtureHandler)
        self.upstream.requests = []
        self.up_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.up_thread.start()
        upstream_url = f"http://127.0.0.1:{self.upstream.server_address[1]}"
        self.proxy = proxy.make_server(
            "127.0.0.1", 0, self.tmp.name,
            chat_upstreams={"deepseek": upstream_url},
            gateway_upstreams={
                "openrouter": upstream_url,
                # A gateway upstream that carries a base path (like the real
                # openrouter/vercel/concentrate upstreams) — exercises the
                # path-join so the doubled-/api/v1 bug can't regress.
                "orv1": upstream_url + "/api/v1",
                # Cloudflare's endpoint carries a deep base path ending /compat.
                "cloudflare": upstream_url + "/v1/acct/default/compat",
            },
            anthropic_upstreams={"deepseek": upstream_url},
            openai_upstream=upstream_url,
            subbridge_upstream=upstream_url,
            cursor_upstream=upstream_url,
        )
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.host, self.port = self.proxy.server_address[:2]

    def tearDown(self):
        self.proxy.shutdown(); self.proxy.server_close()
        self.upstream.shutdown(); self.upstream.server_close()
        self.tmp.cleanup()

    def _post(self, path, body=None):
        body = body or {"model": "deepseek-v4-flash", "temperature": 0.2, "api_key": SECRET}
        data = json.dumps(body).encode()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        conn.request("POST", path, body=data, headers={
            "content-type": "application/json",
            "content-length": str(len(data)),
            "authorization": f"Bearer {SECRET}",
        })
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200, payload)
        return payload

    def _ledger(self, token):
        path = os.path.join(self.tmp.name, token + ".jsonl")
        deadline = time.time() + 2
        while (not os.path.exists(path) or os.path.getsize(path) == 0) and time.time() < deadline:
            time.sleep(0.01)
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def test_json_usage_sampling_observation_and_auth_scrubbing(self):
        self._post("/cell/tok-json/chat/deepseek/chat/completions")
        self.assertEqual(self.upstream.requests[-1]["path"], "/chat/completions")
        self.assertEqual(self.upstream.requests[-1]["auth"], f"Bearer {SECRET}")
        self._ledger("tok-json")
        with open(os.path.join(self.tmp.name, "tok-json.jsonl"), encoding="utf-8") as fh:
            ledger_text = fh.read()
        self.assertNotIn(SECRET, ledger_text)
        row = json.loads(ledger_text)
        self.assertEqual(row["usage"]["prompt_tokens"], 20)
        self.assertEqual(row["sampling_observed"]["model"], "deepseek-v4-flash")
        self.assertNotIn("api_key", row["sampling_observed"])

    def test_nested_and_compressed_sampling_observation(self):
        body = gzip.compress(json.dumps({
            "request": {"model_slug": "nested-model", "reasoning": {"effort": "medium"}},
            "stream": True,
        }).encode())
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        conn.request("POST", "/cell/tok-gzip/chat/deepseek/chat/completions", body=body, headers={
            "content-type": "application/json",
            "content-encoding": "gzip",
            "content-length": str(len(body)),
        })
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        self.assertEqual(resp.status, 200, payload)
        row = self._ledger("tok-gzip")[0]
        self.assertEqual(row["sampling_observed"]["model"], "nested-model")
        self.assertEqual(row["sampling_observed"]["reasoning"], {"effort": "medium"})
        self.assertIs(row["sampling_observed"]["stream"], True)

    def test_protocol_links_extract_opaque_conversation_chain(self):
        request = json.dumps({
            "conversation": {"id": "conversation-secret"},
            "previous_response_id": "response-secret",
        }).encode()
        self.assertEqual(proxy.protocol_links(request), {
            "session": "conversation-secret",
            "previous_response": "response-secret",
        })
        response = b'data: {"type":"response.completed","response":{"id":"next-secret"}}\n\n'
        self.assertEqual(proxy.protocol_links(response, response=True), {
            "response": "next-secret",
        })

    def test_sse_usage_parsing(self):
        self._post("/cell/tok-sse/anthropic/deepseek/sse")
        row = self._ledger("tok-sse")[0]
        self.assertEqual(row["usage"]["input_tokens"], 10)
        self.assertEqual(row["usage"]["output_tokens_details"]["reasoning_tokens"], 1)

    def test_anthropic_json_usage_parsing(self):
        path = "/cell/tok-anthropic/" + "anthropic/deepseek/" + "anthropic/v1/messages"
        self._post(path)
        row = self._ledger("tok-anthropic")[0]
        self.assertEqual(self.upstream.requests[-1]["path"], "/anthropic/v1/messages")
        self.assertEqual(row["usage"]["input_tokens"], 11)
        mapped = run.proxy_split_from_usage(row["usage"])
        self.assertEqual(mapped["tokens_proxy_input_uncached"], 11)
        self.assertEqual(mapped["tokens_proxy_cache_read"], 3)
        self.assertEqual(mapped["tokens_proxy_cache_write"], 2)
        self.assertEqual(mapped["tokens_proxy_output"], 7)

    def test_cliproxyapi_upstream_is_metered_with_mock_http(self):
        self._post(
            "/cell/tok-subbridge/subbridge/v1/chat/completions",
            {"model": "gpt-5.6", "stream": False},
        )
        row = self._ledger("tok-subbridge")[0]
        request = self.upstream.requests[-1]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["host"], f"127.0.0.1:{self.upstream.server_address[1]}")
        self.assertEqual(request["auth"], f"Bearer {SECRET}")
        self.assertEqual(row["route"], "subbridge")
        self.assertEqual(row["usage"]["prompt_tokens"], 20)
        self.assertEqual(row["sampling_observed"]["model"], "gpt-5.6")

    def test_cursor_private_protocol_route_is_forwarded(self):
        self._post("/cell/tok-cursor/cursor/agent/v1/run")
        row = self._ledger("tok-cursor")[0]
        self.assertEqual(self.upstream.requests[-1]["path"], "/agent/v1/run")
        self.assertEqual(row["route"], "cursor")

    def test_registered_token_gate_rejects_arbitrary_docker_client(self):
        self.proxy.require_registered_tokens = True
        data = b'{"model":"gpt-5.6"}'
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        conn.request("POST", "/cell/guessed/" + "subbridge/v1/chat/completions", body=data, headers={
            "content-type": "application/json",
            "content-length": str(len(data)),
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 502)
        self.assertEqual(self.upstream.requests, [])

        with open(os.path.join(self.tmp.name, "registered.meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"harness": "grokbuild", "model": "gpt-5.6"}, fh)
        self._post("/cell/registered/" + "subbridge/v1/chat/completions", {"model": "gpt-5.6"})
        self.assertEqual(self.upstream.requests[-1]["path"], "/v1/chat/completions")

    def test_cell_token_routing_isolation(self):
        self._post("/cell/tok-a/chat/deepseek/chat/completions")
        self._post("/cell/tok-b/chat/deepseek/chat/completions")
        self._ledger("tok-a")
        self._ledger("tok-b")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "tok-a.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "tok-b.jsonl")))
        self.assertEqual(len(self._ledger("tok-a")), 1)
        self.assertEqual(len(self._ledger("tok-b")), 1)

    def test_ledger_to_row_mapping(self):
        row = {}
        run.apply_proxy_ledger(row, [
            {"usage": {"prompt_tokens": 20, "completion_tokens": 5,
                       "prompt_tokens_details": {"cached_tokens": 2},
                       "completion_tokens_details": {"reasoning_tokens": 1}},
             "sampling_observed": {"model": "m", "temperature": 0.1}},
            {"usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 3},
                       "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 1}}},
        ])
        self.assertEqual(row["tokens_proxy_input_uncached"], 25)
        self.assertEqual(row["tokens_proxy_cache_read"], 5)
        self.assertEqual(row["tokens_proxy_output"], 9)
        self.assertEqual(row["tokens_proxy_reasoning"], 2)
        self.assertEqual(row["tokens_proxy_calls"], 2)
        self.assertEqual(row["token_basis_proxy"], "proxy_measured")
        self.assertEqual(row["sampling_observed"], [{"model": "m", "temperature": 0.1}])

    def test_truncated_ledger_does_not_claim_proxy_measured(self):
        row = {}
        run.apply_proxy_ledger(row, [
            {"usage": {"prompt_tokens": 20, "completion_tokens": 5},
             "capture_truncated": True},
        ])
        self.assertTrue(row.get("proxy_capture_truncated"))
        self.assertEqual(row["tokens_proxy_calls"], 1)
        self.assertEqual(row["tokens_proxy_input_uncached"], 20)
        self.assertNotIn("token_basis_proxy", row)

    def test_remaining_lane_support_matrix(self):
        self.assertFalse(run.proxy_supported_for_cell("cursor", "gpt-5.5-medium"))
        self.assertTrue(run.proxy_supported_for_cell("grokbuild", "deepseek-v4-flash"))
        self.assertFalse(run.proxy_supported_for_cell("devin", "gpt-5.5-medium"))

    def test_gateway_route_meters_served_model_and_cost(self):
        self._post("/cell/tok-gw/gateway/openrouter/gateway-chat",
                   {"model": "anthropic/claude-sonnet-4.5", "stream": False})
        self.assertEqual(self.upstream.requests[-1]["path"], "/gateway-chat")
        row = self._ledger("tok-gw")[0]
        self.assertEqual(row["route"], "gateway/openrouter")
        # served_model comes from the RESPONSE (what the gateway actually served),
        # not the request's requested model.
        self.assertEqual(row["served_model"], "anthropic/claude-sonnet-4.5")
        self.assertEqual(row["cost"], 0.0123)
        self.assertEqual(row["upstream_cost"], 0.01)
        self.assertEqual(row["usage"]["prompt_tokens"], 30)

    def test_gateway_sse_served_model_and_cost(self):
        self._post("/cell/tok-gw-sse/gateway/openrouter/gateway-sse",
                   {"model": "openai/gpt-5.6", "stream": True})
        row = self._ledger("tok-gw-sse")[0]
        self.assertEqual(row["served_model"], "openai/gpt-5.6")
        self.assertEqual(row["cost"], 0.0009)
        self.assertEqual(row["usage"]["total_tokens"], 15)

    def test_unknown_gateway_is_rejected(self):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        data = b'{"model":"x"}'
        conn.request("POST", "/cell/tok-badgw/gateway/nope/chat/completions", body=data, headers={
            "content-type": "application/json", "content-length": str(len(data)),
        })
        resp = conn.getresponse()
        resp.read()
        conn.close()
        self.assertEqual(resp.status, 502)

    def test_extract_response_model_and_cost_units(self):
        self.assertEqual(
            proxy.extract_response_model(b'{"model":"a/b","usage":{"prompt_tokens":1}}'),
            "a/b")
        self.assertEqual(
            proxy.extract_response_model(
                b'data: {"model":"a"}\n\ndata: {"model":"b"}\n\n'),
            "b")
        self.assertEqual(
            proxy.extract_response_model(b'{"response":{"model":"r/m"}}'), "r/m")
        self.assertIsNone(proxy.extract_response_model(b'{"usage":{"prompt_tokens":1}}'))
        self.assertEqual(
            proxy.extract_cost({"cost": 0.5, "cost_details": {"upstream_inference_cost": 0.4}}),
            {"cost": 0.5, "upstream_cost": 0.4})
        self.assertEqual(proxy.extract_cost({"prompt_tokens": 1}), {})
        # booleans must not be mistaken for numeric cost
        self.assertEqual(proxy.extract_cost({"cost": True}), {})

    def test_gateway_upstream_base_path_is_not_doubled(self):
        # pi hands the proxy a base URL of .../gateway/orv1 (no path tail); pi's
        # openai-completions api then appends /chat/completions. With an upstream
        # that carries /api/v1, the forwarded path must be /api/v1/chat/completions
        # — NOT /api/v1/api/v1/chat/completions.
        self._post("/cell/tok-orv1/gateway/orv1/chat/completions",
                   {"model": "openai/gpt-5.6", "stream": False})
        self.assertEqual(self.upstream.requests[-1]["path"], "/api/v1/chat/completions")
        row = self._ledger("tok-orv1")[0]
        self.assertEqual(row["route"], "gateway/orv1")

    def test_pi_proxied_base_url_for_gateway_drops_path_tail(self):
        # Regression: the gateway upstream registry carries the /api/v1 base, so
        # the pi base URL must be .../gateway/<name> with no model-base-url tail.
        env = {
            "OPENBENCH_PROXY": "1",
            "OPENBENCH_PROXY_BASE_URL": "http://127.0.0.1:9",
            "OPENBENCH_PROXY_CELL_TOKEN": "tok",
        }
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            url = pi._proxied_base_url("openrouter", "https://openrouter.ai/api/v1")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual(url, "http://127.0.0.1:9/cell/tok/gateway/openrouter")
        self.assertNotIn("api/v1", url)

    def test_pi_gateway_models_cover_all_gateways(self):
        for name in ("openrouter", "vercel", "concentrate", "cloudflare"):
            self.assertIn(f"{name}/openai/gpt-5.6", pi.GATEWAY_MODELS)
            spec = pi.GATEWAY_MODELS[f"{name}/anthropic/claude-sonnet-4.5"]
            self.assertEqual(spec["provider"], name)
            self.assertEqual(spec["model_id"], "anthropic/claude-sonnet-4.5")
            self.assertIn(name, pi.GATEWAY_PROVIDERS)

    def test_cloudflare_uses_per_provider_key_and_templated_base_url(self):
        env = {"CLOUDFLARE_ACCOUNT_ID": "acct123", "CLOUDFLARE_GATEWAY_ID": "gw9"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            models = pi._build_gateway_models()
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        # Cloudflare forwards the underlying provider's own key, not one gateway key.
        self.assertEqual(models["cloudflare/openai/gpt-5.6"]["env_key"], "OPENAI_API_KEY")
        self.assertEqual(
            models["cloudflare/anthropic/claude-sonnet-4.5"]["env_key"], "ANTHROPIC_API_KEY")
        # Base URL is templated from the account + gateway id.
        self.assertEqual(
            models["cloudflare/openai/gpt-5.6"]["base_url"],
            "https://gateway.ai.cloudflare.com/v1/acct123/gw9/compat")

    def test_cloudflare_gateway_id_defaults_to_default(self):
        old_acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        old_gw = os.environ.get("CLOUDFLARE_GATEWAY_ID")
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acctX"
        os.environ.pop("CLOUDFLARE_GATEWAY_ID", None)
        try:
            self.assertEqual(
                pi._cloudflare_base_url(),
                "https://gateway.ai.cloudflare.com/v1/acctX/default/compat")
        finally:
            if old_acct is None:
                os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
            else:
                os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_acct
            if old_gw is not None:
                os.environ["CLOUDFLARE_GATEWAY_ID"] = old_gw

    def test_gateway_upstreams_for_proxy_templates_cloudflare(self):
        old_acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        old_gw = os.environ.get("CLOUDFLARE_GATEWAY_ID")
        os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acctY"
        os.environ.pop("CLOUDFLARE_GATEWAY_ID", None)
        try:
            ups = run._gateway_upstreams_for_proxy("cloudflare/openai/gpt-5.6")
            self.assertEqual(
                ups["cloudflare"],
                "https://gateway.ai.cloudflare.com/v1/acctY/default/compat")
            # Non-cloudflare arms add nothing; missing account id yields nothing.
            self.assertEqual(run._gateway_upstreams_for_proxy("openrouter/openai/gpt-5.6"), {})
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
            self.assertEqual(run._gateway_upstreams_for_proxy("cloudflare/openai/gpt-5.6"), {})
        finally:
            if old_acct is None:
                os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
            else:
                os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_acct
            if old_gw is not None:
                os.environ["CLOUDFLARE_GATEWAY_ID"] = old_gw

    def test_cloudflare_gateway_route_path_join(self):
        # base path ends /compat; pi appends /chat/completions -> exactly one join.
        self._post("/cell/tok-cf/gateway/cloudflare/chat/completions",
                   {"model": "openai/gpt-5.6", "stream": False})
        self.assertEqual(
            self.upstream.requests[-1]["path"], "/v1/acct/default/compat/chat/completions")
        self.assertEqual(self._ledger("tok-cf")[0]["route"], "gateway/cloudflare")

    def test_pi_gateway_cells_are_proxy_supported(self):
        self.assertTrue(run.proxy_supported_for_cell("pi", "openrouter/openai/gpt-5.6"))
        self.assertTrue(
            run.proxy_supported_for_cell("pi", "openrouter/anthropic/claude-sonnet-4.5"))
        self.assertTrue(run.proxy_supported_for_cell("pi", "cloudflare/openai/gpt-5.6"))
        self.assertFalse(run.proxy_supported_for_cell("opencode", "openrouter/openai/gpt-5.6"))

    def test_apply_proxy_ledger_aggregates_served_model_and_cost(self):
        row = {}
        run.apply_proxy_ledger(row, [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2},
             "served_model": "openai/gpt-5.6", "cost": 0.001, "upstream_cost": 0.0008},
            {"usage": {"prompt_tokens": 5, "completion_tokens": 1},
             "served_model": "openai/gpt-5.6", "cost": 0.0005},
        ])
        self.assertEqual(row["served_model"], ["openai/gpt-5.6"])
        self.assertAlmostEqual(row["cost"], 0.0015)
        self.assertAlmostEqual(row["upstream_cost"], 0.0008)
        self.assertEqual(row["tokens_proxy_calls"], 2)

    def test_proxy_records_ttft_ms(self):
        # Every proxied response yields at least a first-byte timestamp, so the
        # ledger row must carry a non-negative ttft_ms.
        self._post("/cell/tok-ttft/chat/deepseek/chat/completions")
        row = self._ledger("tok-ttft")[0]
        self.assertIn("ttft_ms", row)
        self.assertIsInstance(row["ttft_ms"], int)
        self.assertGreaterEqual(row["ttft_ms"], 0)

    def test_apply_proxy_ledger_aggregates_latency(self):
        # ttft_ms -> median across calls; gen_ms -> summed; output_tps derived
        # from proxy output tokens over total generation seconds.
        row = {}
        run.apply_proxy_ledger(row, [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 100},
             "ttft_ms": 200, "gen_ms": 1000},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 100},
             "ttft_ms": 400, "gen_ms": 1000},
            {"usage": {"prompt_tokens": 10, "completion_tokens": 100},
             "ttft_ms": 600, "gen_ms": 2000},
        ])
        self.assertEqual(row["proxy_ttft_ms"], 400)     # median(200,400,600)
        self.assertEqual(row["proxy_gen_ms"], 4000)     # 1000+1000+2000
        # 300 output tokens over 4.0s = 75.0 tok/s
        self.assertEqual(row["proxy_output_tps"], 75.0)

    def test_apply_proxy_ledger_latency_absent_when_no_timing(self):
        row = {}
        run.apply_proxy_ledger(row, [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
        ])
        self.assertIsNone(row.get("proxy_ttft_ms"))
        self.assertIsNone(row.get("proxy_gen_ms"))
        self.assertIsNone(row.get("proxy_output_tps"))


class ReportLatencyTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertIsNone(report.percentile([], 50))
        self.assertEqual(report.percentile([5], 95), 5.0)
        self.assertEqual(report.percentile([10, 20, 30], 50), 20.0)
        self.assertEqual(report.percentile([0, 100], 95), 95.0)

    def test_format_latency_reports_ttft_and_tps(self):
        rows = [
            {"harness": "pi", "model": "openrouter/openai/gpt-5.6", "task": "t",
             "success": True, "score": 1.0,
             "proxy_ttft_ms": 200, "proxy_output_tps": 80.0},
            {"harness": "pi", "model": "openrouter/openai/gpt-5.6", "task": "t",
             "success": True, "score": 1.0,
             "proxy_ttft_ms": 600, "proxy_output_tps": 40.0},
        ]
        arms, _tasks, stats = report.aggregate(rows)
        text = report.format_latency(arms, stats)
        self.assertIn("ttft_p50_ms", text)
        self.assertIn("openrouter/openai/gpt-5.6", text)
        st = stats[("pi", "openrouter/openai/gpt-5.6")]
        self.assertEqual(report.percentile(st["ttft_vals"], 50), 400.0)
        self.assertEqual(report.percentile(st["tps_vals"], 50), 60.0)

    def test_format_latency_empty_when_no_proxy_timing(self):
        rows = [{"harness": "pi", "model": "m", "task": "t",
                 "success": True, "score": 1.0}]
        arms, _tasks, stats = report.aggregate(rows)
        text = report.format_latency(arms, stats)
        self.assertIn("no proxy-measured latency", text)


if __name__ == "__main__":
    unittest.main()
