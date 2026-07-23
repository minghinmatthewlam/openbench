# Bring your own harness

Pass one or more candidate files with `--candidate`; each file's `name` becomes
the independent results group label. `--candidate` accepts a filesystem path to
a TOML manifest **or** an installed harness-pack ref
(`org/name[@version][:manifest-stem]` — see [`docs/task-packs.md`](task-packs.md)).
Candidate `run_id` values add a short content digest (`name@digest:...`) so
editing a spec/config cannot silently reuse stale rows. Candidates may be mixed
with stock `--harness` names.

## Config variant

```toml
kind = "config-variant"
name = "codex-v2"
base_adapter = "codex"
config_dir = "../../ablation/codex-home-v2"
config_files = [
  # String entries copy unchanged. Table entries may rename and expand
  # {config_dir}/{workspace}/{model} in text files while staging.
  { source = "candidate-config.toml", destination = "config.toml", template = true },
  "pi-style-instructions.md",
]
[env]
CODEX_HOME = "{config_dir}"
[[auth_files]]
source = "~/.codex/auth.json"
destination = "auth.json"
```

Files are copied to a disposable directory. `{config_dir}`, `{workspace}`, and
`{model}` may be used in environment values and in config entries marked
`template = true`. Config entries may also rename a source with `destination`.
The base adapter retains model mapping, output parsing, version capture, and
proxy behavior. The checked-in V2 declaration is
`ablation/codex-home-v2/candidate.toml`; its staged config is byte-equivalent to
the former `env_override` composer.

A config variant may also select an adapter-supported experimental toggle. The
Codex multi-agent ON arm is checked in at
`experiments/multiagent-toggle/codex-on.toml`:

```toml
kind = "config-variant"
name = "codex-multiagent-on"
base_adapter = "codex"
config_dir = "codex-home"
config_files = ["config.toml"]
[env]
CODEX_HOME = "{config_dir}"
OPENBENCH_CODEX_MULTI_AGENT = "enabled"
```

That marker is consumed by the Codex adapter and changes only its explicit
`multi_agent` feature pin from `--disable` to `--enable`; an inherited host
environment variable cannot turn on the stock arm.

## Generic manifest

```toml
kind = "manifest"
name = "my-cli"
isolate_home = true
command = ["my-cli", "run", "--model", "{model}", "--workspace", "{workspace}",
           "{workspace_files}", "{prompt}"]
workspace_file_globs = ["src/**/*", "*.toml"]
version_command = ["my-cli", "--version"]
# Admission policy pins must name argv entries that are present in command.
policy_headless_args = ["run"]
policy_auto_approve_args = ["--yes"]
# Set only when this provider cannot be routed through the counting proxy.
# unmetered = true
# The safe default does not inherit arbitrary host variables. Name only the
# credentials/settings this CLI needs; Docker forwards these without values in argv.
pass_env = ["VENDOR_API_KEY"]
unset_env = ["MY_CLI_CONFIG"]
base_url_env = "MY_CLI_BASE_URL"
proxy_route = "chat/vendor/v1"
# Optional: write rotated auth_files back to the host masters (default false).
# persist_auth = true

[models]
"gpt-5.5-medium" = "gpt-5.5"
[env]
MY_CLI_HOME = "{home}/.my-cli"
[[auth_files]]
source = "~/.my-cli/auth.json"
destination = ".my-cli/auth.json"
```

Commands are argv arrays and never run through a shell. By default the child
environment contains only basic process variables, declared `pass_env` names,
manifest `[env]` values, and runner proxy variables. `inherit_env = true` is an
explicit compatibility escape hatch for stock-equivalence cases; it may expose
unrelated host credentials and should not be used for new manifests.
Supported scalar placeholders are `{prompt}`, `{workspace}`, `{model}`, and
`{home}`. The special whole-argument placeholder `{workspace_files}` expands to
sorted, de-duplicated relative file paths matched by `workspace_file_globs`;
the two must be declared together. Matches are contained within the disposable
workspace. This supports CLIs that require editable files as positional argv
instead of discovering them from their working directory. Auth files are copied to
the disposable home; sources must use home-relative `~/...` paths and missing
files return `SETUP-NEEDED`. Auth persist-back is **off by default** for
candidates: set `persist_auth = true` only when the CLI rotates tokens in those
declared `auth_files` and you want the host masters updated after the cell
(same atomic schema-preserving path stock adapters use). Without the flag,
disposable copies are discarded. `base_url_env` and
`proxy_route` opt the CLI into the counting proxy. `proxy_route` is the path
after `/cell/<token>/` (for example `chat/zai/api/paas/v4`); the CLI must honor
the declared base-URL environment variable. Generic output is retained as a
transcript; token fields remain unknown unless independently metered by the
proxy. See `obench/examples/pi-harness.toml` for a complete invocation-equivalent
manifest (it deliberately does not claim proxy support because Pi's native
adapter routes its subscription endpoint through a generated config file).

## Gateways / model routers

An AI gateway (or model router) exposes many providers behind one
OpenAI-compatible endpoint and one key. OpenBench can benchmark a harness
*through* a gateway: the harness still talks to the counting proxy, and the proxy
forwards to the gateway as its upstream, so tokens, latency, status, the
gateway-served model, and gateway-reported cost are all recorded per call.

```
pi -> counting proxy (/cell/<token>/gateway/<name>) -> gateway -> provider
```

Built-in gateway upstreams (`obench/proxy.py`):

| name          | upstream                                       | API key env                             |
|---------------|------------------------------------------------|-----------------------------------------|
| `openrouter`  | `https://openrouter.ai/api/v1`                 | `OPENROUTER_API_KEY`                    |
| `vercel`      | `https://ai-gateway.vercel.sh/v1`              | `AI_GATEWAY_API_KEY`                    |
| `concentrate` | `https://api.concentrate.ai/v1`                | `CONCENTRATE_API_KEY`                   |
| `cloudflare`  | `gateway.ai.cloudflare.com/v1/<acct>/<gw>/compat` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |

Add or override an upstream (e.g. a private/enterprise gateway) without code
changes: `obench run ... --gateway-upstream mygw=https://gateway.example.com/v1`.

**Cloudflare AI Gateway** differs from the others: its endpoint embeds your
account and gateway id, and its OpenAI-compatible endpoint forwards the
*underlying provider's own* key rather than a single gateway key. So it is
configured from the environment — set `CLOUDFLARE_ACCOUNT_ID` (and optionally
`CLOUDFLARE_GATEWAY_ID`, default `default`), plus the provider key the arm needs
(`cloudflare/openai/*` uses `OPENAI_API_KEY`, `cloudflare/anthropic/*` uses
`ANTHROPIC_API_KEY`). This targets an *unauthenticated* gateway; an
authenticated gateway (requiring a `cf-aig-authorization` header) is not yet
supported.

```bash
export CLOUDFLARE_ACCOUNT_ID=... OPENAI_API_KEY=...
obench run --harness pi --model cloudflare/openai/gpt-5.6 \
  --task make-it-run --proxy --trials 3
```

Model names are `\<gateway\>/\<provider\>/\<model\>`, so one gateway key reaches
different vendors in the same run (fixed-model mode — one model, no router
fallback):

```bash
export OPENROUTER_API_KEY=...   # or AI_GATEWAY_API_KEY / CONCENTRATE_API_KEY
obench run --harness pi --model openrouter/anthropic/claude-sonnet-4.5 \
  --task make-it-run --proxy --trials 3
obench run --harness pi --model vercel/openai/gpt-5.6 \
  --task make-it-run --proxy --trials 3
```

The ledger row for a gateway cell adds `served_model` (the model the gateway
actually served — may differ from the requested one under routing), `cost`, and
`upstream_cost` where the gateway reports them in the response `usage`.

Because the proxy sees every streamed byte, it also measures latency per call and
rolls it up per cell: `proxy_ttft_ms` (median time-to-first-token),
`proxy_gen_ms` (total streaming time), and `proxy_output_tps` (output
tokens/second). These are populated only for streaming calls. `obench report`
prints a per-`(harness, model)` latency summary (TTFT p50/p95 and output tok/s);
`obench report --latency` shows just that table.

## Doctor preflight

`obench doctor` accepts the same `--candidate` flag as `obench run`. For a
manifest it checks (no tokens spent):

1. **CLI** — `command[0]` is on `PATH`
2. **VERSION** — `version_command` exits 0 with non-empty output
3. **AUTH** — every declared `auth_files` source exists
4. **ENV** — declared `pass_env` names are set (warn/`INFO` if missing; does not
   fail the preflight unless you treat warnings as blocking yourself)
5. **MODEL** — `--model` resolves via `[models]` (any model is accepted when the
   map is empty)

For a config-variant it runs the base adapter's stock CLI/AUTH/MODEL/VERSION
checks plus a **CONFIG** check that `config_dir` and each `config_files` source
exist. Unknown plain `--harness` names still fail and list known stock names,
with a hint to pass `--candidate path/to/spec.toml`.

Stock adapters may optionally export doctor metadata so they stay in the
preflight table without editing `doctor.py`:

```python
DOCTOR = {"cli": "pi", "auth": _doctor_auth}  # auth(probes) -> (ok, detail)
```

`obench/adapters/pi.py` is the worked example; adapters without `DOCTOR` keep
the built-in fallback entries.

In Docker mode the candidate file's directory is mounted read-only, so config
sources must live in that directory tree. Declared auth sources must be under
the user's home directory; they are mounted read-only and copied into the
container's disposable home.

Both kinds record their spec digest, configuration digests, command/model data,
auth paths, environment policy, and environment variable names in
`candidate_provenance`, including the full candidate identity digest. Values of
environment variables and auth contents are deliberately excluded.

## Admission gate

A candidate must pass the admission gate before its rows may enter published
comparison tables:

```bash
# Safe schema/policy preview; launches no harness and spends no tokens.
obench gate experiments/candidates/aider.toml \
  --model deepseek-v4-flash

# Paid checks (run only by an operator with the intended credentials).
obench gate experiments/candidates/aider.toml \
  --model deepseek-v4-flash --live

# Optional, expensive n=1 calibration over the fixed 15-task set.
obench gate experiments/candidates/aider.toml \
  --model deepseek-v4-flash --live --calibrate
```

The command prints one `PASS`/`FAIL` line per check, a final verdict, and a JSON
record suitable for archiving. Without `--live` it validates the schema and
prints the expanded command that would run; it does not start a proxy, invoke a
CLI, or probe its version. A dry `--calibrate` only previews the fixed set;
the calibration cells run only when it is combined with `--live`.

The checklist protects comparability with native arms:

1. **Metering.** A smoke cell is routed through the counting proxy and must
   produce at least one ledger call. A genuinely non-routable provider may
   explicitly declare `unmetered = true`; every resulting benchmark row then
   carries `token_basis = "unmetered"` so reports can badge the limitation.
2. **Isolation.** The gate plants unique content in a canary file under a fake
   host `HOME`, runs the manifest with `isolate_home = true`, and rejects canary
   content found in captured transcript/workspace evidence. This is a simple
   heuristic, not syscall tracing: a harness that reads but never reproduces
   the bytes cannot be detected, and workspace scanning is bounded to 16 MiB.
   Config variants inherit their native
   adapter's isolation behavior.
3. **Policy parity.** Generic manifests declare `policy_headless_args` and
   `policy_auto_approve_args`; every declared argv entry must occur in
   `command`. A deterministic blocking executable is run through the candidate's
   own five-second timeout implementation and must return
   `failure_class = "timeout"`, proving the timeout path does not hang without
   depending on model compliance with a prompt.
   Config variants inherit the native adapter's policy pins.
4. **Version.** `version_command` must return non-empty output and that exact
   value must be stamped as `harness_version` on the metered smoke row.
5. **Failure honesty.** A smoke cell with declared provider-key environment
   variables set to a bogus value must be classified `infra` or `rate_limited`,
   never `wrong_answer`, and consume fewer than 100 tokens.
6. **Calibration.** `--calibrate` runs one trial on each of the fixed 15 tasks,
   prints the solve count, and fails suspicious all-zero or all-perfect
   outcomes. It is off by default because it spends money.

`experiments/candidates/aider.toml` is the worked example: it declares Aider's
`--message` headless pin and `--yes-always` approval pin, an isolated home,
version probe, DeepSeek proxy route, and provider-key environment name. Its
current result should be understood conceptually as: schema/policy dry-run
passes; live metering, timeout, version stamping, failure honesty, and optional
calibration remain operator-run evidence. This documentation does not claim a
live result and no live call is needed to review the manifest.

## Publish a shareable claim

After the gate and a matrix run against stock arms, build a postable bundle:

```bash
obench publish --results-path results/my-claim.jsonl --candidate my-cli
obench verify openbench-publish/<bundle>
```

The bundle is self-contained HTML + filtered `results.jsonl` + digests in
`provenance.json`. Transcripts stay local. See
[`docs/publish.md`](publish.md) for the full show-off workflow, what verify
proves, and what it deliberately does not.
