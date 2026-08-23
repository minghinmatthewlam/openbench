#!/usr/bin/env python3
"""Failure taxonomy shared by the runner, reports, and backfill tool.

The runner classifies rows at write time using the adapter's full untruncated
output when adapters provide ``full_output``. Backfill can only use fields saved
in old JSONL rows (usually ``output_tail``), so old-file detection is inherently
weaker for markers that were truncated away.

Rows that ride the wall-time cap normally classify as ``timeout``. A cap-riding
row with no evidence of agent work is treated as harness/provider infra instead:
no tokens, no turns, and effectively empty saved output/text after generic
headers and timeout boilerplate are removed. Genuine timeouts still show work via
tokens, turns, or meaningful transcript/error text and remain ``timeout``.

Zero-work classifier gate (P0 hardening)
-----------------------------------------
A completed cell with zero output tokens AND at most 1 turn AND no meaningful
adapter-output text is infrastructure, never a capability "wrong answer."
The canonical pattern is a gpt-5.6-sol cell that completes in 2--9s wall time
with 0 output tokens and exactly 1 turn — the harness started, the model
returned nothing useful (connection dropped, auth rejected early, or the
adapter's minimal turn was a probe that got no response), and the cell
finished without the adapter crashing visibly.

This is symmetric with the crash-marker / adapter-traceback gate: when no
model output was produced, the failure belongs to the harness or provider, not
the model's ability to solve the task.
"""

import re

STALLED = "stalled"

FAILURE_CLASSES = ("solved", "wrong_answer", "timeout", "rate_limited", "infra", STALLED)
EXCLUDED_FROM_SOLVE_RATE = ("rate_limited", "infra", STALLED)
NEAR_ZERO_TOKEN_LIMIT = 100
_TOKEN_FIELDS = (
    "tokens", "tokens_fresh", "tokens_input_uncached", "tokens_cache_read",
    "tokens_cache_write", "tokens_output", "tokens_reasoning",
    "tokens_proxy_input_uncached", "tokens_proxy_cache_read",
    "tokens_proxy_cache_write", "tokens_proxy_output", "tokens_proxy_reasoning",
)

# Rate-limit classification excludes rows from solve-rate denominators, so the
# detector intentionally requires provider/API-error context rather than any
# domain text that happens to mention e.g. HTTP 429 or a rate limiter task.
_RATE_LIMIT_RES = [
    re.compile(r"\b(?:APIError|API error|HTTPError|HTTP error|provider|vendor)\b[^\n]{0,200}\b(?:429|rate[_ -]?limit|TPD|quota|too many requests)", re.IGNORECASE),
    re.compile(r"\b(?:HTTP|status(?: code)?|response)\s*[:=]?\s*429\b[^\n]{0,200}\b(?:rate[_ -]?limit|TPD|quota|too many requests)", re.IGNORECASE),
    re.compile(r"\b(?:429|rate[_ -]?limit|TPD|quota|too many requests)\b[^\n]{0,200}\b(?:APIError|API error|HTTPError|HTTP error|provider|vendor|status(?: code)?|response)\b", re.IGNORECASE),
    # Moonshot/Kimi daily-token exhaustion uses this distinctive provider text.
    re.compile(r"\bTPD\b[^\n]{0,160}\b(?:current|limit|rate)", re.IGNORECASE),
]

_INFRA_RE = re.compile(
    r"("
    r"docker (?:daemon not reachable|unavailable|desktop)|"
    r"DockerUnavailable|"
    r"container produced no result sentinel|"
    r"no result sentinel|"
    r"SETUP-NEEDED|"
    r"missing [^\n]*(?:auth|credential|api[_ -]?key)|"
    r"No API key for provider|"
    r"not logged in|login required|please log in|"
    r"No such image:\s*[^\s]+|"
    r"image ['\"]?[^'\"\n]+['\"]? not found|"
    r"Cannot connect to the Docker daemon|"
    r"proxy_upstream_failed|"
    # Provider auth rejections in vendor/liteLLM phrasing (aider et al.):
    r"authentication_error|AuthenticationError|"
    r"api key[^\n]{0,40}\binvalid|invalid[^\n]{0,20}api key|incorrect api key"
    r")",
    re.IGNORECASE,
)

# Adapter-side crash before any model call: an uncaught Python traceback
# (e.g. FileNotFoundError for a missing harness binary) is infrastructure —
# but only when there is no evidence a model ran. Agents debugging Python
# tasks legitimately print tracebacks in their transcripts, so these markers
# must never reclassify a cell that shows real work (tokens or turns).
_ADAPTER_CRASH_RE = re.compile(
    r"Traceback \(most recent call last\)|FileNotFoundError",
    re.IGNORECASE,
)

_TIMEOUT_RE = re.compile(r"\b(?:timeout|timed out)\b", re.IGNORECASE)


def _text(*parts):
    return "\n".join(str(p) for p in parts if p is not None)


def has_rate_limit_marker(text):
    """Return True when CLI output contains a high-confidence rate-limit signature."""
    return any(rx.search(text or "") for rx in _RATE_LIMIT_RES)


def has_infra_marker(text):
    """Return True when row/error/output text indicates harness infrastructure."""
    return bool(_INFRA_RE.search(text or ""))


def has_instant_cli_exit_shape(row):
    """Return True for near-instant adapter CLI exits with no model traffic.

    The post-run gate uses the same predicate as a drift guard. The contract is
    deliberately narrow and structural: a bare ``exit N`` before 30 seconds with
    no aggregate token count is a local CLI/configuration failure, not a task
    wrong answer.
    """
    row = row or {}
    if bool(row.get("completed")):
        return False
    if not re.match(r"^exit \d+$", str(row.get("error") or "")):
        return False
    if row.get("tokens") not in (None, 0):
        return False
    wall = row.get("wall_time_s")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool):
        return False
    return float(wall) < 30.0


def has_no_work_incomplete_shape(row, text=""):
    """True for a measured incomplete run with zero evidence the model worked.

    Generalizes ``has_instant_cli_exit_shape`` beyond its <30s window. A run that
    explicitly did NOT complete, reported no tokens across every field, ran no
    turns, and left the workspace untouched is a harness/provider abandonment --
    an auth reject, a dropped connection, or a provider-overload retry loop that
    answers "high demand / Reconnecting 1..5/5" for minutes before exiting. It
    burns real wall time, so the instant-exit gate (too slow) and the cap-rider
    gate (never reached the cap) both miss it, and its throttle text is written to
    the transcript rather than the saved row fields -- so a structural no-work
    shape catches it where marker matching cannot.

    The gate demands POSITIVE evidence of an abandoned run rather than mere
    absence of telemetry: ``completed`` explicitly False and a numeric
    ``wall_time_s``. A sparse/old row simply missing those fields is not swallowed
    (that would clobber a backfilled class), and meaningful transcript text or any
    timeout signal (handled before this gate) keeps the cell out. Callers must
    invoke this AFTER the timeout classification so a real timeout still wins.
    """
    row = row or {}
    if row.get("harness") == "null":
        return False
    if row.get("completed") is not False or bool(row.get("success")):
        return False
    wall = row.get("wall_time_s")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool):
        return False
    if any(row.get(field) for field in _TOKEN_FIELDS):
        return False
    if row.get("turns") not in (None, 0):
        return False
    if _has_workspace_work_evidence(row):
        return False
    if len(_meaningful_work_text(text)) >= 200:
        return False
    return True


def per_reply_outputs(row):
    """Per-REQUEST output token counts for a cell, or () if unavailable.

    Exists because ``tokens_output`` is the SUM over every turn, while an output
    cap (``max_completion_tokens``) bounds a SINGLE reply. Comparing the two
    directly is a unit error that reports any multi-turn run as truncated --
    verified on a real cell: turns=18, tokens_output=28466, which equals the sum
    of its 18 per-call outputs exactly.

    That mistake was made twice in one session, producing "224 truncated cells"
    (real: 11) and "56% of laguna / 65% of deepseek cells hit the cap"
    (real: 14 and 0). Both times the falsifying check was one division: median
    output PER TURN was 864 against a cap of 8192.

    ``usage_raw`` holds one entry per model call, so this is exact rather than
    an average. Use this -- never ``tokens_output`` -- against a per-reply cap.
    """
    usage = (row or {}).get("usage_raw")
    if not isinstance(usage, list):
        return ()
    out = []
    for call in usage:
        if isinstance(call, dict):
            value = call.get("output")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(int(value))
    return tuple(out)


def has_per_reply_truncation(row, cap=None, ratio=0.99):
    """True when any single reply reached the per-request output cap.

    ``cap`` defaults to the smallest cap actually observed on the cell's own
    requests (``sampling_observed``), so it needs no external table.
    """
    row = row or {}
    if cap is None:
        caps = [s.get("max_completion_tokens") or s.get("max_tokens")
                for s in row.get("sampling_observed") or () if isinstance(s, dict)]
        caps = [c for c in caps
                if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0]
        if not caps:
            return False
        cap = max(caps)
    replies = per_reply_outputs(row)
    if not replies:
        return False
    # If a reply EXCEEDS its own cap, the cap is not being enforced on this
    # route, so "reply sits at the cap" carries no information and no truncation
    # can be inferred. Measured: deepseek-v4-flash overshoots on 29% of cells,
    # by up to 4.9x (a 17096-token reply against an 8192 cap) -- its endpoint
    # either ignores max_tokens or counts reasoning tokens outside it. laguna,
    # inkling and kimi-k3 never exceed their cap, so the signal is valid there.
    if max(replies) > cap:
        return False
    return max(replies) >= ratio * cap


STARVED_OUTPUT_BUDGET = 64


def has_throttle_dominated_replies(row):
    """True when more assistant replies died on a provider throttle than arrived.

    The signal the checker-owned-verdict rule cannot see on its own: a cell can
    complete, produce SOME tokens, touch the workspace, and receive a real
    checker FAIL -- while most of its model calls were 429-killed. That is a
    measurement of provider capacity, not of the model. wide25: laguna had 38 of
    50 cells in this state (370 replies ending "429 ... temporarily
    rate-limited") while deepseek ran 0 of 50; laguna's published 6% was
    starvation wearing a verdict.

    Requires the adapter-reported ``replies_ok``/``replies_throttled`` fields.
    Rows that predate them return False here -- their storms are only visible in
    transcripts (see the backfill tool).
    """
    row = row or {}
    ok, throttled = row.get("replies_ok"), row.get("replies_throttled")
    if not isinstance(throttled, int) or not isinstance(ok, int):
        return False
    return throttled > ok


def has_starved_output_budget(row):
    """True when a request was sent with an output budget too small to answer.

    pi derives each request's max output tokens from the model's configured
    context window minus the prompt, so on a long conversation the budget decays
    monotonically -- observed on one deepseek cell as
    8192 -> 8173 -> 4998 -> ... -> 1258 -> 1. A model asked to reply in 1 token
    cannot answer, so scoring the cell as wrong_answer measures the harness's
    budget arithmetic, not the model.

    The effect is large and one-sided: deepseek cells containing a starved
    request had median 114 turns and solved 3/21 (14%), against 11.5 turns and
    103/197 (52%) for the rest. It also hits long-horizon task sets harder --
    16% of tb-mid cells versus 8% of TB-7 -- which manufactured an apparent
    model x task-set interaction that vanished once these cells were excluded.

    Only capped models can trip this. gpt-5.6-sol runs with no cap at all, so
    the handicap fell entirely on the open models it was being compared against.
    """
    for sample in (row or {}).get("sampling_observed") or ():
        if not isinstance(sample, dict):
            continue
        cap = sample.get("max_completion_tokens") or sample.get("max_tokens")
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and 0 < cap <= STARVED_OUTPUT_BUDGET:
            return True
    return False


def has_zero_output_and_minimal_turns(row, text=""):
    """True when a completed cell produced no model output and at most 1 turn.

    A harness that started but returned zero output tokens in its only turn
    (or zero turns) is infrastructure: the model never produced a useful
    response.  This prevents cells that complete in a few seconds with no
    output tokens from being misclassified as ``wrong_answer`` (the model
    couldn't have answered wrong if it never answered).

    The canonical pattern: gpt-5.6-sol cells with 2--9s wall time,
    0 tokens_output, 1 turn, untouched workspace.

    Pass ``text`` (adapter output) to suppress the gate when meaningful
    model/adapter text exists despite zero reported tokens — some adapters
    print diagnostic output on stdout without going through the token parser.
    """
    row = row or {}
    if row.get("harness") == "null":
        return False
    if not bool(row.get("completed")):
        return False
    if bool(row.get("success")):
        return False
    # Check tokens_output specifically — zero model output is the signal.
    out = row.get("tokens_output")
    if out not in (None, 0):
        return False
    # Also check aggregate tokens to catch proxy-only token fields.
    if row.get("tokens") not in (None, 0):
        return False
    # At most 1 turn — a single probe-turn that produced nothing is infra.
    turns = row.get("turns")
    if turns is None or (isinstance(turns, (int, float)) and turns > 1):
        return False
    # Suppress gate when there's meaningful output text despite zero tokens.
    if text and len(_meaningful_work_text(text)) >= 10:
        return False
    return True


def _wall_rode_cap(row, timeout_s):
    if timeout_s is None:
        timeout_s = row.get("timeout_s")
    if not timeout_s:
        return False
    wall = row.get("wall_time_s")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool):
        return False
    try:
        cap = float(timeout_s)
    except (TypeError, ValueError):
        return False
    if cap <= 0:
        return False
    return float(wall) >= (cap * 0.98)


def _meaningful_work_text(text):
    """Return text evidence after dropping generic transcript/timeout boilerplate."""
    kept = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"[-=]{3,}", line):
            continue
        if re.fullmatch(r"(?:transcript|output|stdout|stderr|tail|error|metadata)\s*:?", line, re.IGNORECASE):
            continue
        if re.fullmatch(r"(?:timeout|timed out)(?: after)? \d+(?:\.\d+)?s?", line, re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def has_checker_owned_verdict(row, text=""):
    """True when the checker already judged a cell that genuinely ran.

    Three conditions, all required:

    * ``completed`` -- the harness exited on its own rather than being killed,
      so the agent was not cut off mid-answer.
    * ``checker_exit`` in {0, 1} -- a real verdict. Anything else is a checker
      crash and stays infra (see ``has_checker_crash``); "timeout" is handled by
      the caller.
    * the cell did real work -- guards against a cell that was throttled before
      producing anything and then trivially failed the checker. Those keep their
      infra/rate_limited class, because a model that never answered cannot have
      answered wrong.

    The work guards are deliberately the SAME ones the taxonomy already uses for
    the no-work case, so this predicate cannot resurrect a cell that those gates
    would exclude on their own.
    """
    row = row or {}
    if not bool(row.get("completed")):
        return False
    if row.get("checker_exit") not in (0, 1):
        return False
    # Affirmative evidence the model actually produced something, NOT merely a
    # truthy turn counter. has_zero_output_and_minimal_turns is deliberately
    # suppressed when the transcript carries text, and a provider error IS text
    # -- so a cell with turns=1, zero output and nothing but a 429 in its
    # transcript would satisfy that gate and be scored wrong_answer, which is
    # precisely the "never answered" case this predicate must reject.
    # No such row exists in the corpus today (of 81 rows this correction
    # promotes, none has turns<=1 with zero output), so this closes a hole
    # rather than repairing live data.
    produced_output = any(
        isinstance(row.get(field), (int, float)) and not isinstance(row.get(field), bool)
        and row.get(field) > 0
        for field in ("tokens_output", "tokens_proxy_output")
    )
    if not produced_output and not row.get("workspace_changed"):
        return False
    if has_zero_output_and_minimal_turns(row, text):
        return False
    return True


def has_near_zero_agent_tokens(row):
    """True when every reported agent/proxy token field is absent or below 100."""
    values = [row.get(field) for field in _TOKEN_FIELDS]
    return all(value in (None, 0) or (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and 0 <= value < NEAR_ZERO_TOKEN_LIMIT
    ) for value in values)


def _has_workspace_work_evidence(row):
    """Recognize explicit change summaries without treating pristine file lists as work."""
    if row.get("workspace_changed") is True:
        return True
    changes = row.get("workspace_changes")
    return bool(changes) if isinstance(changes, (dict, list, tuple, set, str)) else False


def _has_model_work_evidence(row, text):
    """Conservative evidence that a model ran despite missing token telemetry.

    Token parsers can fail for otherwise healthy adapters.  Turns, meaningful
    model/adapter output (10+ characters), or explicit workspace-change evidence
    therefore prevent the silent-failure reclassification.  Merely listing the
    checker workspace is not evidence because it includes pristine task files.
    """
    return (not has_near_zero_agent_tokens(row) or bool(row.get("turns"))
            or len(_meaningful_work_text(text)) >= 10
            or _has_workspace_work_evidence(row))


def _has_no_work_evidence(row, text):
    # Cap-rider handling predates the completed-run heuristic and intentionally
    # treats any nonzero aggregate token report as evidence of work. Proxy-
    # metered tokens count too: a BYO cell killed at the cap reports
    # tokens=None while the proxy saw hundreds of real calls -- that is a
    # genuine capability timeout, not a silent hang.
    any_tokens = any(row.get(field) for field in _TOKEN_FIELDS)
    return (not any_tokens and not row.get("turns")
            and len(_meaningful_work_text(text)) < 200)


def is_silent_no_model_call(row, adapter_output=""):
    """True for a completed, unsolved cell with no evidence the model ran.

    The zero-work classifier gate (``has_zero_output_and_minimal_turns``) is
    checked here for symmetry — a cell that produced zero output tokens in <=1
    turn is infra regardless of whether near-zero aggregate tokens or turns=1
    happens to satisfy the broader ``_has_model_work_evidence`` heuristic.
    """
    row = row or {}
    text = _text(adapter_output, row.get("output_tail"), row.get("error"))
    return (row.get("harness") != "null"
            and bool(row.get("completed")) and not bool(row.get("success"))
            and (has_zero_output_and_minimal_turns(row, text)
                 or not _has_model_work_evidence(row, text)))


def has_checker_crash(row):
    """True when the checker itself failed to execute (not a graded verdict).

    A checker communicates its verdict through exit 0 (pass) / 1 (fail), or the
    literal ``"timeout"`` sentinel. Any other exit code means the checker never
    reached a verdict — docker refusing to start it (125/126/127), a signal, or
    an interpreter crash. Those cells measure our infrastructure, not the model,
    so they must never be scored as ``wrong_answer``.
    """
    exit_code = (row or {}).get("checker_exit")
    if exit_code is None or exit_code == "timeout":
        return False
    try:
        return int(exit_code) not in (0, 1)
    except (TypeError, ValueError):
        return False


def classify_failure_reason(row, adapter_output=""):
    """Return a stable diagnostic reason without overriding stronger markers."""
    row = row or {}
    combined = _text(adapter_output, row.get("output_tail"), row.get("error"),
                     row.get("checker_exit"))
    if (bool(row.get("success")) or has_rate_limit_marker(combined)
            or has_infra_marker(combined) or has_instant_cli_exit_shape(row)):
        return None
    if is_silent_no_model_call(row, adapter_output):
        return "silent-no-model-call"
    return None


def classify_failure(row, adapter_output="", timeout_s=None):
    """Classify a benchmark row into the self-auditing failure taxonomy.

    ``adapter_output`` should be the full adapter stdout+stderr at write time;
    callers that only have an old row may pass ``output_tail`` instead.
    """
    row = row or {}
    combined = _text(
        adapter_output,
        row.get("output_tail"),
        row.get("error"),
        row.get("checker_exit"),
    )
    structured_status = _text(row.get("error"), row.get("checker_exit"))

    if bool(row.get("success")):
        return "solved"
    # Checked BEFORE the checker-owned verdict: these cells DO complete and the
    # checker DOES judge them, so the verdict rule would otherwise score them as
    # wrong_answer. A model given a 1-token budget never got to answer.
    if has_starved_output_budget(row):
        return "infra"
    # Also before the checker-owned verdict: a throttle-dominated cell has a
    # verdict about a starved run, not about the model.
    if has_throttle_dominated_replies(row):
        return "rate_limited"
    # A transient provider hiccup does NOT erase a verdict the checker already
    # reached. When the agent ran to completion, did real work, and the checker
    # exited with a real verdict (0/1), the checker owns the oracle -- fall
    # through to wrong_answer/timeout instead of excluding the cell.
    #
    # Without this, a single 429 anywhere in a long transcript excluded a cell
    # the checker had already FAILED on its merits. Campaign-wide that was 67 of
    # 90 rate_limited and 11 of 85 infra cells -- 78 real failures removed from
    # denominators, all with a checker FAIL message (e.g. "primer/vector overlap
    # mismatch"), median 11 turns and 15,885 output tokens, and not one of them a
    # hidden solve. It inflated rates in proportion to how often an arm got
    # throttled, so the thin-capacity models gained most (laguna 44 cells,
    # deepseek 13, kimi-k3 6, inkling 6, versus gpt-5.6 5) -- which flattered
    # exactly the comparison we were trying to measure.
    if has_checker_owned_verdict(row, combined):
        return "timeout" if row.get("checker_exit") == "timeout" else "wrong_answer"
    if has_rate_limit_marker(combined):
        return "rate_limited"
    if has_infra_marker(combined):
        return "infra"
    if (_ADAPTER_CRASH_RE.search(combined)
            and has_near_zero_agent_tokens(row) and not row.get("turns")):
        return "infra"
    if has_instant_cli_exit_shape(row):
        return "infra"
    if is_silent_no_model_call(row, adapter_output):
        return "infra"
    if has_checker_crash(row):
        return "infra"
    # Wall time riding the cap only means "timeout" when the runner killed the
    # agent; a CLI that exited on its own (completed=True) just ran slow.
    rode_cap = _wall_rode_cap(row, timeout_s) and not bool(row.get("completed"))
    # Cap-riding with zero work evidence (no tokens, no turns, empty transcript)
    # is a silent retry loop or hang, not a capability timeout.
    if rode_cap and _has_no_work_evidence(row, combined):
        return "infra"
    if row.get("checker_exit") == "timeout" or _TIMEOUT_RE.search(structured_status) or rode_cap:
        return "timeout"
    # After timeout handling: a measured incomplete run that did zero work (no
    # tokens, no turns, untouched workspace) and did not ride the cap is a
    # provider abandonment -- e.g. an OpenRouter overload retry loop that gave up
    # after minutes. The model never answered, so this is infra, not wrong_answer.
    if has_no_work_incomplete_shape(row, combined):
        return "infra"
    # wrong_answer must be checker-owned (exit 1). A row with NO checker exit
    # whose error is a Python traceback means the runner's own checker path
    # raised (evidence capture, scrub, or run_checker itself) -- that measures
    # our infrastructure, not the model. Narrow to tracebacks: plain error
    # strings (budget exhaustion, adapter messages) are legitimate unsolved
    # outcomes and must keep counting against the model.
    if row.get("checker_exit") is None and "Traceback" in str(row.get("error") or ""):
        return "infra"
    return "wrong_answer"


def class_for_report(row):
    """Return a row's failure_class, deriving one from saved fields if absent.

    Also CORRECTS a stored exclusion that the row's own fields refute. The
    write-time classifier sees the full transcript and used to let one 429
    anywhere in it override a verdict the checker had already reached; the
    transcript is not persisted, so re-running the classifier on a stored row
    cannot detect that -- no marker text survives to re-evaluate. The row's own
    fields do survive, and ``completed`` + ``checker_exit`` in {0,1} + real work
    is sufficient proof that the checker owned the oracle.

    Correcting here rather than rewriting the JSONLs keeps one source of truth
    for every reader (report, results-query) and fixes already-published
    datasets in place. 78 cells campaign-wide are affected, every one a real
    failure wrongly excluded, which inflated rates in proportion to how often an
    arm was throttled.
    """
    row = row or {}
    stored = row.get("failure_class")
    if stored in FAILURE_CLASSES:
        # A checker that could not run its verifier container never reached a
        # verdict: the imported tb2 checkers print this marker and exit 1 when
        # the reward file is missing (image absent -> docker exit 125, or the
        # verifier's /tmp log mount resolving VM-local under colima), which the
        # write path recorded as wrong_answer. The marker never appears on a
        # genuinely judged run -- the verifier writes reward.txt for pass AND
        # fail -- so it is proof of infra, not capability. 100% of tb2-tier
        # wide25 rows carried it (missing checker images on the mini).
        # This must take precedence over EVERY other correction, including the
        # checker-owned-verdict promotion below: a checker_exit of 1 that came
        # from the wrapper's own "verifier missing" branch is not a verdict, so
        # a stored-excluded row (e.g. rate_limited during a 429 storm) carrying
        # the marker must stay excluded rather than be promoted to wrong_answer.
        # (Found live: 8 pi x laguna tb2 cells stored rate_limited were being
        # promoted to phantom wrong-answers, skewing that arm's denominator.)
        if not row.get("success") \
                and "verifier did not produce" in (row.get("checker_stdout") or ""):
            return "infra"
        # Correct the other direction too: a stored VERDICT on a cell whose
        # request budget was starved is not a capability result. sampling_observed
        # is persisted, so unlike the 429 case this is recoverable from history.
        if stored not in EXCLUDED_FROM_SOLVE_RATE and not row.get("success") \
                and has_starved_output_budget(row):
            return "infra"
        if stored not in EXCLUDED_FROM_SOLVE_RATE and not row.get("success") \
                and has_throttle_dominated_replies(row):
            return "rate_limited"
        # Promotion must not resurrect a cell whose request budget was starved:
        # such a cell completes with a real checker_exit, so
        # has_checker_owned_verdict alone would flip the correctly-stored
        # exclusion back into a phantom wrong_answer. Throttle-domination is
        # deliberately NOT a gate here: the campaign's standing rule is that a
        # checker-owned verdict backed by affirmative work counts even when
        # most replies were throttled (recovered-with-tokens = genuine); the
        # per-cell contamination call is made by rerunning, not classifying.
        if stored in EXCLUDED_FROM_SOLVE_RATE and has_checker_owned_verdict(row) \
                and not has_starved_output_budget(row):
            return "timeout" if row.get("checker_exit") == "timeout" else "wrong_answer"
        return stored
    return classify_failure(row, row.get("output_tail") or "")


def is_excluded_from_solve_rate(row):
    return class_for_report(row) in EXCLUDED_FROM_SOLVE_RATE
