# Codex 0.153.0 completed-turn usage

This usage-only event was captured on September 5, 2026, from a Codex CLI 0.153.0
run requesting `gpt-5.6-sol` at medium effort. It contains no transcript or account
identifiers. The event reports `cache_write_input_tokens: 0`; the field is
available, rather than unknown.

Source: [original public event](https://github.com/sjh9714/astra-sweetspot/blob/937a6f82d75404eca2d6b95c637cd7583b82c2b4/results/receipts/2026-09-05T06-24-04-221Z-sol-medium-48beb0/usage-event.json)
and [receipt](https://github.com/sjh9714/astra-sweetspot/blob/937a6f82d75404eca2d6b95c637cd7583b82c2b4/results/receipts/2026-09-05T06-24-04-221Z-sol-medium-48beb0/receipt.json).
Published under MIT. The served model identifier was not independently exposed.

The regression replays this event through the legacy adapter with the child
process mocked and an empty temporary Codex home. Its missing-field and positive
cache-write variants are synthetic schema checks, not additional model runs.
