# Separate Request Token Buckets Design

## Goal

Allow operators to configure Prometheus histogram bucket boundaries for prompt
token counts independently from generation token counts.

## Interface

Replace the shared `request_token_length_buckets` setting and
`--request-token-length-buckets` CLI option with two required-by-name,
independently optional settings:

- `request_prompt_token_length_buckets` exposed as
  `--request-prompt-token-length-buckets`
- `request_generation_token_length_buckets` exposed as
  `--request-generation-token-length-buckets`

Both settings default to an empty list. The old setting and CLI option are
removed without a compatibility alias or fallback.

## Configuration Flow

`EngineArgs` exposes both lists and passes them into matching fields on
`ObservabilityConfig`. `PrometheusStatLogger` merges each list independently
with the existing 1-2-5 token buckets. The prompt result initializes only
`vllm:request_prompt_tokens`; the generation result initializes only
`vllm:request_generation_tokens`.

The existing `build_request_token_length_buckets` helper remains shared because
the merge algorithm is identical. No inference, scheduling, sampling, metric
name, or unrelated token histogram changes are in scope.

## Validation and Defaults

Each field independently rejects zero and negative bucket boundaries. Custom
boundaries are sorted, deduplicated, and limited to `max_model_len`, preserving
the current behavior. An empty list on either side yields the standard 1-2-5
buckets for that side regardless of the other setting.

## Verification

Tests cover independent configuration validation, CLI parsing, removal of the
old CLI option, the existing merge behavior, and metric construction with
different prompt and generation buckets. The thin-overlay image verifier checks
that both new `ObservabilityConfig` and `EngineArgs` fields are present and
independent.
