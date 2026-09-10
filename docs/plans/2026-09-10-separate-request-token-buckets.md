# Separate Request Token Buckets Implementation Plan

> **For agents:** Use `executing-plans` (inline) or Task subagents per task. Steps use `- [ ]` checkboxes.

**Goal:** Configure prompt-token and generation-token Prometheus histogram buckets independently, with no legacy shared option.

**Architecture:** Split the existing observability field and CLI plumbing into two parallel fields, then build two bucket lists in `PrometheusStatLogger` and bind each to its matching histogram. Reuse the existing bucket merge helper and existing test files; do not change unrelated token histograms.

**Reference implementations:** The current `request_token_length_buckets` path in `vllm/config/observability.py`, `vllm/engine/arg_utils.py`, and `vllm/v1/metrics/loggers.py`; list-valued CLI tests in `tests/engine/test_arg_utils.py`.

---

## File map

- Modify `vllm/config/observability.py`: replace the shared field and validator with prompt and generation fields.
- Modify `vllm/engine/arg_utils.py`: expose, parse, and forward two CLI options.
- Modify `vllm/v1/metrics/loggers.py`: build and bind independent bucket lists.
- Modify `tests/v1/metrics/test_request_token_length_buckets.py`: cover validation, defaults, merging, and metric binding.
- Modify `tests/engine/test_arg_utils.py`: cover both new CLI options and removal of the old one.
- Modify `docker/flowgpt-0.29.0-metric-length/verify_in_image.py`: verify the split interface inside the overlay image.

## Task 1: Split configuration and CLI plumbing

- [x] Add one failing configuration test that constructs:

  ```python
  config = ObservabilityConfig(
      request_prompt_token_length_buckets=[6000],
      request_generation_token_length_buckets=[100, 200],
  )
  assert config.request_prompt_token_length_buckets == [6000]
  assert config.request_generation_token_length_buckets == [100, 200]
  ```

- [x] Implement only the two configuration fields and shared validator, then run
  that test to GREEN.

- [x] Add one parameterized failing validation test against both field names and
  assert `ValidationError` contains the matching field name; make the smallest
  validator adjustment needed and run it to GREEN.

- [x] Add one failing CLI test in `tests/engine/test_arg_utils.py` which parses:

  ```python
  [
      "--request-prompt-token-length-buckets", "6000", "6500",
      "--request-generation-token-length-buckets", "100", "200",
  ]
  ```

  Assert the two `EngineArgs` lists and the resulting `ObservabilityConfig` lists
  differ as supplied. Implement the two `EngineArgs` fields, parser entries, and
  forwarding assignments, then run the test to GREEN.

- [x] Add one failing old-option-removal test. With
  `FlexibleArgumentParser(exit_on_error=False)`, assert
  `--request-token-length-buckets 6000` raises `ArgumentError`; confirm it passes
  without adding a compatibility alias.

- [x] Before each minimal implementation above, run the newly added test and
  confirm it fails because the corresponding field or option does not exist:

  ```bash
  .venv/bin/python -m pytest \
    tests/v1/metrics/test_request_token_length_buckets.py \
    tests/engine/test_arg_utils.py -k 'request_token_length_buckets' -v
  ```

- [x] Use these exact fields and one validator:

  ```python
  request_prompt_token_length_buckets: list[int] = Field(default_factory=list)
  request_generation_token_length_buckets: list[int] = Field(default_factory=list)

  @field_validator(
      "request_prompt_token_length_buckets",
      "request_generation_token_length_buckets",
  )
  @classmethod
  def _validate_request_token_length_buckets(
      cls, value: list[int], info: ValidationInfo
  ) -> list[int]:
      if any(bucket <= 0 for bucket in value):
          raise ValueError(f"{info.field_name} values must be positive integers")
      return value
  ```

  Import `ValidationInfo` from Pydantic using the module's existing import style.

- [x] Re-run the focused tests and confirm the configuration and CLI cases pass.

## Task 2: Bind independent buckets to the two histograms

- [x] Add one failing logger-construction test using a minimal `SimpleNamespace`
  configuration with prompt extras `[6000]` and generation extras `[100, 200]`.
  Instantiate the real `PrometheusStatLogger`, render the public registry with
  `prometheus_client.generate_latest()`, and assert the exported `le` label sets
  contain `6000.0` only for the prompt metric and `100.0`/`200.0` only for the
  generation metric. The core public-output assertions are:

  ```python
  output = generate_latest().decode()
  assert 'vllm:request_prompt_tokens_bucket' in output
  assert 'le="6000.0"' in prompt_samples
  assert 'le="6000.0"' not in generation_samples
  assert 'le="100.0"' in generation_samples
  assert 'le="200.0"' in generation_samples
  ```

  Also assert an empty list on one side yields `build_1_2_5_buckets(8192)` even
  when the other side has extras.

- [x] Run the metric test and confirm it fails because both histograms still use
  the removed shared field:

  ```bash
  .venv/bin/python -m pytest \
    tests/v1/metrics/test_request_token_length_buckets.py -v
  ```

- [x] In `PrometheusStatLogger.__init__`, calculate:

  ```python
  request_prompt_token_buckets = build_request_token_length_buckets(
      max_model_len,
      vllm_config.observability_config.request_prompt_token_length_buckets,
  )
  request_generation_token_buckets = build_request_token_length_buckets(
      max_model_len,
      vllm_config.observability_config.request_generation_token_length_buckets,
  )
  ```

  Pass the first only to `vllm:request_prompt_tokens` and the second only to
  `vllm:request_generation_tokens`. Leave every histogram currently using
  `default_token_buckets` unchanged.

- [x] Re-run the metric test and confirm all cases pass.

## Task 3: Update overlay verification and run final checks

- [x] Change `verify_in_image.py` to construct `ObservabilityConfig` with distinct
  prompt and generation lists, verify both `EngineArgs.__dataclass_fields__`
  entries, and compare each merged result to its expected list. Remove all
  assertions for the old field.

- [x] Scan the changed source, tests, and overlay verifier for stale shared-field
  references; only the generic helper name may remain:

  ```bash
  rg -n "request_token_length_buckets|request-token-length-buckets" \
    vllm/config/observability.py vllm/engine/arg_utils.py \
    vllm/v1/metrics/loggers.py tests/v1/metrics \
    tests/engine/test_arg_utils.py \
    docker/flowgpt-0.29.0-metric-length/verify_in_image.py
  ```

- [x] Run the complete focused suites:

  ```bash
  .venv/bin/python -m pytest \
    tests/v1/metrics/test_request_token_length_buckets.py \
    tests/engine/test_arg_utils.py -v
  ```

- [x] Run lint on every changed Python file:

  ```bash
  pre-commit run ruff-check --files \
    vllm/config/observability.py vllm/engine/arg_utils.py \
    vllm/v1/metrics/loggers.py \
    tests/v1/metrics/test_request_token_length_buckets.py \
    tests/engine/test_arg_utils.py \
    docker/flowgpt-0.29.0-metric-length/verify_in_image.py
  ```

- [x] Review `git diff --check` and `git diff` to verify the change is limited to
  the approved interface split. Do not commit or push unless the user requests it.
