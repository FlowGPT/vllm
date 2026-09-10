# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from prometheus_client import generate_latest
from pydantic import ValidationError

from vllm.config.observability import ObservabilityConfig
from vllm.v1.metrics.loggers import (
    PrometheusStatLogger,
    build_1_2_5_buckets,
    build_request_token_length_buckets,
)


def test_prompt_and_generation_bucket_configurations_are_independent():
    config = ObservabilityConfig(
        request_prompt_token_length_buckets=[6000],
        request_generation_token_length_buckets=[100, 200],
    )

    assert config.request_prompt_token_length_buckets == [6000]
    assert config.request_generation_token_length_buckets == [100, 200]


@pytest.mark.parametrize(
    "field_name",
    [
        "request_prompt_token_length_buckets",
        "request_generation_token_length_buckets",
    ],
)
@pytest.mark.parametrize("invalid_bucket", [0, -1])
def test_request_token_bucket_configurations_reject_non_positive_values(
    field_name, invalid_bucket
):
    with pytest.raises(ValidationError, match=field_name):
        ObservabilityConfig(**{field_name: [invalid_bucket]})


def _export_request_token_histogram_samples(prompt_buckets, generation_buckets):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=8192,
            is_diffusion=False,
        ),
        observability_config=ObservabilityConfig(
            request_prompt_token_length_buckets=prompt_buckets,
            request_generation_token_length_buckets=generation_buckets,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
    )

    PrometheusStatLogger(config)
    output = generate_latest().decode()
    prompt_samples = "\n".join(
        line
        for line in output.splitlines()
        if line.startswith("vllm:request_prompt_tokens_bucket")
    )
    generation_samples = "\n".join(
        line
        for line in output.splitlines()
        if line.startswith("vllm:request_generation_tokens_bucket")
    )
    return prompt_samples, generation_samples


def test_prompt_and_generation_histograms_export_independent_buckets():
    prompt_samples, generation_samples = _export_request_token_histogram_samples(
        [6000], [125, 250]
    )

    assert 'le="6000.0"' in prompt_samples
    assert 'le="6000.0"' not in generation_samples
    assert 'le="125.0"' in generation_samples
    assert 'le="250.0"' in generation_samples
    assert 'le="125.0"' not in prompt_samples
    assert 'le="250.0"' not in prompt_samples


def test_empty_prompt_bucket_configuration_keeps_prompt_defaults():
    prompt_samples, generation_samples = _export_request_token_histogram_samples(
        [], [125]
    )

    assert 'le="5000.0"' in prompt_samples
    assert 'le="125.0"' not in prompt_samples
    assert 'le="125.0"' in generation_samples


def test_default_matches_1_2_5_buckets():
    for max_value in (8192, 32768):
        assert build_request_token_length_buckets(
            max_value, None
        ) == build_1_2_5_buckets(max_value)
        assert build_request_token_length_buckets(max_value, []) == build_1_2_5_buckets(
            max_value
        )


def test_extra_bucket_inserted_and_sorted():
    buckets = build_request_token_length_buckets(8192, [6000])
    assert buckets == [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        6000,
    ]


def test_extra_buckets_deduped_and_clipped_to_max_model_len():
    buckets = build_request_token_length_buckets(8192, [6000, 5000, 9000, 6500, 6000])
    assert buckets == [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        6000,
        6500,
    ]
