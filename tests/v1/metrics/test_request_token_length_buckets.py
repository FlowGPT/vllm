# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.metrics.loggers import (
    build_1_2_5_buckets,
    build_request_token_length_buckets,
)


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
