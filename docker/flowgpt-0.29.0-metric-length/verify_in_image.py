#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Smoke-check the flowgpt metric-length overlay inside the image."""

from __future__ import annotations

import sys

from vllm.config.observability import ObservabilityConfig
from vllm.v1.metrics.loggers import (
    build_1_2_5_buckets,
    build_request_token_length_buckets,
)


def main() -> None:
    cfg = ObservabilityConfig(
        request_prompt_token_length_buckets=[6000, 6500],
        request_generation_token_length_buckets=[125, 250],
    )
    assert cfg.request_prompt_token_length_buckets == [6000, 6500]
    assert cfg.request_generation_token_length_buckets == [125, 250]

    base = build_1_2_5_buckets(8192)
    prompt_buckets = build_request_token_length_buckets(8192, [6000, 6500])
    generation_buckets = build_request_token_length_buckets(8192, [125, 250])
    assert prompt_buckets == sorted(set(base) | {6000, 6500})
    assert generation_buckets == sorted(set(base) | {125, 250})

    from vllm.engine.arg_utils import EngineArgs

    assert "request_prompt_token_length_buckets" in EngineArgs.__dataclass_fields__
    assert "request_generation_token_length_buckets" in EngineArgs.__dataclass_fields__
    assert "request_token_length_buckets" not in EngineArgs.__dataclass_fields__

    print("VERIFY_OK", prompt_buckets[-3:], generation_buckets[-3:])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VERIFY_FAIL {exc}", file=sys.stderr)
        raise
