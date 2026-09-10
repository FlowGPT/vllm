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
    cfg = ObservabilityConfig(request_token_length_buckets=[6000, 6500])
    assert cfg.request_token_length_buckets == [6000, 6500]

    base = build_1_2_5_buckets(8192)
    merged = build_request_token_length_buckets(8192, [6000, 6500])
    assert merged == sorted(set(base) | {6000, 6500}), (base, merged)

    # Only prompt/generation histograms use the helper; import must succeed.
    from vllm.engine.arg_utils import EngineArgs

    assert "request_token_length_buckets" in EngineArgs.__dataclass_fields__

    print("VERIFY_OK", merged[-3:])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VERIFY_FAIL {exc}", file=sys.stderr)
        raise
