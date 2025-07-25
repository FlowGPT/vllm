# SPDX-License-Identifier: Apache-2.0
from vllm.v1.worker.gpu_input_batch import InputBatch
import os


def is_spec_decode_supported(req_id: str, input_batch: InputBatch) -> bool:
    if req_id in input_batch.min_p_reqs:
        # Spec decode doesn't support min_p sampling.
        return os.environ.get("VLLM_SPEC_DECODE_MIN_P_SUPPORTED", "false").lower() == "true"
    elif (req_id in input_batch.frequency_penalties_reqs
          or req_id in input_batch.presence_penalties_reqs
          or req_id in input_batch.repetition_penalties_reqs):
        # Spec decode doesn't support penalties.
        return False
    elif req_id in input_batch.num_logprobs:
        # Spec decode doesn't support logprobs.
        return False

    return True
