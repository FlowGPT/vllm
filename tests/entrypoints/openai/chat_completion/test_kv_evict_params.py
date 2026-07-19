# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry-layer mapping for truncation-aware KV eviction.

The scheduler reads conversation_id/truncated off kv_transfer_params (see
Scheduler._maybe_evict_truncated_prefix). These keys are derived from the
X-Flow-Conversation-Id header and the enable_kv_evict body flag, never the
client body. Serving passes the header value into to_sampling_params.
"""

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def _make_request(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            **extra,
        }
    )


def _kv_transfer_params(request: ChatCompletionRequest, conversation_id):
    params = request.to_sampling_params(
        max_tokens=16,
        default_sampling_params={},
        conversation_id=conversation_id,
    )
    if params.extra_args is None:
        return None
    return params.extra_args.get("kv_transfer_params")


def test_header_plus_flag_marks_truncated():
    request = _make_request(enable_kv_evict=True)
    kv = _kv_transfer_params(request, "conv-1")
    assert kv == {"conversation_id": "conv-1", "truncated": True}


def test_header_without_flag_records_only_conversation_id():
    # Non-truncated turn: conversation_id is still needed so the scheduler can
    # record this turn's hash chain as the next turn's baseline, but no
    # truncated flag means no eviction is triggered.
    request = _make_request(enable_kv_evict=False)
    kv = _kv_transfer_params(request, "conv-1")
    assert kv == {"conversation_id": "conv-1"}


def test_no_header_injects_nothing():
    # Missing X-Flow-Conversation-Id: eviction cannot run, so the flag alone
    # must not add any kv_transfer_params.
    request = _make_request(enable_kv_evict=True)
    assert _kv_transfer_params(request, None) is None


def test_client_cannot_forge_eviction_target():
    # Old path removed: conversation_id/truncated in the client body are
    # stripped; only the header/flag drive eviction.
    request = _make_request(
        kv_transfer_params={"conversation_id": "spoofed", "truncated": True}
    )
    assert _kv_transfer_params(request, None) is None


def test_disaggregated_fields_preserved_and_merged():
    request = _make_request(
        kv_transfer_params={"do_remote_prefill": True},
        enable_kv_evict=True,
    )
    kv = _kv_transfer_params(request, "conv-1")
    assert kv == {
        "do_remote_prefill": True,
        "conversation_id": "conv-1",
        "truncated": True,
    }
