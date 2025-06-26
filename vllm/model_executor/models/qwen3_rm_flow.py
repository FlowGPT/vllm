# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# kaonai/Qwen3-0.6B-rm
# Copyright 2024 The KaonAI team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen3-RM model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import (
    IntermediateTensors,
    PoolerOutput,
    PoolingSequenceGroupOutput,
)

from .interfaces import SupportsLoRA, SupportsPP
from .qwen3 import Qwen3Model
from .utils import AutoWeightsLoader, maybe_prefix

from vllm.logger import init_logger

logger = init_logger(__name__)

logger.info("!!Qwen3RewardBaseModel!!")


class Qwen3RewardBaseModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen3Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.hidden_size = self.model.config.hidden_size

        self.score = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # for qwen3 model, hidden_states shape is [1, 1024]
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        # 对每个序列的最后一个 token 进行打分
        final_scores = []
        for state in hidden_states:
            # 取最后一个 token 的 hidden state
            last_token_state = state[-1:, :]  # shape: [1, 1024]
            # 对最后一个 token 进行打分
            score = self.score(last_token_state)  # shape: [1, score_dim]
            # 取出分数值
            final_scores.append(score[-1])  # 假设 score 输出是 [1, 1]

        outputs = [
            PoolingSequenceGroupOutput(data=final_scores[i])
            for i in range(len(final_scores))
        ]
        return PoolerOutput(outputs=outputs)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]),
        )
        return loader.load_weights(weights)
