# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# kaonai/Qwen3-0.6B-rm
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
"""Inference-only Qwen3-RM model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Optional, Union

from huggingface_hub import hf_hub_download
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .qwen3 import Qwen3Model
from .utils import maybe_prefix


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

        self.score_model_pt_path = hf_hub_download(
            repo_id=config.name_or_path, filename="reward_head.pt"
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
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )  # [B, L, H]
        eos_mask = input_ids == self.config.eos_token_id
        eos_index = eos_mask.int().argmax(dim=1)
        eos_hidden = hidden_states[
            torch.arange(input_ids.size(0)), eos_index
        ]  # [B, H]
        logits = self.score(eos_hidden)  # [B, 1]
        return logits.squeeze(-1)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        super().load_weights(weights)
        self.score.load_state_dict(
            torch.load(self.score_model_pt_path), map_location="cuda"
        )
