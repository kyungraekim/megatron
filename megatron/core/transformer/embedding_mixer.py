# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, save_to_aux_losses_tracker
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none


@dataclass
class EmbeddingMixerSubmodules:
    """
    Configuration class for specifying the submodules of an EmbeddingMixer.

    This class defines the down-projection and up-projection linear layers
    for the embedding mixer module.

    Args:
        down_proj (Union[ModuleSpec, type]): Specification for the down-projection layer
            (hidden_size → latent_size).
        up_proj (Union[ModuleSpec, type]): Specification for the up-projection layer
            (latent_size → hidden_size).
    """

    down_proj: Union[ModuleSpec, type] = None
    up_proj: Union[ModuleSpec, type] = None


class EmbeddingMixer(MegatronModule):
    """
    Per-token embedding selection and mixing module.

    This module runs alongside the MLP as an additional residual block in the
    transformer layer. For each token, it:
    1. Down-projects hidden states to a latent dimension
    2. Computes gate logits for routing to embeddings
    3. Selects top-k embeddings and performs soft weighted sum
    4. Mixes (element-wise multiplies) the down-projected states with selected embeddings
    5. Up-projects back to hidden dimension

    The module returns (output, bias) following the pattern of other transformer sublayers.

    Args:
        config (TransformerConfig): Transformer configuration
        submodules (EmbeddingMixerSubmodules): Specifications for down/up projection layers
        layer_number (Optional[int]): Layer number for identification, defaults to None
        tp_group (Optional[torch.distributed.ProcessGroup]): Tensor parallel group
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: EmbeddingMixerSubmodules,
        layer_number: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=False)

        # Extract configuration parameters
        latent_size = config.embedding_mixer_latent_size
        num_embeddings = config.embedding_mixer_num_embeddings
        topk = config.embedding_mixer_topk
        hidden_size = config.hidden_size

        if latent_size is None:
            raise ValueError(
                "embedding_mixer_latent_size must not be None when creating an EmbeddingMixer"
            )

        if topk > num_embeddings:
            raise ValueError(
                f"embedding_mixer_topk ({topk}) cannot exceed embedding_mixer_num_embeddings ({num_embeddings})"
            )

        # Build down-projection layer (hidden_size → latent_size)
        self.down_proj = build_module(
            submodules.down_proj,
            input_size=hidden_size,
            output_size=latent_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False,
            tp_group=self.tp_group,
        )

        # Gate linear layer for routing (hidden_size → num_embeddings)
        # Not TP-sharded, duplicated across TP ranks
        self.gate = nn.Linear(hidden_size, num_embeddings, bias=False)
        if config.init_method is not None:
            config.init_method(self.gate.weight)

        # Learned embedding table (not TP-sharded)
        self.embedding_table = nn.Embedding(num_embeddings, latent_size)
        if config.init_method is not None:
            config.init_method(self.embedding_table.weight)

        # Build up-projection layer (latent_size → hidden_size)
        self.up_proj = build_module(
            submodules.up_proj,
            input_size=latent_size,
            output_size=hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            tp_group=self.tp_group,
        )

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the embedding mixer.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: A tuple containing:
                - output: Tensor of shape [s, b, h] (mixed hidden states)
                - bias: Bias tensor if add_bias_linear is True, else None
        """

        s, b, h = hidden_states.shape
        topk = self.config.embedding_mixer_topk
        num_embeddings = self.config.embedding_mixer_num_embeddings

        # Down-project: [s, b, h] → [s, b, d] where d is latent_size
        down_output_with_bias = self.down_proj(hidden_states)
        if isinstance(down_output_with_bias, tuple):
            latent_proj, _ = down_output_with_bias
        else:
            latent_proj = down_output_with_bias

        # Compute gate logits for routing: [s, b, h] → [s*b, E]
        # Flatten for gating: [s*b, h]
        hidden_flat = hidden_states.view(-1, h)
        gate_logits = self.gate(hidden_flat)  # [s*b, E]

        # ── Z-loss ──────────────────────────────────────────────────────────────
        z_loss_coeff = self.config.embedding_mixer_z_loss_coeff
        if self.training and z_loss_coeff is not None and z_loss_coeff > 0:
            z_loss = torch.mean(torch.square(torch.logsumexp(gate_logits, dim=-1))) * z_loss_coeff
            save_to_aux_losses_tracker(
                "emb_mixer_z_loss",
                z_loss / z_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
            gate_logits = MoEAuxLossAutoScaler.apply(gate_logits, z_loss)

        # Full softmax over all E embeddings (needed for aux loss P_i)
        router_probs = torch.softmax(gate_logits, dim=-1)  # [s*b, E]

        # ── Load-balance aux loss ────────────────────────────────────────────────
        aux_loss_coeff = self.config.embedding_mixer_aux_loss_coeff
        if self.training and aux_loss_coeff > 0:
            total_tokens = s * b

            # routing_map: one-hot [s*b, E] indicating which embeddings are selected
            topk_indices_for_map = torch.topk(gate_logits.detach(), k=topk, dim=-1).indices
            routing_map = torch.zeros_like(router_probs).scatter_(1, topk_indices_for_map, 1.0)

            # switch_load_balancing_loss_func formula:
            #   loss = coeff * E * sum(aggregated_probs * tokens_per_embedding)
            #              / (topk * N * N)
            aggregated_probs = router_probs.sum(dim=0)  # [E], sum of P_i over tokens
            tokens_per_embedding = routing_map.sum(dim=0)  # [E], count of tokens routed to i

            aux_loss = (
                num_embeddings
                * aux_loss_coeff
                * torch.sum(aggregated_probs * tokens_per_embedding)
                / (topk * total_tokens * total_tokens)
            )
            save_to_aux_losses_tracker(
                "emb_mixer_load_balance_loss",
                aux_loss / aux_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
            # Inject aux_loss gradient through router_probs (which flows to output)
            router_probs = MoEAuxLossAutoScaler.apply(router_probs, aux_loss)

        # ── Top-k routing ────────────────────────────────────────────────────────
        topk_values = router_probs.gather(1, torch.topk(gate_logits.detach(), k=topk, dim=-1).indices)
        topk_indices = torch.topk(gate_logits.detach(), k=topk, dim=-1).indices
        mixture_weights = torch.softmax(topk_values, dim=-1)  # [s*b, topk]

        # Gather embeddings for selected indices
        # embedding_table.weight: [E, d]
        selected_embeddings = self.embedding_table.weight[topk_indices]  # [s*b, topk, d]

        # Weighted sum of selected embeddings: [s*b, topk, d] → [s*b, d]
        weighted_embedding = torch.sum(
            mixture_weights.unsqueeze(-1) * selected_embeddings, dim=1
        )  # [s*b, d]

        # Reshape back to sequence/batch format
        latent_size = latent_proj.shape[-1]
        weighted_embedding = weighted_embedding.view(s, b, latent_size)  # [s, b, d]

        # Element-wise multiply: down-projected states with selected embedding
        mixed_latent = latent_proj * weighted_embedding

        # Up-project back to hidden size: [s, b, d] → [s, b, h]
        up_output_with_bias = self.up_proj(mixed_latent)

        return up_output_with_bias
