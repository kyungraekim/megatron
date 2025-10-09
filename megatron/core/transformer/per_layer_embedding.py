# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class PerLayerEmbeddingSubmodules:
    """
    Submodules for per-layer embedding.

    Per-layer embeddings add a gated residual connection with layer-specific
    learned embeddings to enhance model capacity.

    Architecture:
        input -> per_layer_input_gate -> activation -> element-wise multiply with layer_embedding
              -> per_layer_projection -> add to input -> output

    Args:
        per_layer_input_gate: Column parallel linear layer that projects from hidden_size
            to hidden_size_per_layer_input
        per_layer_projection: Row parallel linear layer that projects from
            hidden_size_per_layer_input back to hidden_size
    """

    per_layer_input_gate: Union[ModuleSpec, type] = None
    per_layer_projection: Union[ModuleSpec, type] = None


class PerLayerEmbedding(MegatronModule):
    """
    Per-layer embedding module that adds layer-specific learned embeddings.

    This module implements a gated residual connection where the input is projected,
    activated, multiplied element-wise with a layer-specific embedding, projected back,
    and added to the original input as a residual connection.

    The layer embedding is a vocabulary parallel embedding that is indexed using input_ids
    and provides layer-specific contextual information.

    Returns:
        output (Tensor): Transformed output with per-layer embedding applied
        bias (None): Always returns None for bias to maintain API consistency
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: PerLayerEmbeddingSubmodules,
        vocab_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """
        Initialize per-layer embedding module.

        Args:
            config: Transformer configuration
            submodules: Submodule specifications for gate and projection
            vocab_size: Vocabulary size for the layer embedding
            tp_group: Tensor parallel process group
        """
        super().__init__(config=config)

        self.config = config

        # Validate configuration
        if config.hidden_size_per_layer_input is None:
            raise ValueError(
                "hidden_size_per_layer_input must be set in config when using per-layer embeddings"
            )

        # Layer-specific vocabulary embedding
        self.layer_embedding = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=config.hidden_size_per_layer_input,
            init_method=config.init_method,
            config=config,
        )

        # Gate projection: hidden_size -> hidden_size_per_layer_input
        self.per_layer_input_gate = build_module(
            submodules.per_layer_input_gate,
            config.hidden_size,
            config.hidden_size_per_layer_input,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="per_layer_input_gate",
            tp_group=tp_group,
        )

        # Activation function from config
        self.activation_func = config.activation_func

        # Output projection: hidden_size_per_layer_input -> hidden_size
        self.per_layer_projection = build_module(
            submodules.per_layer_projection,
            config.hidden_size_per_layer_input,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="per_layer_projection",
            tp_group=tp_group,
        )

    def forward(self, mlp_output: torch.Tensor, layer_embedding: torch.Tensor):
        """
        Apply per-layer embedding transformation.

        Args:
            mlp_output: Output from MLP layer, shape [s, b, h]
            layer_embedding: Layer-specific embedding, shape [s, b, hidden_size_per_layer_input]

        Returns:
            output: Transformed output, shape [s, b, h]
            bias: Always None (for API consistency)
        """
        # Project to intermediate dimension
        # [s, b, h] -> [s, b, hidden_size_per_layer_input]
        gated_output, gate_bias = self.per_layer_input_gate(mlp_output)

        # Skip bias addition since we're using skip_bias_add=True
        # The bias should be None based on config validation
        assert gate_bias is None, "Per-layer embeddings do not support bias"

        # Apply activation and element-wise multiply with layer embedding
        # [s, b, hidden_size_per_layer_input]
        activated = self.activation_func(gated_output)
        gated_with_embedding = activated.mul(layer_embedding)

        # Project back to hidden size
        # [s, b, hidden_size_per_layer_input] -> [s, b, h]
        projected_output, proj_bias = self.per_layer_projection(gated_with_embedding)

        # Skip bias addition
        assert proj_bias is None, "Per-layer embeddings do not support bias"

        # Add residual connection
        output = projected_output + mlp_output

        return output, None

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary for distributed checkpointing."""
        sharded_state_dict = {}

        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            sharded_state_dict.update(sub_sd)

        return sharded_state_dict
