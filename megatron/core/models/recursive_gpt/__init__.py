# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Recursive GPT models with iterative refinement.

This package implements recursive refinement for GPT models, where the same
network is applied iteratively to refine latent and output representations.

Key components:
    RecursiveGPTModel: Main model class that wraps GPTModel with recursive refinement.
    RecursiveTransformerConfig: Configuration dataclass with recursive-specific parameters.

Example usage:
    >>> from megatron.core.models.recursive_gpt import RecursiveGPTModel, RecursiveTransformerConfig
    >>> from megatron.core.models.gpt import GPTModel
    >>>
    >>> config = RecursiveTransformerConfig(
    ...     num_layers=12,
    ...     hidden_size=768,
    ...     num_attention_heads=12,
    ...     num_refinement_blocks=3,
    ...     num_latent_refinements=6,
    ... )
    >>> gpt = GPTModel(config=config, ...)
    >>> recursive_model = RecursiveGPTModel(gpt_model=gpt, config=config)
    >>>
    >>> # Training
    >>> outputs = recursive_model(input_ids, labels=labels)
    >>> loss = outputs['loss']
    >>>
    >>> # Inference with adaptive halting
    >>> outputs = recursive_model.generate(input_ids, max_length=100)
"""

from megatron.core.models.recursive_gpt.recursive_gpt_config import RecursiveTransformerConfig
from megatron.core.models.recursive_gpt.recursive_gpt_model import RecursiveGPTModel

__all__ = [
    'RecursiveGPTModel',
    'RecursiveTransformerConfig',
]
