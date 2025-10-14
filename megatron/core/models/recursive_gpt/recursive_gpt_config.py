# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Configuration for Recursive GPT models with iterative refinement."""

from dataclasses import dataclass

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class RecursiveTransformerConfig(TransformerConfig):
    """Configuration for recursive refinement GPT models.

    This extends TransformerConfig with parameters specific to recursive
    refinement, where a network is applied iteratively to refine latent
    and output representations.

    The recursive refinement process:
    - Initialize output and latent states
    - For T refinement blocks:
        - Refine latents n times: latents = network(outputs + latents + inputs)
        - Refine outputs once: outputs = network(outputs + latents)
    - Generate final predictions from outputs

    Args:
        num_refinement_blocks (int): Number of deep refinement iterations (T in paper).
            Each refinement block consists of n latent refinements followed by
            one output refinement. Default: 3.
        num_latent_refinements (int): Number of latent refinements per output
            refinement (n in paper). Default: 6.
        halt_loss_weight (float): Weight for adaptive halting loss. The halting
            loss encourages the model to stop refining when predictions are correct.
            Default: 1.0.
        enable_recursive_refinement (bool): Toggle recursive refinement mode.
            If False, the model behaves like a standard GPT. This is useful for
            comparing recursive vs non-recursive training. Default: True.
        detach_between_refinements (bool): If True, only the last refinement block
            receives gradients (memory efficient). If False, gradients flow through
            all refinements (more expressive but memory intensive). Default: True.
        max_inference_refinement_steps (int): Maximum number of refinement steps
            during inference with adaptive halting. Default: 12.
        halt_threshold (float): Halting probability threshold for early exit during
            inference. Samples with halt_prob >= threshold stop refining. Default: 0.5.
    """

    # Recursive refinement parameters
    num_refinement_blocks: int = 3
    """Number of deep refinement iterations (T in paper)."""

    num_latent_refinements: int = 6
    """Latent refinements per output refinement (n in paper)."""

    halt_loss_weight: float = 1.0
    """Weight for adaptive halting loss."""

    enable_recursive_refinement: bool = True
    """Toggle recursive mode. If False, behaves like standard GPT."""

    detach_between_refinements: bool = True
    """If True, only last refinement gets gradients (memory efficient).
    If False, gradients flow through all refinements (more expressive)."""

    max_inference_refinement_steps: int = 12
    """Maximum refinement steps during inference."""

    halt_threshold: float = 0.5
    """Halting probability threshold for early exit."""

    def __post_init__(self):
        """Validate recursive-specific configuration."""
        super().__post_init__()

        if self.num_refinement_blocks < 1:
            raise ValueError(
                f"num_refinement_blocks must be >= 1, got {self.num_refinement_blocks}"
            )

        if self.num_latent_refinements < 1:
            raise ValueError(
                f"num_latent_refinements must be >= 1, got {self.num_latent_refinements}"
            )

        if self.halt_loss_weight < 0:
            raise ValueError(f"halt_loss_weight must be >= 0, got {self.halt_loss_weight}")

        if not 0.0 <= self.halt_threshold <= 1.0:
            raise ValueError(
                f"halt_threshold must be in [0, 1], got {self.halt_threshold}"
            )

        if self.max_inference_refinement_steps < 1:
            raise ValueError(
                f"max_inference_refinement_steps must be >= 1, "
                f"got {self.max_inference_refinement_steps}"
            )