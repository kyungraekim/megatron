# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Recursive GPT model with iterative refinement."""

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.recursive_gpt.recursive_gpt_config import RecursiveTransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec


class RecursiveGPTModel(nn.Module):
    """GPT model with recursive refinement capability.

    This model wraps a standard GPTModel and adds recursive refinement where
    the network is applied iteratively to refine latent and output representations.

    Key Innovation:
        Instead of processing inputs through layers once, we:
        1. Initialize output and latent states
        2. Iteratively refine them by calling the same decoder multiple times
        3. Apply adaptive halting to stop when confident

    Architecture:
        inputs = embedding(tokens)  # Fixed throughout
        outputs, latents = initialize_states()

        for t in range(T):  # T refinement blocks
            for i in range(n):  # n latent refinements
                latents = decoder(outputs + latents + inputs)
            outputs = decoder(outputs + latents)

        logits = output_layer(outputs)

    Memory Efficiency:
        By detaching gradients between refinements (keeping only the last),
        we get memory footprint of single forward pass while doing T*(n+1)
        forward passes through the decoder.

    Args:
        config (RecursiveTransformerConfig): Model configuration
        transformer_layer_spec (ModuleSpec): Specification for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): Maximum sequence length
        pre_process (bool): Include embedding layer (for pipeline parallelism)
        post_process (bool): Include output layer (for pipeline parallelism)
        **kwargs: Additional arguments passed to GPTModel
    """

    def __init__(
        self,
        config: RecursiveTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process

        # Standard GPT model (embeddings + decoder + output layer)
        # We reuse this decoder multiple times for refinement
        self.gpt = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            **kwargs,
        )

        # Initial state parameters (learned)
        # These initialize the output and latent representations
        self.output_init_embed = nn.Parameter(torch.randn(config.hidden_size) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(config.hidden_size) * 1e-2)

        # Halt predictor: predicts when to stop refining
        # Maps hidden states -> probability of halting
        if post_process:
            pg_collection = kwargs.get('pg_collection')
            tp_group = pg_collection.tp if pg_collection and hasattr(pg_collection, 'tp') else None

            self.halt_predictor = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                1,
                config=config,
                bias=False,
                gather_output=True,  # Need full prediction across TP
                tp_group=tp_group,
            )
        else:
            self.halt_predictor = None

        # Cache configuration for convenience
        self.num_refinement_blocks = config.num_refinement_blocks
        self.num_latent_refinements = config.num_latent_refinements
        self.halt_loss_weight = config.halt_loss_weight
        self.enable_recursive = config.enable_recursive_refinement
        self.detach_between_refinements = config.detach_between_refinements

    def get_initial_state(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Initialize output and latent states.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Device to create tensors on

        Returns:
            outputs: Initial output state [seq_len, batch, hidden]
            latents: Initial latent state [seq_len, batch, hidden]
        """
        # Megatron uses [S, B, H] format (sequence-first)
        outputs = (
            self.output_init_embed.view(1, 1, -1)
            .expand(seq_len, batch_size, -1)
            .contiguous()
            .to(device)
        )

        latents = (
            self.latent_init_embed.view(1, 1, -1)
            .expand(seq_len, batch_size, -1)
            .contiguous()
            .to(device)
        )

        return outputs, latents

    def refine_once(
        self,
        inputs: Tensor,
        outputs: Tensor,
        latents: Tensor,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """One refinement cycle: refine latents n times, then outputs once.

        This is the core of recursive refinement. We call the decoder
        multiple times with different input combinations.

        Args:
            inputs: Embedded input tokens [S, B, H]
            outputs: Current output state [S, B, H]
            latents: Current latent state [S, B, H]
            attention_mask: Attention mask
            rotary_pos_emb: Rotary position embeddings (if using RoPE)
            rotary_pos_cos: Rotary cosine (for flash attention)
            rotary_pos_sin: Rotary sine (for flash attention)
            **kwargs: Additional arguments for decoder

        Returns:
            outputs: Refined output state [S, B, H]
            latents: Refined latent state [S, B, H]
        """
        # === Latent refinement (n times) ===
        # Each time, we combine outputs, latents, and inputs, then refine latents
        for _ in range(self.num_latent_refinements):
            combined = outputs + latents + inputs

            latents = self.gpt.decoder(
                hidden_states=combined,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                **kwargs,
            )

        # === Output refinement (once) ===
        # Combine outputs with refined latents
        combined = outputs + latents
        outputs = self.gpt.decoder(
            hidden_states=combined,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            **kwargs,
        )

        return outputs, latents

    def deep_refinement(
        self,
        inputs: Tensor,
        outputs: Tensor,
        latents: Tensor,
        attention_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Deep refinement: T rounds of refinement with gradient control.

        This performs T refinement blocks, where only the last block
        receives gradients (for memory efficiency).

        Args:
            inputs: Embedded input tokens [S, B, H]
            outputs: Initial output state [S, B, H]
            latents: Initial latent state [S, B, H]
            attention_mask: Attention mask
            rotary_pos_emb: Rotary position embeddings
            rotary_pos_cos: Rotary cosine
            rotary_pos_sin: Rotary sine
            **kwargs: Additional decoder arguments

        Returns:
            outputs: Final refined output [S, B, H]
            latents: Final refined latent [S, B, H]
        """
        for step in range(1, self.num_refinement_blocks + 1):
            is_last = step == self.num_refinement_blocks

            # Gradient control: only last refinement gets gradients
            if self.detach_between_refinements and not is_last:
                context = torch.no_grad()
            else:
                context = nullcontext()

            with context():
                outputs, latents = self.refine_once(
                    inputs=inputs,
                    outputs=outputs,
                    latents=latents,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    **kwargs,
                )

                # Explicitly detach to prevent gradient accumulation
                if self.detach_between_refinements and not is_last:
                    outputs = outputs.detach()
                    latents = latents.detach()

        return outputs, latents

    def compute_halt_prob(self, hidden_states: Tensor) -> Tensor:
        """Compute halting probability from hidden states.

        Args:
            hidden_states: Hidden states [S, B, H]

        Returns:
            halt_prob: Halting probability [B]
        """
        if self.halt_predictor is None:
            # No halt predictor (not in post-process stage)
            return torch.zeros(hidden_states.size(1), device=hidden_states.device)

        # Average over sequence dimension: [S, B, H] -> [B, H]
        pooled = hidden_states.mean(dim=0)

        # Project to scalar and sigmoid: [B, H] -> [B, 1] -> [B]
        halt_logit = self.halt_predictor(pooled)
        halt_prob = torch.sigmoid(halt_logit).squeeze(-1)

        return halt_prob

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        **kwargs,
    ):
        """Forward pass with optional recursive refinement.

        Args:
            input_ids: Input token IDs [B, S]
            position_ids: Position IDs [B, S]
            attention_mask: Attention mask
            labels: Labels for loss computation [B, S]
            **kwargs: Additional arguments

        Returns:
            If labels is None:
                logits: [B, S, V]
            If labels provided:
                (total_loss, (ce_loss, halt_loss), logits, halt_prob)
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device

        # === Step 1: Embed inputs ===
        # Use GPT's preprocessing to get embeddings and RoPE
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ) = self.gpt._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=None,
            inference_context=None,
            packed_seq_params=None,
        )

        # Ensure [S, B, H] format (Megatron standard)
        if decoder_input.dim() == 3 and decoder_input.size(0) == batch_size:
            # Input is [B, S, H], convert to [S, B, H]
            inputs = decoder_input.transpose(0, 1).contiguous()
        else:
            inputs = decoder_input

        # === Step 2: Recursive refinement or standard forward ===
        if not self.enable_recursive or not self.training:
            # Standard GPT forward pass (no recursion)
            hidden_states = self.gpt.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                **kwargs,
            )
        else:
            # Initialize states
            outputs, latents = self.get_initial_state(batch_size, seq_len, device)

            # Deep refinement
            outputs, latents = self.deep_refinement(
                inputs=inputs,
                outputs=outputs,
                latents=latents,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                **kwargs,
            )

            hidden_states = outputs

        # === Step 3: Generate logits ===
        # Use GPT's output layer
        logits, _ = self.gpt.output_layer(hidden_states)

        # Convert to [B, S, V] format
        if logits.size(0) == seq_len:
            # [S, B, V] -> [B, S, V]
            logits = logits.transpose(0, 1).contiguous()

        # === Step 4: Compute halt probability ===
        if self.enable_recursive and self.training and self.halt_predictor is not None:
            # Ensure [S, B, H] format for halt computation
            if hidden_states.size(0) != seq_len:
                hidden_states_for_halt = hidden_states.transpose(0, 1)
            else:
                hidden_states_for_halt = hidden_states

            halt_prob = self.compute_halt_prob(hidden_states_for_halt)
        else:
            halt_prob = torch.zeros(batch_size, device=device)

        # === Step 5: Return or compute losses ===
        if labels is None:
            return logits

        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size), labels.reshape(-1), reduction='none'
        )
        ce_loss = ce_loss.view(batch_size, seq_len).mean(dim=1)  # [B]

        # Halt loss: encourage halting when all predictions are correct
        if self.enable_recursive and self.training and self.halt_predictor is not None:
            is_all_correct = (logits.argmax(dim=-1) == labels).all(dim=-1).float()  # [B]
            halt_loss = F.binary_cross_entropy(halt_prob, is_all_correct, reduction='none')
        else:
            halt_loss = torch.zeros_like(ce_loss)

        # Total loss
        total_loss = (ce_loss + halt_loss * self.halt_loss_weight).sum()

        return total_loss, (ce_loss, halt_loss), logits, halt_prob

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        max_refinement_steps: Optional[int] = None,
        halt_threshold: Optional[float] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Generation with adaptive halting.

        Args:
            input_ids: Input token IDs [B, S]
            position_ids: Position IDs [B, S]
            attention_mask: Attention mask
            max_refinement_steps: Maximum refinement steps (default: from config)
            halt_threshold: Halting threshold (default: from config)
            **kwargs: Additional arguments

        Returns:
            pred_ids: Predicted token IDs [B, S]
            exit_steps: Refinement step at which each sample halted [B]
        """
        if max_refinement_steps is None:
            max_refinement_steps = self.config.max_inference_refinement_steps
        if halt_threshold is None:
            halt_threshold = self.config.halt_threshold

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        device = input_ids.device

        # Embed inputs
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            _,
        ) = self.gpt._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=None,
            inference_context=None,
            packed_seq_params=None,
        )

        # Ensure [S, B, H] format
        if decoder_input.dim() == 3 and decoder_input.size(0) == batch_size:
            inputs = decoder_input.transpose(0, 1).contiguous()
        else:
            inputs = decoder_input

        # Initialize states
        outputs, latents = self.get_initial_state(batch_size, seq_len, device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        final_logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        exit_steps = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Adaptive refinement loop
        for step in range(1, max_refinement_steps + 1):
            if not active_mask.any():
                break

            # Refine only active samples
            active_inputs = inputs[:, active_mask, :]
            active_outputs = outputs[:, active_mask, :]
            active_latents = latents[:, active_mask, :]
            active_attention_mask = (
                attention_mask[active_mask] if attention_mask is not None else None
            )

            # One full deep refinement
            for _ in range(self.num_refinement_blocks):
                active_outputs, active_latents = self.refine_once(
                    inputs=active_inputs,
                    outputs=active_outputs,
                    latents=active_latents,
                    attention_mask=active_attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    **kwargs,
                )

            # Compute halt probability
            halt_prob = self.compute_halt_prob(active_outputs)
            should_halt = (halt_prob >= halt_threshold) | (step == max_refinement_steps)

            # Store results for halted samples
            if should_halt.any():
                halted_indices = torch.where(active_mask)[0][should_halt]

                # Get logits for halted samples
                halted_outputs = active_outputs[:, should_halt, :]
                halted_logits, _ = self.gpt.output_layer(halted_outputs)
                halted_logits = halted_logits.transpose(0, 1).contiguous()  # [S, B', V] -> [B', S, V]

                final_logits[halted_indices] = halted_logits
                exit_steps[halted_indices] = step

                # Update active mask
                active_indices = torch.where(active_mask)[0]
                active_mask[active_indices[should_halt]] = False

            # Update states for next iteration
            if active_mask.any() and step < max_refinement_steps:
                outputs[:, active_mask, :] = active_outputs[:, ~should_halt, :]
                latents[:, active_mask, :] = active_latents[:, ~should_halt, :]

        pred_ids = final_logits.argmax(dim=-1)
        return pred_ids, exit_steps
