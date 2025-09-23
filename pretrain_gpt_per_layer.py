#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT with Per-Layer Embedding support."""

import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

# Import our per-layer embedding components
from per_layer_embedding import PerLayerEmbedding, PerLayerProjection, PerLayerGate


stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model with per-layer embedding support.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    print_rank_0('building GPT model with per-layer embedding ...')

    # NEW: Check if per-layer embedding is enabled
    if getattr(args, 'use_per_layer_embedding', False):
        print_rank_0('Per-layer embedding enabled!')
        config = core_transformer_config_from_args(get_args())

        # Add per-layer embedding configuration
        config.use_per_layer_embedding = True
        config.per_layer_vocab_size = args.per_layer_vocab_size
        config.per_layer_hidden_size = args.per_layer_hidden_size

        # Use custom layer spec that includes per-layer components
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=args.moe_grouped_gemm,
                qk_layernorm=args.qk_layernorm,
                fp8_input_store=args.fp8_input_store
            )
        else:
            transformer_layer_spec = get_per_layer_gpt_layer_spec()  # NEW: Custom spec with per-layer support
    else:
        config = core_transformer_config_from_args(get_args())

        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=args.num_experts,
                moe_grouped_gemm=args.moe_grouped_gemm,
                qk_layernorm=args.qk_layernorm,
                fp8_input_store=args.fp8_input_store
            )
        else:
            transformer_layer_spec = get_gpt_layer_local_spec()

    # Build the model
    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    else:
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
            rotary_base=args.rotary_base,
        )

    # NEW: If per-layer embedding enabled, wrap the model
    if getattr(args, 'use_per_layer_embedding', False):
        model = PerLayerGPTModelWrapper(model, config, args)

    return model


class PerLayerGPTModelWrapper(torch.nn.Module):
    """
    Wrapper that adds per-layer embedding functionality to GPTModel
    """

    def __init__(self, gpt_model, config, args):
        super().__init__()
        self.gpt_model = gpt_model
        self.config = config
        self.args = args

        # Per-layer embedding components
        self.per_layer_embedding = PerLayerEmbedding(
            config=config,
            vocab_size_per_layer=args.per_layer_vocab_size,
            hidden_size_per_layer=args.per_layer_hidden_size,
            num_layers=config.num_layers,
        )

        self.per_layer_projection = PerLayerProjection(
            config=config,
            hidden_size_per_layer=args.per_layer_hidden_size,
            num_layers=config.num_layers,
        )

        # Per-layer gates for each transformer layer
        self.per_layer_gates = torch.nn.ModuleList([
            PerLayerGate(
                config=config,
                hidden_size_per_layer=args.per_layer_hidden_size,
                layer_idx=layer_idx,
            )
            for layer_idx in range(config.num_layers)
        ])

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                per_layer_input_ids=None, labels=None, **kwargs):
        """Forward pass with per-layer embedding integration"""

        # Standard token embeddings
        if hasattr(self.gpt_model, 'embedding'):
            token_embeds = self.gpt_model.embedding(input_ids)
        else:
            # For newer GPT models, embedding might be part of the transformer
            token_embeds = input_ids  # Will be handled by the transformer

        # Per-layer embeddings (if provided)
        if per_layer_input_ids is not None:
            per_layer_embeds = self.per_layer_embedding(per_layer_input_ids)

            # Combine with main embeddings if we have token embeddings
            if isinstance(token_embeds, torch.Tensor):
                combined_per_layer = self.per_layer_projection(token_embeds, per_layer_embeds)
            else:
                # Handle case where embeddings are computed inside transformer
                combined_per_layer = per_layer_embeds
        else:
            combined_per_layer = None

        # Modified forward pass through transformer layers
        if combined_per_layer is not None:
            # We need to modify the transformer forward pass to include per-layer processing
            # This is a simplified version - real implementation would modify transformer layers
            return self._forward_with_per_layer(
                input_ids, position_ids, attention_mask, combined_per_layer, labels, **kwargs
            )
        else:
            # Standard forward pass
            return self.gpt_model(input_ids, position_ids, attention_mask, labels=labels, **kwargs)

    def _forward_with_per_layer(self, input_ids, position_ids, attention_mask,
                               combined_per_layer, labels, **kwargs):
        """Forward pass with per-layer processing integrated"""
        # This is a simplified implementation
        # In practice, you'd need to modify the transformer layers directly

        # For now, run standard model and then apply per-layer processing
        output = self.gpt_model(input_ids, position_ids, attention_mask, labels=labels, **kwargs)

        # Apply per-layer gates (simplified - in practice this would be integrated into layers)
        if hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
            for layer_idx, gate in enumerate(self.per_layer_gates):
                layer_per_layer_input = combined_per_layer[:, :, layer_idx, :]
                hidden_states = gate(hidden_states, layer_per_layer_input)

            # Update output
            if hasattr(output, '_replace'):
                output = output._replace(last_hidden_state=hidden_states)

        return output


def get_per_layer_gpt_layer_spec():
    """Get layer spec that includes per-layer embedding support"""
    # This would extend the standard layer spec to include per-layer components
    # For now, use the standard spec
    return get_gpt_layer_local_spec()


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets with per-layer support."""
    args = get_args()

    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')

    # Standard dataset building
    if args.mock_data:
        dataset_config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=get_blend_and_blend_per_split(args.data_path, args.train_data_path),
            mock=True,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

        if train_val_test_num_samples[0] > 0:
            train_ds = MockGPTDataset(dataset_config, "train", train_val_test_num_samples[0])

        if train_val_test_num_samples[1] > 0:
            valid_ds = MockGPTDataset(dataset_config, "valid", train_val_test_num_samples[1])

        if train_val_test_num_samples[2] > 0:
            test_ds = MockGPTDataset(dataset_config, "test", train_val_test_num_samples[2])

    else:
        # Use real data
        dataset_config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=get_blend_and_blend_per_split(args.data_path, args.train_data_path),
            blend_per_split=[
                get_blend_and_blend_per_split(args.train_data_path, args.train_data_path),
                get_blend_and_blend_per_split(args.valid_data_path, args.valid_data_path),
                get_blend_and_blend_per_split(args.test_data_path, args.test_data_path)
            ],
            split=args.split,
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=get_tokenizer(),
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

        # NEW: If per-layer embedding enabled, we need dual datasets
        if getattr(args, 'use_per_layer_embedding', False) and hasattr(args, 'per_layer_data_path'):
            print_rank_0('Building datasets with per-layer embedding support...')
            # Build per-layer dataset alongside standard dataset
            train_ds, valid_ds, test_ds = build_dual_datasets(
                dataset_config, train_val_test_num_samples, args
            )
        else:
            # Standard dataset building
            if any(train_val_test_num_samples):
                builder = BlendedMegatronDatasetBuilder(
                    GPTDataset,
                    train_val_test_num_samples,
                    dataset_config,
                )
                train_ds, valid_ds, test_ds = builder.build()

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds


def build_dual_datasets(dataset_config, train_val_test_num_samples, args):
    """Build datasets that include both standard and per-layer tokens"""

    # This is a placeholder implementation
    # In practice, you'd need to create a custom dataset class that:
    # 1. Loads both standard and per-layer tokenized data
    # 2. Ensures alignment between the two token streams
    # 3. Returns batches with both input_ids and per_layer_input_ids

    print_rank_0('WARNING: Dual dataset building not fully implemented. Using standard datasets.')

    if any(train_val_test_num_samples):
        builder = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            dataset_config,
        )
        train_ds, valid_ds, test_ds = builder.build()
    else:
        train_ds, valid_ds, test_ds = None, None, None

    return train_ds, valid_ds, test_ds


def forward_step(data_iterator, model):
    """Forward training step with per-layer embedding support."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    keys = ['text']

    # NEW: Add per-layer keys if enabled
    if getattr(args, 'use_per_layer_embedding', False):
        keys.append('per_layer_text')

    datatype = args.fp16

    # Micro batch.
    data_b = get_batch_on_this_cp_rank(data_iterator)
    data_i = get_batch_on_this_tp_rank(data_b)

    tokens = data_i['text'].long()

    # NEW: Get per-layer tokens if available
    per_layer_tokens = None
    if 'per_layer_text' in data_i:
        per_layer_tokens = data_i['per_layer_text'].long()

    labels = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()

    if per_layer_tokens is not None:
        per_layer_tokens = per_layer_tokens[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    timers('batch-generator').stop()

    # Forward pass through the model.
    if getattr(args, 'use_per_layer_embedding', False) and per_layer_tokens is not None:
        output_tensor = model(tokens, position_ids, attention_mask,
                            per_layer_input_ids=per_layer_tokens, labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def loss_func(loss_mask, output_tensor):
    """Loss function."""
    args = get_args()

    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()
        output_tensor = output_tensor[:, :args.curriculum_seqlen, :].contiguous()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids,
                                   reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""
    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                               device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder,
             forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})