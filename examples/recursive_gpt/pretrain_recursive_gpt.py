#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain Recursive GPT with iterative refinement.

This script trains a Recursive GPT model that applies the same transformer
network iteratively to refine latent and output representations. The model
learns both what to compute and when to stop refining through adaptive halting.

Example usage:
    # Single GPU training
    python examples/recursive_gpt/pretrain_recursive_gpt.py \\
        --num-layers 12 \\
        --hidden-size 768 \\
        --num-attention-heads 12 \\
        --seq-length 1024 \\
        --max-position-embeddings 1024 \\
        --micro-batch-size 4 \\
        --global-batch-size 32 \\
        --lr 1.5e-4 \\
        --train-iters 100000 \\
        --vocab-file /path/to/vocab.json \\
        --merge-file /path/to/merges.txt \\
        --data-path /path/to/dataset \\
        --split 949,50,1 \\
        --num-refinement-blocks 3 \\
        --num-latent-refinements 6 \\
        --halt-loss-weight 1.0 \\
        --detach-between-refinements

    # Multi-GPU training with tensor and pipeline parallelism
    torchrun --nproc_per_node=8 examples/recursive_gpt/pretrain_recursive_gpt.py \\
        --tensor-model-parallel-size 2 \\
        --pipeline-model-parallel-size 4 \\
        --num-layers 24 \\
        --hidden-size 1024 \\
        --num-attention-heads 16 \\
        --seq-length 2048 \\
        --max-position-embeddings 2048 \\
        --micro-batch-size 2 \\
        --global-batch-size 128 \\
        --lr 1.5e-4 \\
        --train-iters 500000 \\
        --vocab-file /path/to/vocab.json \\
        --merge-file /path/to/merges.txt \\
        --data-path /path/to/dataset \\
        --split 949,50,1 \\
        --num-refinement-blocks 5 \\
        --num-latent-refinements 8 \\
        --halt-loss-weight 0.5
"""

import torch
from functools import partial
from typing import Optional

from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.recursive_gpt import RecursiveGPTModel, RecursiveTransformerConfig
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)


def recursive_transformer_config_from_args(args):
    """Create RecursiveTransformerConfig from command line arguments.

    Extends the standard TransformerConfig with recursive-specific parameters.
    """
    # Get base transformer config
    base_config_dict = vars(core_transformer_config_from_args(args))

    # Add recursive-specific parameters
    recursive_config = RecursiveTransformerConfig(
        **base_config_dict,
        num_refinement_blocks=args.num_refinement_blocks,
        num_latent_refinements=args.num_latent_refinements,
        halt_loss_weight=args.halt_loss_weight,
        enable_recursive_refinement=args.enable_recursive_refinement,
        detach_between_refinements=args.detach_between_refinements,
        max_inference_refinement_steps=args.max_inference_refinement_steps,
        halt_threshold=args.halt_threshold,
    )

    return recursive_config


def model_provider(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
    """Build the RecursiveGPTModel.

    Args:
        pre_process: Include embedding layer (first stage in pipeline parallelism)
        post_process: Include output layer (last stage in pipeline parallelism)
        add_encoder: Include encoder (not used for GPT, kept for compatibility)
        add_decoder: Include decoder (transformer layers)

    Returns:
        RecursiveGPTModel instance
    """
    args = get_args()

    print_rank_0('Building Recursive GPT model ...')

    # Create recursive transformer config
    config = recursive_transformer_config_from_args(args)

    # Build base GPT model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    # Wrap with RecursiveGPTModel
    recursive_model = RecursiveGPTModel(gpt_model=gpt_model, config=config)

    print_rank_0(f'Recursive GPT Model Configuration:')
    print_rank_0(f'  - Refinement blocks: {config.num_refinement_blocks}')
    print_rank_0(f'  - Latent refinements per block: {config.num_latent_refinements}')
    print_rank_0(f'  - Halt loss weight: {config.halt_loss_weight}')
    print_rank_0(f'  - Detach between refinements: {config.detach_between_refinements}')
    print_rank_0(f'  - Enable recursive refinement: {config.enable_recursive_refinement}')

    return recursive_model


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch from the data iterator.

    Args:
        data_iterator: Iterator over training data
        vp_stage: Virtual pipeline stage (for interleaved pipeline parallelism)

    Returns:
        Tuple of (tokens, labels, loss_mask, attention_mask, position_ids)
    """
    # Only first and last pipeline stages need data
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # Get batches based on the TP rank
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[RecursiveGPTModel] = None):
    """Compute loss for Recursive GPT.

    The output_tensor from RecursiveGPTModel contains both the language modeling
    losses and the halting losses. This function combines them appropriately.

    Args:
        loss_mask: Mask for valid tokens (ignore padding)
        output_tensor: Output from the model containing losses
        model: The RecursiveGPTModel instance (for accessing config)

    Returns:
        Tuple of (total_loss, num_tokens, reporting_dict)
    """
    args = get_args()

    # If output_tensor is a dict (from RecursiveGPTModel), extract lm_loss and halt_loss
    if isinstance(output_tensor, dict):
        lm_losses = output_tensor['lm_loss'].view(-1).float()
        halt_losses = output_tensor.get('halt_loss', torch.tensor(0.0, device=lm_losses.device))

        # Apply loss mask to language modeling loss
        loss_mask = loss_mask.view(-1).float()
        lm_loss = torch.sum(lm_losses * loss_mask)

        # Halt loss is already averaged, just scale it
        halt_loss = halt_losses.mean() if halt_losses.numel() > 1 else halt_losses

        # Combine losses
        total_loss = lm_loss + args.halt_loss_weight * halt_loss

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)

        # Reporting metrics
        reporting_loss = torch.cat([
            lm_loss.clone().detach().view(1),
            halt_loss.clone().detach().view(1),
            total_loss.clone().detach().view(1),
            num_tokens.view(1)
        ])

        return (
            total_loss,
            num_tokens,
            {
                'lm loss': reporting_loss[:1],
                'halt loss': reporting_loss[1:2],
                'total loss': reporting_loss[2:3],
            }
        )
    else:
        # Fallback for standard output tensor (shouldn't happen with RecursiveGPTModel)
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

        return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: RecursiveGPTModel):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model: The RecursiveGPTModel

    Returns:
        Tuple of (output_tensor, loss_func_partial)
    """
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    # Forward pass through RecursiveGPTModel
    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    return output, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    """Determine if dataset should be built on this rank."""
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def core_gpt_dataset_config_from_args(args):
    """Create GPTDatasetConfig from command line arguments."""
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build train, validation, and test datasets.

    Args:
        train_val_test_num_samples: List of [train_samples, valid_samples, test_samples]
        vp_stage: Virtual pipeline stage

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    dataset_type = MockGPTDataset if args.mock_data else GPTDataset

    print_rank_0('> Building train, validation, and test datasets for Recursive GPT ...')

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        partial(is_dataset_built_on_rank, vp_stage=vp_stage),
        config,
    ).build()

    print_rank_0('> Finished creating Recursive GPT datasets ...')

    return train_ds, valid_ds, test_ds


def add_recursive_gpt_args(parser):
    """Add Recursive GPT specific arguments to the parser.

    Args:
        parser: ArgumentParser instance

    Returns:
        parser with added recursive GPT arguments
    """
    group = parser.add_argument_group(title='Recursive GPT')

    group.add_argument(
        '--num-refinement-blocks',
        type=int,
        default=3,
        help='Number of deep refinement iterations (T in paper). '
        'Each block consists of n latent refinements followed by one output refinement.',
    )

    group.add_argument(
        '--num-latent-refinements',
        type=int,
        default=6,
        help='Number of latent refinements per output refinement (n in paper). '
        'Higher values allow more iterative reasoning.',
    )

    group.add_argument(
        '--halt-loss-weight',
        type=float,
        default=1.0,
        help='Weight for adaptive halting loss. This encourages the model to learn '
        'when to stop refining based on prediction confidence.',
    )

    group.add_argument(
        '--enable-recursive-refinement',
        action='store_true',
        default=True,
        help='Enable recursive refinement mode. If False, model behaves like standard GPT.',
    )

    group.add_argument(
        '--disable-recursive-refinement',
        action='store_false',
        dest='enable_recursive_refinement',
        help='Disable recursive refinement (fall back to standard GPT behavior).',
    )

    group.add_argument(
        '--detach-between-refinements',
        action='store_true',
        default=True,
        help='Detach gradients between refinement blocks (memory efficient). '
        'Only the last refinement block receives gradients.',
    )

    group.add_argument(
        '--no-detach-between-refinements',
        action='store_false',
        dest='detach_between_refinements',
        help='Allow gradients to flow through all refinement blocks (more expressive but memory intensive).',
    )

    group.add_argument(
        '--max-inference-refinement-steps',
        type=int,
        default=12,
        help='Maximum number of refinement steps during inference with adaptive halting.',
    )

    group.add_argument(
        '--halt-threshold',
        type=float,
        default=0.5,
        help='Halting probability threshold for early exit during inference. '
        'Samples with halt_prob >= threshold stop refining.',
    )

    return parser


if __name__ == "__main__":
    # Mark that datasets are distributed
    train_valid_test_datasets_provider.is_distributed = True

    # Run pretraining
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_recursive_gpt_args,
    )
