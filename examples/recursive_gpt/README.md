# Recursive GPT Training Examples

This directory contains training scripts and examples for Recursive GPT models with iterative refinement.

## Overview

Recursive GPT applies the same transformer network iteratively to refine latent and output representations. The model learns:
- **What to compute**: Through iterative refinement of latent states
- **When to stop**: Through adaptive halting based on prediction confidence

### Key Features

- **Iterative Refinement**: Apply the decoder multiple times with state updates
- **Adaptive Halting**: Learn when to stop refining during inference
- **Memory Efficient**: Optional gradient detaching between refinements
- **Compatible**: Works with Megatron's tensor and pipeline parallelism

## Architecture

The recursive refinement process:
1. Initialize output and latent states (zeros)
2. For T refinement blocks:
   - Refine latents n times: `latents = decoder(outputs + latents + inputs)`
   - Refine outputs once: `outputs = decoder(outputs + latents)`
3. Generate final predictions from outputs

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_refinement_blocks` | 3 | Number of deep refinement iterations (T) |
| `num_latent_refinements` | 6 | Latent refinements per output refinement (n) |
| `halt_loss_weight` | 1.0 | Weight for adaptive halting loss |
| `enable_recursive_refinement` | True | Toggle recursive mode on/off |
| `detach_between_refinements` | True | Detach gradients for memory efficiency |
| `max_inference_refinement_steps` | 12 | Max steps during inference |
| `halt_threshold` | 0.5 | Halting probability threshold |

## Quick Start

### Single GPU Training

```bash
python examples/recursive_gpt/pretrain_recursive_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 1.5e-4 \
    --train-iters 100000 \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --data-path /path/to/dataset \
    --split 949,50,1 \
    --num-refinement-blocks 3 \
    --num-latent-refinements 6 \
    --halt-loss-weight 1.0 \
    --detach-between-refinements
```

### Multi-GPU Training with Tensor Parallelism

```bash
torchrun --nproc_per_node=8 examples/recursive_gpt/pretrain_recursive_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --lr 1.5e-4 \
    --train-iters 500000 \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --data-path /path/to/dataset \
    --split 949,50,1 \
    --num-refinement-blocks 5 \
    --num-latent-refinements 8 \
    --halt-loss-weight 0.5
```

## Training Configurations

### Small Model (Testing)

- Layers: 6
- Hidden size: 512
- Attention heads: 8
- Refinement blocks: 2
- Latent refinements: 4
- Sequence length: 512
- **Use case**: Quick experiments, debugging

### Medium Model

- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Refinement blocks: 3
- Latent refinements: 6
- Sequence length: 1024
- **Use case**: Research, proof of concept

### Large Model

- Layers: 24
- Hidden size: 1024
- Attention heads: 16
- Refinement blocks: 5
- Latent refinements: 8
- Sequence length: 2048
- **Use case**: Production, benchmarking

## Advanced Usage

### Disabling Recursive Refinement

To compare recursive vs non-recursive training:

```bash
python examples/recursive_gpt/pretrain_recursive_gpt.py \
    --disable-recursive-refinement \
    [other args...]
```

This makes the model behave like standard GPT for ablation studies.

### Enabling Full Gradient Flow

For maximum expressiveness (at the cost of memory):

```bash
python examples/recursive_gpt/pretrain_recursive_gpt.py \
    --no-detach-between-refinements \
    [other args...]
```

This allows gradients to flow through all refinement blocks, not just the last one.

### Adjusting Halting Behavior

Control when the model stops refining during inference:

```bash
python examples/recursive_gpt/pretrain_recursive_gpt.py \
    --halt-threshold 0.7 \
    --max-inference-refinement-steps 20 \
    [other args...]
```

Higher thresholds = earlier stopping (faster but potentially lower quality).
Lower thresholds = more refinement (slower but potentially higher quality).

## Loss Function

The total loss combines two components:

```
total_loss = lm_loss + halt_loss_weight * halt_loss
```

- **lm_loss**: Standard cross-entropy language modeling loss
- **halt_loss**: Encourages model to stop when predictions are correct
- **halt_loss_weight**: Balance between LM accuracy and halting efficiency

## Memory Considerations

Recursive refinement requires more memory than standard GPT:

| Configuration | Memory Factor |
|---------------|---------------|
| `detach_between_refinements=True` | ~1.5x standard GPT |
| `detach_between_refinements=False` | ~(T × n)x standard GPT |

Where T = `num_refinement_blocks`, n = `num_latent_refinements`

### Recommendations

- Start with `detach_between_refinements=True`
- Use gradient accumulation for large models
- Reduce `micro_batch_size` if OOM
- Consider pipeline parallelism for very large models

## Monitoring Training

Key metrics to watch:

1. **LM Loss**: Should decrease steadily (language modeling quality)
2. **Halt Loss**: Should stabilize (adaptive halting learning)
3. **Total Loss**: Combined metric
4. **Refinement Steps** (inference): Average steps taken before halting

Example logging:

```
iteration      100/  100000 | consumed samples:          400 |
lm loss: 3.456 | halt loss: 0.234 | total loss: 3.690 |
learning rate: 1.500E-04 | global batch size:    32 |
```

## Troubleshooting

### OOM (Out of Memory)

1. Enable `--detach-between-refinements`
2. Reduce `--micro-batch-size`
3. Reduce `--num-refinement-blocks` or `--num-latent-refinements`
4. Enable activation checkpointing: `--recompute-activations`
5. Use tensor parallelism: `--tensor-model-parallel-size 2`

### Halt Loss Not Decreasing

1. Increase `--halt-loss-weight` (try 2.0 or 5.0)
2. Ensure labels are correct
3. Check if `enable_recursive_refinement=True`
4. Try training longer (halting may take time to learn)

### Performance Issues

1. Use `--detach-between-refinements` for speed
2. Reduce `--num-refinement-blocks` during training
3. Use adaptive halting during inference only
4. Profile with `--profile` flag

## Citation

If you use Recursive GPT in your research, please cite:

```bibtex
@article{recursive-gpt-2025,
  title={Recursive GPT: Iterative Refinement for Language Models},
  author={Your Team},
  journal={arXiv preprint},
  year={2025}
}
```

## Related Files

- `megatron/core/models/recursive_gpt/recursive_gpt_model.py` - Main model implementation
- `megatron/core/models/recursive_gpt/recursive_gpt_config.py` - Configuration
- `tests/unit_tests/models/test_recursive_gpt_model.py` - Unit tests

## Support

For questions and issues:
- Open an issue on GitHub
- Check the main Megatron-LM documentation
- Review the implementation guide: `RECURSIVE_GPT_IMPLEMENTATION_GUIDE.md`
