# CLAUDE.md - optimizer

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Distributed optimizer: ZeRO-style sharding, gradient clipping, mixed precision
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

## Overview

This directory implements **distributed optimizers** for training large models efficiently on
multiple GPUs. It provides ZeRO-style optimizer state sharding, gradient clipping, mixed
precision training, and CPU offloading to reduce memory overhead.

The core problem: Optimizer state (e.g., momentum in Adam) scales with model size. For a 70B
parameter model with Adam (2 states per param), this adds ~560GB of GPU memory. ZeRO-style
sharding partitions optimizer state across GPUs, reducing per-GPU memory from O(M) to O(M/N)
where N is the number of GPUs.

**Key concepts**:
- **DistributedOptimizer**: Shards optimizer state across data-parallel ranks using ZeRO-2 or
  ZeRO-3 strategies
- **MegatronOptimizer**: Wrapper around PyTorch/Apex optimizers (FusedAdam, FusedSGD) with loss
  scaling and gradient clipping
- **Loss Scaling**: ConstantGradScaler (fixed) and DynamicGradScaler (auto-adjust) for mixed
  precision training
- **Gradient Clipping**: Global norm clipping with mixed-precision awareness
- **CPU Offloading**: Optional CPU-GPU hybrid state for ultra-large models

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `optimizer.py` | 1419 | MegatronOptimizer wrapper, training loop integration | MegatronOptimizer, get_megatron_optimizer |
| `distrib_optimizer.py` | 2603 | ZeRO-style distributed optimizer state sharding | DistributedOptimizer |
| `optimizer_config.py` | 375 | Configuration dataclass with all optimizer knobs | OptimizerConfig |
| `clip_grads.py` | 247 | Gradient clipping utilities | clip_grad_norm_fp32, clip_grad_norm_fp16 |
| `grad_scaler.py` | 142 | Mixed precision loss scaling | ConstantGradScaler, DynamicGradScaler |
| `cpu_offloading/hybrid_optimizer.py` | 472 | CPU-GPU hybrid optimizer for extreme scale | HybridOptimizer |

## Architecture

**Component Hierarchy**:
```
MegatronOptimizer (wrapper, loss scaling, clipping)
    ↓
DistributedOptimizer (ZeRO state sharding)
    ↓
PyTorch/Apex Optimizer (FusedAdam, FusedSGD)
    ↓
GPU/CPU Parameters
```

**Data Flow** (typical training step):
1. Forward pass → compute loss
2. Loss scaling: `scaled_loss = loss * loss_scale` (mixed precision)
3. Backward: `scaled_loss.backward()` → gradients
4. AllReduce: sync gradients across data-parallel ranks
5. Gradient clipping: `clip_grad_norm_fp32(parameters, max_norm)`
6. Optimizer step:
   - DistributedOptimizer gathers required params from other ranks (ZeRO-3)
   - Updates parameter shards with fused kernels
   - Synchronizes updated params back to all ranks
7. Loss scale adjustment: increase if no overflow, decrease if overflow

**Design Patterns**:
- **Config-driven**: All settings in OptimizerConfig dataclass
- **Modularity**: Loss scaling, clipping, sharding independent
- **Compatibility**: Wraps PyTorch/Apex optimizers (pluggable)
- **FSDP integration**: Works with PyTorch FSDP for multi-node scaling

## Common Tasks

**Initialize optimizer with ZeRO state sharding**:
```python
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

config = OptimizerConfig(
    optimizer='adam',
    lr=1e-4,
    use_distributed_optimizer=True,  # Enable ZeRO sharding
)
optimizer = get_megatron_optimizer(model, config)
```

**Enable dynamic loss scaling for mixed precision**:
```python
config = OptimizerConfig(
    use_loss_scaling=True,
    loss_scale_window=1000,
    loss_scale=2**16,  # Initial scale
)
```

**Clip gradients by global norm**:
```python
from megatron.core.optimizer import clip_grad_norm_fp32

total_norm = clip_grad_norm_fp32(model.parameters(), max_norm=1.0)
```

**Use CPU offloading** (models > 200B):
```python
config = OptimizerConfig(
    use_distributed_optimizer=True,
    cpu_offload=True,
)
```

**Access loss scale for logging**:
```python
if optimizer.grad_scaler:
    print(f"Loss scale: {optimizer.grad_scaler.loss_scale}")
```

## Dependencies

**Depends on**:
- `megatron.core.parallel_state` - Process group management, rank queries
- `megatron.core.tensor_parallel` - Parameter utilities
- `transformer_engine` (optional) - FusedAdam kernels
- `torch.optim`, `apex.optimizers` - Underlying optimizers

**Used by**:
- `megatron.training.training` - Main training loop calls optimizer.step()
- `megatron.training.arguments` - CLI flags parsed into OptimizerConfig
- Training scripts (e.g., `examples/pretrain_gpt.py`)

## Gotchas

1. **ZeRO-3 requires AllReduce before loss.backward()**: DistributedOptimizer handles this,
   but custom loops must call `optimizer.synchronize()` first.

2. **Loss scaling mismatch**: If using `torch.autocast()` or manual loss scaling, ensure value
   matches DynamicGradScaler's internal state.

3. **CPU offloading adds latency**: `cpu_offload=True` trades memory for compute. Only use for
   models > 200B.

4. **Gradient clipping does AllReduce**: Computing global norm is ~1-5% of step time. Disable
   if not needed.

5. **Distributed optimizer requires DP > 1**: ZeRO sharding only useful with multi-GPU DP.
   Single-GPU training falls back to standard optimizer.

## See Also

- `megatron/core/dist_checkpointing/` - Checkpointing optimizer state
- `megatron/training/arguments.py` - CLI argument definitions
- `examples/pretrain_gpt.py` - Full training loop example
