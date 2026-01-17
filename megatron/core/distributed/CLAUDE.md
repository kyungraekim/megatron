# CLAUDE.md - distributed

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Data parallelism: DDP, FSDP, ZeRO-1/2/3, and gradient buffer management
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/distributed/` provides data parallelism implementations for distributed
training: DistributedDataParallel (DDP), Fully Sharded Data Parallel (FSDP), and ZeRO
optimizer states. This directory handles gradient synchronization, parameter sharding,
and memory-efficient training across multiple GPUs.

**Problem Solved**: Scaling training across GPU nodes requires efficient gradient
aggregation and parameter management. Naive all-reduce becomes a bottleneck. ZeRO partitions
optimizer states (Stage 1), gradients (Stage 2), and parameters (Stage 3) across DP ranks
to reduce memory by N_DP. Custom FSDP is ~15% faster than PyTorch's implementation via
optimized collective communication patterns and FP8 support.

**Key Concepts**:
- **Gradient Buffers**: Contiguous grad storage per bucket for efficient collectives
- **ZeRO Sharding**: Partition optimizer / gradients / params across DP ranks
- **Overlap**: Async all-reduce/reduce-scatter during backward for compute/comm overlap
- **FSDP**: Automatic parameter sharding (ZeRO-3) with all-gather/reduce-scatter
- **FP8 Support**: Reduced-precision grad communication with optional FP32 accumulation

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `param_and_grad_buffer.py` | 1016 | Buffer management, bucketing, collective ops | `_ParamAndGradBuffer`, `partition_buckets()` |
| `distributed_data_parallel.py` | 592 | Main DDP wrapper, grad sync, bucketing | `DistributedDataParallel` |
| `finalize_model_grads.py` | 488 | Allreduce/reduce-scatter for final gradients | `finalize_model_grads()` |
| `fsdp/mcore_fsdp_adapter.py` | 435 | Custom FSDP wrapper, ~15% faster than PyTorch | `FullyShardedDataParallel` |
| `distributed_data_parallel_config.py` | 171 | Config: grad reduce, overlap, ZeRO, FP8 | `DistributedDataParallelConfig` |
| `torch_fully_sharded_data_parallel.py` | 154 | PyTorch FSDP compatibility wrapper | `TorchFullyShardedDataParallel` |
| `reduce_scatter_with_fp32_accumulation.py` | 92 | Optimize reduce-scatter with FP32 accumulation | `reduce_scatter_with_fp32_accumulation()` |
| `data_parallel_base.py` | 96 | Abstract base class template | `_BaseDataParallel` |
| `fsdp/` | - | Custom FSDP implementation (12+ files) | - |

---

## Architecture

**Parallelism Modes**:

```
DDP (Data Parallel)           FSDP (Fully Sharded)
┌─────────────────┐          ┌────────────────────┐
│ Rank 0: Model   │          │ Rank 0: 1/N Params │
│ Rank 1: Model   │ AllReduce │ Rank 1: 1/N Params │ AllGather
│ Rank N: Model   │ -------> │ Rank N: 1/N Params │ -------->
└─────────────────┘          └────────────────────┘
   Broadcast                     ReduceScatter
   Shared params                 Distributed opt
```

**ZeRO Stages**:

1. **ZeRO-1 (Optimizer States)**: Partition optimizer state (Adam moments) across DP ranks
2. **ZeRO-2**: + Partition gradients across DP ranks (reduce-scatter instead of all-reduce)
3. **ZeRO-3**: + Partition model parameters across DP ranks (FSDP, all-gather on forward)

**Gradient Flow** (with bucketing & overlap):

```
1. Forward Pass                    2. Backward Pass
   Model (TP shards)                  Grad accumulation in buffer
          |                                 |
   Loss Computation              3. Start Async All-Reduce/Reduce-Scatter
                                    (per bucket, async if overlap=True)
                                          |
                                   4. Finish Grad Sync
                                    (wait for collectives)
                                          |
                                   5. Optimizer Step
```

**Buffer Hierarchy** (per DP rank):

```
_ParamAndGradBuffer (full model gradients)
  └─ _ParamAndGradBucket[0]  (first N params)
  └─ _ParamAndGradBucket[1]  (next M params)
  └─ _ParamAndGradBucket[k]  (reduce-scatter'd collectively)
```

---

## Common Tasks

### 1. Initialize DDP with Config

```python
from megatron.core.distributed import (
    DistributedDataParallel, DistributedDataParallelConfig
)
from megatron.core import parallel_state

ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,  # Async grad all-reduce
    use_distributed_optimizer=False,  # Standard all-reduce (ZeRO-1)
    grad_reduce_in_fp32=True,  # Accumulate grads in FP32
)
model = DistributedDataParallel(config, ddp_config, module=model)
```

### 2. Enable ZeRO-2/3 (Reduce-Scatter + Parameter Sharding)

```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,  # ZeRO-2+
    data_parallel_sharding_strategy='optim_grads',  # ZeRO-2
    overlap_grad_reduce=True,
    overlap_param_gather=True,  # ZeRO-3: gather params on forward
)
```

### 3. Use Custom FSDP (~15% Faster)

```python
from megatron.core.distributed import FullyShardedDataParallel

fsdp_config = DistributedDataParallelConfig(
    use_megatron_fsdp=True,  # Use custom FSDP
    data_parallel_sharding_strategy='optim_grads_params',  # Full sharding
)
model = FullyShardedDataParallel(config, fsdp_config, module=model)
```

### 4. FP8 Gradient Communication (Reduced Precision)

```python
ddp_config = DistributedDataParallelConfig(
    reduce_scatter_with_fp32_accumulation=True,  # Send FP8, accumulate FP32
)
# Optional: use NCCL userbuffer for perf
ddp_config.nccl_ub = True  # Register NCCL user buffer
```

### 5. Gradient Scaling & Checks

```python
model.zero_grad_buffer()  # Zero gradients at step start
# ... backward ...
model.scale_gradients(1.0 / gradient_accumulation_steps)  # Scale if needed
model.start_grad_sync()  # Initiate async all-reduce (if overlap=True)
# ... can continue with next iteration ...
model.finish_grad_sync()  # Wait for collectives to complete
```

---

## Dependencies

**Depends On** (imports from):
- `../parallel_state` - Process group initialization (DP, TP, PP groups)
- `../transformer/module` - `MegatronModule` base class
- `../transformer.transformer_config` - `TransformerConfig`
- `../utils` - Logging, memory utilities
- `torch.distributed` - Collectives (all-reduce, all-gather, reduce-scatter)
- `torch.cuda` - GPU memory, NCCL
- `transformer_engine` (optional) - FP8 dtype support

**Used By** (imported by):
- `megatron/training/training.py` - Main training loop wraps model with DDP
- `megatron/core/optimizer/distrib_optimizer.py` - ZeRO optimizer integration
- `megatron/rl/` - RLHF training uses DDP for rollout collection
- User training scripts - Direct DDP instantiation for custom loops

---

## Gotchas & Performance Tips

1. **Bucketing Disabled by Default**: If `overlap_grad_reduce=False`, bucket_size is set to
   infinity (single bucket). Enable overlap for per-bucket async all-reduce.

2. **FP32 Grad Reduce**: `grad_reduce_in_fp32=True` adds overhead but can improve stability
   in mixed-precision training. Trade-off between numerical precision vs. communication cost.

3. **NCCL User Buffers**: `nccl_ub=True` registers persistent buffers with NCCL for SM
   efficiency. Requires `fsdp_double_buffer=True` and incompatible with PyTorch's
   expandable_segments. Provides 10-20% speedup on large models.

4. **Parameter Gather Overlap**: `overlap_param_gather=True` (FSDP/ZeRO-3) overlaps
   all-gather with forward compute. Requires careful scheduling to avoid deadlocks.

5. **Custom FSDP vs PyTorch**: Megatron's custom FSDP (~15% faster) uses optimized
   collective patterns. Not compatible with PyTorch's FSDP directly; use
   `use_megatron_fsdp=True`.

6. **Process Group Setup**: DP groups must align with data parallelism topology. Use
   `parallel_state.get_data_parallel_group()` or pass explicit `ProcessGroupCollection`.

7. **Gradient Scaling**: Always scale before all-reduce (not after) to avoid numerical
   overflow in FP8 mode. Use `scale_gradients()` method.

---

## See Also

- `../parallel_state.py` - Process group initialization and queries
- `../optimizer/distrib_optimizer.py` - ZeRO-enabled distributed optimizer
- `../CLAUDE.md` - Core library overview
- `megatron/training/CLAUDE.md` - Training loop integration
- `/MOE_TRAINING_GUIDE.md` - MoE-specific DP considerations
