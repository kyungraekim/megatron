# CLAUDE.md - core

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Core library with GPU-optimized building blocks for distributed training
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/` is the production-ready composable library that powers distributed large
language model training. It provides modular building blocks (layers, optimizers, parallelism
strategies) that work together to scale transformer training from 2B to 671B+ parameters.

**Problem Solved**: Scaling transformer training across multiple GPUs/nodes requires careful
coordination of tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP),
context parallelism (CP), and expert parallelism (EP). This directory abstracts these
complexities into reusable components with a unified configuration-driven API.

**Key Concepts**:
- **Parallelism-First**: All modules inherit `MegatronModule`, use `parallel_state` for
  coordination
- **Config-Driven**: Everything parameterized via dataclass configs (`TransformerConfig`,
  `ModelParallelConfig`)
- **Spec-Based Models**: Models built from `ModuleSpec` for easy layer customization
- **Distributed Checkpointing**: State-dict agnostic checkpointing across arbitrary
  parallelism configurations
- **GPU-Optimized**: CUDA kernels (fusions), Transformer Engine (FP8), JAX-style FSDP

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `parallel_state.py` | 2143 | Process group management (TP/PP/DP/CP/EP) | `initialize_model_parallel()`, `get_tensor_model_parallel_group()` |
| `model_parallel_config.py` | - | Dataclass for all parallelism config | `ModelParallelConfig` |
| `utils.py` | 2462 | Utilities (log, metrics, profiling) | `GlobalMemoryBuffer`, `print_rank_0()` |
| `optimizer/distrib_optimizer.py` | 2603 | ZeRO-style distributed optimizer | `DistributedOptimizer` |
| `optimizer/optimizer.py` | 1419 | Base optimizer wrapper | `MegatronOptimizer` |
| `transformer/transformer_config.py` | 1857 | Transformer layer configuration | `TransformerConfig` |
| `transformer/transformer_layer.py` | 1116 | Attention + MLP + LayerNorm | `TransformerLayer` |
| `transformer/attention.py` | 1499 | Multi-head/GQA attention implementations | `ParallelAttention` |
| `transformer/transformer_block.py` | 847 | Wrapper for transformer layer + residual | `TransformerBlock` |
| `transformer/cuda_graphs.py` | 1976 | CUDA graph caching for inference | `CudaGraphBuilder` |
| `tensor_parallel/layers.py` | 1315 | TP layers (Linear, Embedding) | `ColumnParallelLinear`, `RowParallelLinear` |
| `tensor_parallel/mappings.py` | 596 | All-reduce, all-gather, reduce-scatter | `gather_from_tensor_model_parallel_region()` |
| `tensor_parallel/random.py` | 688 | RNG seeding for TP | `CudaRNGStatesTracker` |
| `pipeline_parallel/schedules.py` | 2306 | 1F1B, Interleaved schedules | `get_forward_backward_func()` |
| `pipeline_parallel/p2p_communication.py` | 645 | Send/recv tensors across stages | `send_forward()`, `recv_forward()` |
| `distributed/distributed_data_parallel.py` | 592 | DDP wrapper with grad sync | `DistributedDataParallel` |
| `distributed/finalize_model_grads.py` | 488 | Allreduce gradients across DP ranks | `finalize_model_grads()` |
| `datasets/gpt_dataset.py` | 882 | GPT-style dataset sampling | `GPTDataset` |
| `datasets/indexed_dataset.py` | 1028 | Memory-mapped indexed data access | `IndexedDataset` |
| `inference/inference_request.py` | 537 | Request queuing for batch inference | `InferenceRequest` |
| `dist_checkpointing/mapping.py` | 559 | State-dict to shard mapping | `ShardedStateDict` |
| `extensions/transformer_engine.py` | 2486 | Transformer Engine (FP8, fusions) | `TransformerEngineLayer` |
| `extensions/kitchen.py` | 1808 | Kitchen Sink integration | `KitchenSinkLayer` |
| `rerun_state_machine.py` | 1390 | Loss spike recovery via rerun | `RerunnableFunction` |
| `timers.py` | 474 | Distributed timing metrics | `Timers` |

---

## Architecture

**Dependency Hierarchy** (bottom-up):

```
┌─────────────────────────────────────┐
│ parallel_state (process groups)     │
│ model_parallel_config (configs)     │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ TP (tensor_parallel/)               │  Intra-node layer sharding
│ DP (distributed/)                   │  Cross-device gradient sync
│ PP (pipeline_parallel/)             │  Multi-stage communication
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ transformer/ (layers)               │  Attention, MLP, LayerNorm
│ optimizer/ (distributed training)   │  Gradient aggregation, ZeRO
│ datasets/ (data pipeline)           │  Sampling, preprocessing
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ models/ (GPT, BERT, Llama, etc)     │  Composed from specs
│ inference/ (text generation)        │  Batched decoding
│ dist_checkpointing/ (save/load)     │  Parallelism-agnostic format
└─────────────────────────────────────┘
```

**Key Data Flows**:

1. **Training Loop**: Data → Model (fwd) → Loss → Backward → Optimizer Step
   - TP layers split weights across GPUs
   - PP stages pipeline computation across nodes
   - DP synchronizes gradients
   - Optimizer aggregates to shared parameters

2. **Inference**: Batch of requests → Cuda graphs (if enabled) → Attention/MLP →
   KV-cache → Decode loop

3. **Checkpointing**: Model state → ShardedStateDict → Distributed save/load →
   Reshape to new parallelism config

---

## Common Tasks

### 1. Initialize Distributed Training

```python
from megatron.core import parallel_state, ModelParallelConfig

config = ModelParallelConfig(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)
parallel_state.initialize_model_parallel(config)
```

### 2. Create a TP-enabled Layer

```python
from megatron.core.tensor_parallel import ColumnParallelLinear

layer = ColumnParallelLinear(
    input_size=4096, output_size=4096, bias=True,
    parallel_state=parallel_state,
)
```

### 3. Access Current Parallelism Context

```python
tp_rank = parallel_state.get_tensor_model_parallel_rank()
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
dp_rank = parallel_state.get_data_parallel_rank()
```

### 4. Build Model from Spec

```python
from megatron.core.models import GPTModel
from megatron.core.transformer import ModuleSpec

spec = ModuleSpec(
    module_class=TransformerLayer,
    module_config=TransformerConfig(...),
)
model = GPTModel(transformer_layer_spec=spec, vocab_size=50257)
```

### 5. Save/Load Distributed Checkpoint

```python
from megatron.core.dist_checkpointing import save, load

save(model, optimizer, checkpoint_path)
state = load(model, checkpoint_path)
```

---

## Dependencies

**Depends On** (imports from):
- `torch`, `torch.nn.parallel` - PyTorch core
- `transformer_engine` (optional) - FP8 kernels, LayerNorm fusion
- `flash-attn` (optional) - Flash attention backend
- `jax.experimental.pjit` (optional) - FSDP alternative
- `einops` - Tensor reshaping utilities

**Used By** (imported by):
- `megatron/training/` - Training scripts (main entry point)
- `megatron/rl/` - RLHF, GRPO implementations
- `megatron/post_training/` - Quantization, distillation
- `examples/` - Training examples
- Custom user models extending this library

---

## Gotchas & Tips

1. **Process Group Initialization**: Must call `initialize_model_parallel()` exactly once
   before creating any distributed layers. Use `parallel_state.destroy_model_parallel()` in
   tests to reset.

2. **RNG Seeding**: TP requires a `CudaRNGStatesTracker` to ensure deterministic layer
   behavior across tensor-parallel shards. Set seeds via `mpu.set_tensor_model_parallel_seed()`.

3. **Gradient Aggregation**: DDP allreduce only happens between data-parallel replicas. If
   using TP, gradients are already synchronized via backprop through sharded tensors.

4. **Pipeline Bubbles**: PP introduces bubbles (idle GPUs). Use 1F1B schedule (default) for
   optimal throughput on large models.

5. **Checkpoint Compatibility**: ShardedStateDict format is independent of parallelism
   config. You can train with TP=8, PP=1 then load into TP=4, PP=2.

6. **Memory Profiling**: Use `utils.GlobalMemoryBuffer` to track peak allocated memory
   across ranks.

---

## See Also

- `megatron/core/transformer/CLAUDE.md` - Attention, MLP, TransformerConfig details
- `megatron/core/models/CLAUDE.md` - GPT, BERT, Llama architectures
- `megatron/training/CLAUDE.md` - Training loop, arguments, initialization
- `megatron/rl/CLAUDE.md` - RLHF, GRPO training
- `MOE_TRAINING_GUIDE.md` - Mixture of Experts training
