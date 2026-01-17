# CLAUDE.md - tensor_parallel

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Tensor parallelism: ColumnParallel, RowParallel layers, and sequence
> parallelism
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/tensor_parallel/` provides GPU-accelerated layers and communication
primitives for tensor model parallelism (TP). It enables splitting large transformer
layers across multiple GPUs within a node, reducing memory per GPU and enabling training
of larger models.

**Problem Solved**: A single GPU cannot fit large weight matrices (e.g., an 8K×8K
feed-forward layer). This module splits matrices across TP ranks and synchronizes
activations/gradients using optimized collective operations.

**Key Concepts**:
- **ColumnParallelLinear**: Splits output dimension (columns of weight matrix)
- **RowParallelLinear**: Splits input dimension (rows of weight matrix)
- **VocabParallelEmbedding**: Splits vocabulary across TP ranks
- **Sequence Parallelism**: Distribute activations along sequence dimension
- **Synchronized RNG**: Dropout consistency across TP replicas via `CudaRNGStatesTracker`
- **gather_output**: Parameter controlling whether outputs are gathered (replicated) or
  kept distributed

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `layers.py` | 1315 | TP linear, embedding layers | `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding` |
| `mappings.py` | 596 | Collective ops (all-gather, all-reduce) | `gather_from_tensor_model_parallel_region()`, `scatter_to_sequence_parallel_region()` |
| `random.py` | 688 | RNG state tracking for determinism | `CudaRNGStatesTracker`, `get_cuda_rng_tracker()`, `checkpoint()` |
| `cross_entropy.py` | 232 | Loss computation with vocab split | `vocab_parallel_cross_entropy()`, `VocabParallelCrossEntropy` |
| `inference_layers.py` | 294 | Inference-optimized TP layers | `InferenceLayerNormColumnParallelLinear` |
| `utils.py` | 121 | Tensor utilities, vocab range calc | `split_tensor_along_last_dim()`, `VocabUtility` |
| `data.py` | 101 | Data broadcast across TP ranks | `broadcast_data()` |

---

## Architecture

**Data Flow for ColumnParallelLinear**:

```
Input (B, S, H)  [same on all TP ranks]
    ↓
Matmul with split weights [each rank computes column slice]
    ↓
Output (B, S, H/TP_SIZE)  [distributed across TP ranks]
    ↓
[if gather_output=True]
    All-Gather: output becomes (B, S, H) [replicated on all ranks]
```

**Data Flow for RowParallelLinear**:

```
Input (B, S, H) [distributed: each rank has H/TP_SIZE features]
    ↓
Matmul with split weights (transposed)
    ↓
Output (B, S, H_OUT)  [each rank computes partial sum]
    ↓
All-Reduce: sum across TP ranks, result on all ranks
```

**Key Components**:
- `layers.py`: Actual nn.Module implementations with forward/backward logic
- `mappings.py`: Low-level communication (all-reduce, all-gather, reduce-scatter)
- `random.py`: Ensures dropout behaves identically on all TP ranks
- `parallel_state`: Provides TP rank info via `get_tensor_model_parallel_rank()`, etc.

---

## Common Tasks

### 1. Use ColumnParallelLinear in a Model

```python
from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core import ModelParallelConfig

config = ModelParallelConfig(tensor_model_parallel_size=4)
layer = ColumnParallelLinear(
    input_size=4096,
    output_size=8192,
    config=config,
    init_method=torch.nn.init.normal_,
    bias=True,
    gather_output=True,  # All-gather output to all ranks
)
# Forward: input (B, S, 4096) → output (B, S, 8192) [replicated]
```

### 2. Use RowParallelLinear (Feed-Forward Output Projection)

```python
layer = RowParallelLinear(
    input_size=8192,
    output_size=4096,
    config=config,
    init_method=torch.nn.init.normal_,
    bias=True,
    input_is_parallel=True,  # Input already split across TP ranks
)
# Forward: input (B, S, 8192/TP_SIZE) [distributed]
#          → output (B, S, 4096) [replicated after all-reduce]
```

### 3. Custom RNG Seeding for TP Dropout Consistency

```python
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

# Call once at startup to seed all TP ranks with same dropout behavior
model_parallel_cuda_manual_seed(123)
```

### 4. Compute Loss with vocab-parallel Cross-Entropy

```python
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy

logits = model(input_ids)  # shape: (B, S, vocab_size/TP_SIZE)
loss = vocab_parallel_cross_entropy(logits, labels)
# Handles all-reduce internally to compute full softmax
```

### 5. Access RNG Tracker for Checkpoint/Resume

```python
from megatron.core.tensor_parallel import checkpoint, get_cuda_rng_tracker

# Inside forward pass with recomputation:
with checkpoint(get_cuda_rng_tracker()):
    output = model(input)  # Recompute in backward with same RNG
```

---

## Dependencies

**Depends On** (imports from):
- `parallel_state` - TP rank/world-size queries
- `model_parallel_config` - `ModelParallelConfig` dataclass
- `torch.distributed` - Collective operations (all-gather, all-reduce, reduce-scatter)
- `torch` - CUDA kernels, autograd

**Used By** (imported by):
- `megatron/core/transformer/` - `TransformerLayer` uses `ColumnParallelLinear` +
  `RowParallelLinear`
- `megatron/core/models/` - GPT, BERT, Llama models
- `megatron/training/` - Training loop
- All TP-enabled code paths

---

## Gotchas & Tips

1. **gather_output Parameter**: Set `gather_output=True` for ColumnParallelLinear if the
   next layer expects replicated input (most common). Set to `False` only if next layer
   is RowParallelLinear or expects distributed input.

2. **RNG Seeding**: Call `model_parallel_cuda_manual_seed()` ONCE during setup, not per
   iteration. Dropout must be consistent across TP ranks.

3. **input_is_parallel in RowParallelLinear**: Must match whether your input is actually
   distributed. Mismatch causes silent wrong results.

4. **All-Reduce Cost**: RowParallelLinear does all-reduce in backward (expensive). MLP
   output projections usually worth it for model parallelism.

5. **Vocabulary Sharding**: VocabParallelEmbedding + vocab_parallel_cross_entropy() must
   work together. Embedding splits vocab, loss handles partial softmax.

6. **Global Memory Buffer**: Collective ops use `get_global_memory_buffer()` to avoid
   repeated allocation. Must be initialized by `parallel_state.initialize_model_parallel()`.

7. **Sequence Parallelism Orthogonal**: Can combine TP with sequence parallelism via
   `scatter_to_sequence_parallel_region()` / `gather_from_sequence_parallel_region()` for
   long-sequence training.

---

## See Also

- `megatron/core/CLAUDE.md` - Overall core architecture
- `megatron/core/transformer/CLAUDE.md` - How TP layers fit into TransformerLayer
- `megatron/core/models/CLAUDE.md` - Model architectures using TP
- `megatron/core/parallel_state.py` - Process group initialization
- Tests: `tests/unit_tests/tensor_parallel/` - Usage examples
