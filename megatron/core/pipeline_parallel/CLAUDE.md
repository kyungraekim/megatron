# CLAUDE.md - pipeline_parallel

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Pipeline parallelism: 1F1B schedules, p2p communication, and virtual
> pipelines
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/pipeline_parallel/` implements pipeline model parallelism (PP), which splits
model layers across multiple GPUs/nodes and executes them sequentially. This reduces
per-GPU memory but introduces pipeline bubbles (idle time). This module optimizes
throughput via:

- **1F1B (One-Forward-One-Backward)**: Memory-efficient schedule that interleaves forward
  and backward passes to reduce peak activation memory
- **Interleaved/Virtual Pipeline**: Multiple model chunks per GPU to reduce bubble time
- **P2P Communication**: Efficient tensor send/recv between pipeline stages using
  torch.distributed batched operations
- **Bridge Communicator**: Cross-grid communication for modules with different parallelism
  configurations (TP/DP/CP)

**Key Problem**: Scaling to 671B+ parameters requires splitting layers across nodes.
Naive execution creates bubbles where most GPUs idle. 1F1B + interleaving achieve 50%+
MFU on large models.

**Key Concepts**:
- **Pipeline Stage**: A rank that owns N consecutive transformer layers
- **Microbatch**: A small subset of the batch processed through all stages sequentially
- **Pipeline Bubble**: Time when a stage is idle waiting for data from previous stage
- **Virtual Pipeline**: Multiple model chunks per GPU to overlap computation/communication
- **1F1B**: Default schedule; backward of microbatch i runs while computing forward of i+1

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `__init__.py` | 4 | Public API | `get_forward_backward_func()` |
| `schedules.py` | 2306 | Pipeline schedule implementations | `get_forward_backward_func()` |
| `p2p_communication.py` | 645 | P2P send/recv between stages | `send_forward()`, `recv_forward()`, `send_backward()`, `recv_backward()` |
| `bridge_communicator.py` | 922 | Cross-grid communication | `BridgeCommunicator`, `CommRole` |
| `combined_1f1b.py` | 444 | Advanced 1F1B with A2A overlap | `combined_1f1b_schedule_for_no_pipelining()`, `combined_1f1b_schedule_for_interleaved_pipelining()` |
| `utils.py` | 315 | Scheduling utilities | `ScheduleNode`, `AbstractSchedulePlan`, `is_pp_first_stage()`, `is_pp_last_stage()` |

---

## Architecture

### Schedule Selection

```
get_forward_backward_func()
  ├─ if pp_size == 1 and vp_size == 1:
  │   └─ No pipeline (single GPU forward/backward)
  ├─ if using combined_1f1b:
  │   ├─ combined_1f1b_schedule_for_no_pipelining()
  │   └─ combined_1f1b_schedule_for_interleaved_pipelining()
  └─ else (default):
      ├─ forward_backward_no_pipelining() (pp_size==1)
      ├─ forward_backward_pipelining() (pp_size>1, no VP)
      └─ forward_backward_pipelining_with_interleaving() (pp_size>1, VP enabled)
```

### 1F1B Data Flow

```
Microbatch 0:     F0 ──────── B0
Microbatch 1:           F1 ── B1
Microbatch 2:                F2 B2
                    ↓
                 Stage 0  Stage 1  Stage 2

Key: Forward and backward of different microbatches overlap,
     reducing memory (backward starts before all forwards done)
     and improving GPU utilization.
```

### P2P Communication

```
Stage N: forward pass
  ├─ recv activations from stage N-1 (or get from data_iterator if N==0)
  ├─ compute forward (attention + MLP)
  └─ send activations to stage N+1 (or compute loss if N==last)

Stage N: backward pass
  ├─ recv loss gradient from stage N+1 (or compute from loss if N==last)
  ├─ compute backward (gradient through attention + MLP)
  └─ send loss gradient to stage N-1
```

Uses `torch.distributed.batch_isend_irecv()` for efficient batched async communication.

---

## Common Tasks

### 1. Get the Appropriate Forward/Backward Function

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

forward_backward_func = get_forward_backward_func()

# Call during training:
forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=data_iter,
    model=model,
    num_microbatches=4,
    seq_length=2048,
)
```

### 2. Implement forward_step_func

```python
def forward_step(data_iterator, model):
    data = next(data_iterator)
    output = model(data)

    def loss_func(output_tensor):
        loss = compute_loss(output_tensor)
        return loss, {'loss': loss}

    return output, loss_func
```

### 3. Check Pipeline Stage

```python
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage, is_pp_last_stage,
    is_vp_first_stage, is_vp_last_stage,
)
import torch.distributed as dist

pp_group = dist.new_group(ranks=[...])

if is_pp_first_stage(pp_group):
    # Receive data from iterator
    pass

if is_pp_last_stage(pp_group):
    # Compute loss
    pass
```

### 4. Use ScheduleNode for Fine-Grained Control

```python
from megatron.core.pipeline_parallel.utils import ScheduleNode
import torch

stream = torch.cuda.Stream()
event = torch.cuda.Event()

node = ScheduleNode(
    forward_func=lambda x: model(x),
    backward_func=None,  # uses default
    stream=stream,
    event=event,
    name="layer_0",
)

output = node.forward(input_tensor)
grad = node.backward(output_grad)
```

### 5. Cross-Grid Communication

```python
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator

comm = BridgeCommunicator(
    source_grid=source_grid,
    dest_grid=dest_grid,
    parallel_state=parallel_state,
)

# Send forward activations
comm.send_forward(activations)

# Receive and send in one call
result = comm.send_forward_recv_backward(
    forward_data=activations,
)
```

---

## Dependencies

**Depends On** (imports from):
- `megatron.core.parallel_state` - Process group queries
- `megatron.core.enums`, `megatron.core.model_parallel_config` - Config/enum types
- `megatron.core.transformer` - Model layers, cuda graphs
- `torch.distributed` - P2P ops, batched isend_irecv
- `megatron.core.hyper_comm_grid` - Cross-grid topology (for BridgeCommunicator)

**Used By** (imported by):
- `megatron.training.training` - Main training loop calls `get_forward_backward_func()`
- `megatron.core.models` - Model wrappers may use advanced schedules
- `megatron.rl` - RLHF/GRPO may use pipeline schedules

---

## Gotchas & Tips

1. **Microbatches**: Num microbatches should be ≥ 2 × pipeline_size for optimal throughput
   (fills pipeline). Too small = bubbles; too large = memory pressure.

2. **Virtual Pipeline Stages**: Each GPU holds multiple model chunks. VP reduces bubbles
   but increases memory. Use vp_size = pp_size for 50% bubble reduction on simple models.

3. **P2P vs All-Reduce**: P2P (pipeline) only communicates between adjacent stages.
   Gradients still need global all-reduce via data parallelism in `finalize_model_grads()`.

4. **Activation Checkpointing**: PP is memory-efficient, but you can still enable
   activation checkpointing to trade compute for memory. Most useful without VP.

5. **Forward-only Mode**: Set `forward_only=True` to skip backward (for inference or
   validation). Saves compute and memory.

6. **Sequence Parallelism with PP**: Context parallel can coexist with PP. Sequences are
   split across CP ranks, then interleaved with PP stages. Coordinate with
   `combine_1f1b=True` for best perf.

7. **Bridge Communicator**: Used when crossing grid boundaries (e.g., TP=4,DP=2 to
   TP=2,DP=4). Requires careful rank mapping; see `hyper_comm_grid.py`.

---

## See Also

- `megatron/core/CLAUDE.md` - Core library overview
- `megatron/core/transformer/CLAUDE.md` - Attention, MLP configs
- `megatron/training/CLAUDE.md` - Training loop, args parsing
- `MOE_TRAINING_GUIDE.md` - Pipeline + MoE specifics
