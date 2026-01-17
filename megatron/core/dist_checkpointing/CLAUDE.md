# CLAUDE.md - dist_checkpointing

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Distributed checkpointing: async I/O, model resharding, multiple backends
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`dist_checkpointing/` provides parallelism-agnostic distributed checkpoint save/load with
multiple storage backends (torch, zarr, tensorstore, fully_parallel). Unlike traditional
`torch.save()`, this system decouples checkpoint format from model parallelism configuration,
enabling seamless model resharding on load (e.g., train TP=8,PP=1 → load into TP=4,PP=2).

**Problem Solved**: Distributed training produces state dicts spread across many GPUs in
rank-specific shards. Loading into a differently-parallelized model requires intelligent
resharding. This module handles: (1) distributed async I/O to avoid blocking training,
(2) format-agnostic tensor mapping via ShardedStateDict, (3) multi-backend support for
cloud storage (GCS, S3 via tensorstore) and local filesystems.

**Key Concepts**:
- **ShardedTensor/ShardedObject**: Metadata describing how local tensors map to global
  tensors in checkpoint (offset, fragmentation, replica_id)
- **ShardedStateDict**: Model state dict with tensors replaced by ShardedTensor metadata
- **Strategies**: Backend implementations (torch, zarr, fully_parallel) for save/load
- **Async I/O**: Non-blocking checkpointing with AsyncRequest queueing
- **Model Resharding**: Change parallelism config on load without data loss

---

## File Map

| File | Purpose |
|------|---------|
| `serialization.py` (454L) | Main API: `load()`, `save()`, load variants |
| `mapping.py` (559L) | Core classes: ShardedTensor, ShardedObject, factories |
| `validation.py` (545L) | Checkpoint validation, strict loading, metadata checks |
| `exchange_utils.py` (573L) | Inter-rank tensor exchange for resharding |
| `utils.py` (349L) | Utilities (tensor flattening, FP8 handling) |
| `tensor_aware_state_dict.py` (394L) | State dict preprocessing (factories, non-persistent) |
| `core.py` (XXL) | CheckpointingConfig, metadata.json management |
| `dict_utils.py` (XXL) | Dict merging, nested key extraction |
| `state_dict_utils.py` (XXL) | Save/load preprocessing (factories, common) |
| `optimizer.py` (XXL) | Optimizer-specific state dict handling |
| `strategies/torch.py` (947L) | PyTorch distributed.checkpoint backend |
| `strategies/zarr.py` (357L) | Zarr array storage backend |
| `strategies/fully_parallel.py` (520L) | No resharding: 1:1 rank → file mapping |
| `strategies/filesystem_async.py` (645L) | Async I/O wrapper for filesystem writes |
| `strategies/async_utils.py` (611L) | AsyncRequest, finalization, scheduling |
| `strategies/two_stage.py` (266L) | Staging strategy (local tmp → remote) |

---

## Architecture

**Data Flow - Saving**:

```
Model State Dict (distributed across ranks)
  ↓
save_preprocess() [extract factories, check consistency]
  ↓
ShardedStateDict [tensors → ShardedTensor metadata]
  ↓
Strategy.save() [backend-specific serialization]
  ├─ Rank 0: common state dict → common.pt (non-distributed)
  └─ All ranks: sharded tensors → backend storage (torch_dist/, zarr/, etc.)
  ↓
metadata.json [backend + version info]
```

**Data Flow - Loading**:

```
ShardedStateDict (model's target parallelism config)
  ↓
load() [verify checkpoint format]
  ├─ Load common.pt (non-sharded state)
  ├─ Load ShardedTensor metadata from checkpoint
  ├─ Validate checkpoint integrity (strict modes)
  └─ Load ShardedTensor data via strategy
  ↓
exchange_utils exchange data [all-to-all if resharding needed]
  ↓
apply_factory_merges() [rebuild from factored tensors]
  ↓
Loaded State Dict (matching target parallelism)
```

**Sharding Metadata**:

ShardedTensor encodes how a local tensor (e.g., shape [4096, 8192] on rank 2) maps to a
global tensor (shape [16384, 32768]):

```python
ShardedTensor(
    key="transformer.layers.0.attention.query_proj.weight",
    data=tensor([...]),  # local shape [4096, 8192]
    dtype=torch.float32,
    local_shape=(4096, 8192),
    global_shape=(16384, 32768),  # global shape
    global_offset=(4096, 16384),  # where local tensor starts
    axis_fragmentations=(4, 4),   # tensor sharded into 4x4 grid
    replica_id=0,                 # not a duplicate copy
)
```

**Backends**:

| Backend | Format | Use Case |
|---------|--------|----------|
| `torch_dist` | PyTorch distrib.checkpoint + metadata | Default, supports resharding |
| `zarr` | Zarr arrays + group metadata | Cloud storage, S3/GCS compatibility |
| `tensorstore` | TensorStore multi-backend | Enterprise: multi-cloud, versioning |
| `fully_parallel` | Each rank saves own shard only | Resharding disabled (faster, less memory) |

**Strategies Base Classes**:

```python
SaveShardedStrategy
  ├─ save(sharded_state_dict, checkpoint_dir)
  └─ AsyncSaveShardedStrategy
      └─ async_save() → AsyncRequest (finalization fns queued)

LoadShardedStrategy
  ├─ load(sharded_state_dict, checkpoint_dir)
  ├─ load_tensors_metadata()  # shape/dtype only
  └─ load_sharded_metadata()  # ShardedTensor + ShardedObject
```

---

## Common Tasks

### 1. Save Distributed Checkpoint (Synchronous)

```python
from megatron.core.dist_checkpointing import save, ShardedTensor

# Wrap model tensors with ShardedTensor metadata
sharded_state = {
    'transformer.layer.0.weight': ShardedTensor.from_rank_offsets(
        key='transformer.layer.0.weight',
        data=local_tensor,
        (0, rank, world_size),  # axis 0: rank-th of world_size shards
    ),
}

save(sharded_state, checkpoint_dir='./ckpt')
```

### 2. Save with Async I/O (Non-Blocking)

```python
from megatron.core.dist_checkpointing import save

async_request = save(
    sharded_state,
    checkpoint_dir='./ckpt',
    async_sharded_save=True,  # enable async
)

# Continue training...
train_step()

# Wait for checkpoint to finish
if async_request is not None:
    async_request.wait()  # blocks until all I/O done
```

### 3. Load with Model Resharding

```python
from megatron.core.dist_checkpointing import load

# OLD config: TP=8, PP=1 (checkpoint saved with this)
# NEW config: TP=4, PP=2 (model loaded with this)

# Create sharded state dict with NEW parallelism shape
new_sharded_state = {
    'transformer.layer.0.weight': ShardedTensor.from_rank_offsets(
        key='...',
        data=torch.empty(new_local_shape),  # NEW shape per rank
        (0, rank, 4),  # TP=4 (not 8)
        (1, rank, 2),  # PP=2 (not 1)
    ),
}

loaded_state = load(
    new_sharded_state,
    checkpoint_dir='./ckpt',  # from TP=8, PP=1
    validate_access_integrity=True,  # verify no double-loads
)
```

### 4. Load Common State Only (No Sharded Tensors)

```python
from megatron.core.dist_checkpointing import load_common_state_dict

# Get non-distributed state (e.g., loss scale, iteration count)
common = load_common_state_dict(checkpoint_dir='./ckpt')
print(common.keys())  # loss_scale, iteration, etc.
```

### 5. Load Metadata Without Data (For Inspection)

```python
from megatron.core.dist_checkpointing import load_tensors_metadata

metadata = load_tensors_metadata(checkpoint_dir='./ckpt')
# metadata['transformer.layer.0.weight'] = ShardedTensor(data=None, ...)
# Useful for shape/dtype checks before allocating memory
```

### 6. Strict Loading (Validate Checkpoint Completeness)

```python
from megatron.core.dist_checkpointing import load, StrictHandling

loaded, missing, unexpected = load(
    sharded_state,
    checkpoint_dir='./ckpt',
    strict=StrictHandling.RETURN_ALL,  # return mismatches
)

print(f"Missing keys: {missing}")        # in state but not checkpoint
print(f"Unexpected keys: {unexpected}")  # in checkpoint but not state
```

---

## Dependencies

**Imports From** (megatron.core):
- `parallel_state` - world_size, rank, process groups for distributed calls
- `utils` - GlobalMemoryBuffer for memory tracking

**Imports From** (torch):
- `torch.distributed` - all-to-all, barrier, P2P
- `torch.distributed.checkpoint` - PyTorch backend APIs
- `torch.distributed._shard` - ShardedTensor metadata

**Optional** (imported if available):
- `tensorstore` - TensorStore backend (cloud storage)
- `zarr` - Zarr backend
- `transformer_engine.pytorch.float8_tensor` - FP8 tensor handling

**Used By**:
- `megatron/training/training.py` - save/load in training loop
- `megatron/core/inference/` - checkpoint loading for inference
- User training scripts - primary API for checkpointing

---

## Gotchas & Tips

1. **ShardedTensor.from_rank_offsets()** vs **from_rank_offsets_flat()**: Use rank_offsets
   for regular grids only (e.g., TP=4 splits evenly). Use flat constructor for irregular
   sharding.

2. **replica_id and Main Replicas**: By default (replica_id=0), each shard is loaded once.
   If redundantly sharded across ranks, set replica_id to ensure only main replica loads.
   Use `is_main_replica()` helper.

3. **Async Requests Must be Scheduled**: Calling `async_save()` returns AsyncRequest but
   does NOT start I/O. Caller must schedule via training loop's async handler.

4. **validate_access_integrity=True**: Checks each tensor accessed exactly once (avoids
   double-loading). Disable only if you intentionally have redundant shards.

5. **Factory Pattern for Complex States**: Use ShardedTensorFactory to apply transformations
   before save (e.g., optimizer momentums → separate sharded tensors) and after load (merge
   back). See `mapping.py:ShardedTensorFactory`.

6. **FP8 Dequantization on Load**: load() automatically dequantizes FP8 tensors from
   Transformer Engine to higher precision. Avoids amax_history corruption.

7. **Metadata.json Required**: Always present after save(). Documents backend + version.
   Without it, load() cannot determine checkpoint type.

8. **Non-Persistent Objects**: Wrap objects in LocalNonpersistentObject to exclude from
   checkpoint (e.g., sampled train indices). Not saved, but can be reconstructed locally
   on load.

9. **Memory Implications of Resharding**: Exchange_utils performs all-to-all data
   redistribution. For very large models, may spike memory. Use `fully_parallel` backend
   to disable resharding if needed.

10. **Backward Compatibility**: Torch backend v=1 supports v0.14 checkpoints. zarr backend
    version-independent. Check `CheckpointingConfig` after load for format info.

---

## See Also

- [../CLAUDE.md](../CLAUDE.md) - Core library overview
- [../transformer/CLAUDE.md](../transformer/CLAUDE.md) - TransformerConfig for parallelism
- [../parallel_state.py](../parallel_state.py) - Process group management (required for
  dist_checkpoint initialization)
- [../optimizer/CLAUDE.md](../optimizer/CLAUDE.md) - Distributed optimizer state checkpointing
- `/MOE_TRAINING_GUIDE.md` - MoE-specific checkpoint guidance
