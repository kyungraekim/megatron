# Distributed Checkpointing Guide

> Covers: checkpoint save/load, model resharding, async I/O, ShardedTensor format
> Parent: [../CLAUDE.md](../CLAUDE.md)

## Key Concepts

- **ShardedTensor**: local tensor + `global_shape` + `global_offset` + `axis_fragmentations` + `replica_id`. Encodes how a local shard maps to the full global tensor.
- **replica_id**: `0` = main replica (saved/loaded). `>0` = duplicate (skipped on save, not loaded). Use `is_main_replica()` helper. Getting this wrong either causes double-writes or silently drops parameters.
- **ShardedStateDict**: decouples checkpoint format from parallelism config. Train TP=8/PP=1, load into TP=4/PP=2 seamlessly.
- **Backends**: `torch_dist` (default, supports resharding), `fully_parallel` (no resharding, faster, less memory). Zarr support was removed in v0.16.
- **NVSHMEM copy service** (`resharding/nvshmem_copy_service/`): new in v0.16 — GPU-native resharding via NVSHMEM with pipeline executor and double-buffering.

## Gotchas

- `async_save()` returns `AsyncRequest` but does **NOT start I/O**. The caller (training loop) must schedule it separately. Calling it just queues finalization functions.
- `ShardedTensor.from_rank_offsets()` is for **regular grids only** (e.g., TP=4 splits evenly). Use the flat constructor for irregular sharding.
- `validate_access_integrity=True` checks each tensor is accessed exactly once. Disable only for intentionally redundant shards.
- **FP8 auto-dequantization**: `load()` automatically dequantizes FP8 tensors from Transformer Engine to higher precision, avoiding amax_history corruption. Completely non-obvious.
- `metadata.json` is required after save. Without it, `load()` cannot determine checkpoint type.
- **LocalNonpersistentObject**: wrap objects to exclude from checkpoint (e.g., sampled indices). Not saved; reconstructed locally on load.
- **ShardedTensorFactory**: use for optimizer momentums — split before save, merge after load. See `mapping.py`.
- `exchange_utils` performs all-to-all for resharding and **may spike memory** on very large models. Use `fully_parallel` backend to avoid this.
- Persistent checkpoint worker loop now has explicit GC (new in v0.16) to prevent memory leaks in long-running jobs.
