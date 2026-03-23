# Core Library Guide

> Covers: core patterns, tensor parallelism, pipeline parallelism, data parallelism, FSDP, optimizer, CUDA graphs
> Parent: [../../CLAUDE.md](../../CLAUDE.md)

## Key Patterns

- **Config inheritance**: `ModelParallelConfig` → `TransformerConfig`. A single config object carries both model hyperparams (`hidden_size`) and parallelism topology (`tensor_model_parallel_size`). Callbacks (`grad_scale_func`, `no_sync_func`, `param_sync_func`) live on the config, making it a live communication broker.
- **MegatronModule**: Base class for all distributed modules. Provides `state_dict_for_save_checkpoint()` and `sharded_state_dict()` for distributed checkpointing.
- **parallel_state**: Singleton module managing all process groups. The rank ordering string (default `"tp-cp-ep-dp-pp"`) determines how global ranks map to groups. Changing this string produces completely different group memberships. EP and CP cannot both be >1 in the same `RankGenerator`.
- **GlobalMemoryBuffer**: Must be initialized via `initialize_model_parallel()` before any collective ops use it.

## CUDA Graphs

- Capture happens at the **end of the first training step**, in execution order recorded by `_CudagraphGlobalRecord`. All runners must register in the same order every step.
- When `cuda_graph_use_single_mempool=True`, graphs sharing a pool must replay in capture order. Conditionally-skipped modules break this assumption.
- GC is frozen during capture for PyTorch < 2.9 (`FREEZE_GC = True`).
- After replay, `_CudaGraphRunner._run_bwd()` must restore `grad_added_to_main_grad` from a saved snapshot because GEMM kernels don't re-set this flag during replay.
- CUDA graphs are incompatible with dynamic shapes — disable for variable-length or adaptive batching.

## Tensor Parallelism & Sequence Parallelism

- **Sequence parallelism reuses the TP group** — not a separate axis. With SP enabled, after `ColumnParallelLinear` it reduce-scatters (instead of all-gather), and before `RowParallelLinear` it all-gathers (instead of all-reduce). P2P tensor shapes in pipeline schedules divide `seq_len` by `tp_size` when SP is on.
- `ColumnParallelLinear.gather_output`: set `True` if next layer expects replicated input; `False` only if next layer is `RowParallelLinear`. **Wrong setting = silent wrong results.**
- `RowParallelLinear.input_is_parallel`: must match actual input distribution. **Mismatch = silent wrong results.**
- `CudaRNGStatesTracker`: call `model_parallel_cuda_manual_seed()` **once at setup**, not per iteration. Required for dropout consistency across TP shards.
- `VocabParallelEmbedding` + `vocab_parallel_cross_entropy()` must be used together — partial softmax requires both sides.

## Pipeline Parallelism

- **Virtual pipeline backward reverses chunk order**: `get_model_chunk_id()` applies `num_model_chunks - id - 1`. Model chunks must be ordered consistently (chunk-0 first). Wrong order = wrong results with no error.
- **Microbatches**: should be >= 2 * pipeline_size for optimal throughput. Too small = bubbles, too large = memory.
- `deallocate_output_tensor()` replaces `.data` with a 1-element scalar (frees memory but keeps `.grad_fn`). `custom_backward()` calls C++ autograd engine directly because standard `.backward()` validates shapes.
- `forward_step_func` must return `(output, loss_func)` tuple.
- `BridgeCommunicator` required when crossing grid boundaries (e.g., TP=4/DP=2 to TP=2/DP=4).
- **Hybrid CP schedule** (`hybrid_cp_schedule.py`): new in v0.16 — combines context parallelism with pipeline parallelism.
- **Multimodule communicator** (`multimodule_communicator.py`): new in v0.16 — enables communication across independently-defined model modules in PP.
- **Fine-grained activation offloading** (`fine_grained_activation_offload.py`): new in v0.16 — offloads specific activations to CPU during forward, reloads during backward.

## Data Parallelism & Distributed

- **DistributedOptimizer partitioning does NOT respect parameter boundaries**. Each DP rank gets a contiguous slice of the flat gradient buffer that may split mid-parameter. Optimizer states are views into this buffer.
- **DDP bucketing disabled on non-first PP stages** (`bucket_size=None`). Gradient overlap only functions for PP stage 0; other stages all-reduce synchronously. This is intentional.
- **Gradient accumulation fusion**: when enabled, weight gradients accumulate directly into `param.main_grad` inside the GEMM kernel, bypassing `param.grad`. The `grad_added_to_main_grad` flag prevents double-addition.
- Always **scale gradients before all-reduce** (not after) to avoid numerical overflow in FP8 mode.
- **Megatron FSDP** is ~15% faster than PyTorch's (optimized collectives + FP8). Not compatible with PyTorch FSDP directly.
- NCCL user buffers (`nccl_ub=True`): 10-20% speedup but requires `fsdp_double_buffer=True` and is incompatible with PyTorch `expandable_segments`.
- **Embedding weight sync**: when `share_embeddings_and_output_weights=True`, embedding (PP stage 0) and output projection (last PP stage) sync via `_EMBEDDING_GROUP` all-reduce in `finalize_model_grads()`. New replicated params need `param.pipeline_parallel = True`.

## Optimizer

- `DistributedOptimizer` requires DP > 1; single-GPU silently falls back to standard optimizer.
- CPU offloading (`cpu_offload=True`) trades memory for compute — practical only for models > 200B.
- Gradient clipping does AllReduce to compute global norm (~1-5% of step time).
- **Muon optimizer** (`muon.py`): new in v0.16.
- **Layer-wise optimizer** (`layer_wise_optimizer.py`): new in v0.16 — allows different optimizer configs per layer.
