# Megatron-LM AI Guide

> NVIDIA Megatron-LM v0.16.1 | GPU-optimized distributed training for transformer models at scale

## Architecture

- **Megatron Core** (`megatron/core/`) — Production library: parallelism, transformers, optimizers, checkpointing
- **Megatron Training** (`megatron/training/`) — Orchestration: CLI args, training loop, initialization
- **Legacy** (`megatron/legacy/`) — Deprecated, avoid modifying
- Config-driven: `ModelParallelConfig` → `TransformerConfig` (inheritance chain)
- Spec-based: `ModuleSpec` defines layer composition; backends (local/TE) swapped transparently
- Parallelism formula: **Total GPUs = TP x PP x CP x EP x DP**
- Rank ordering string `"tp-cp-ep-dp-pp"` governs all process group assignments

## Code Style (pre-commit enforced)

- **black**: line-length 100, `--skip-magic-trailing-comma --skip-string-normalization`
- **isort**: profile=black, known_first_party=megatron
- **pylint**: megatron/core/ only
- Import order: `__future__` → stdlib → third-party → `megatron.*` → relative
- New files: Apache 2.0 license header
- Format: `./tools/autoformat.sh`

## Workflow

- Branches: `main` (stable), `core_r*` (release)
- PR: pre-checks → `Expert Review` label → approvals → `Final Review` → merge
- Commits: imperative mood, <72 chars, explain "why"

## Cross-Cutting Gotchas

These are the most dangerous non-obvious behaviors in this codebase:

- `initialize_model_parallel()` must be called **exactly once**; use `destroy_model_parallel()` to reset in tests
- `TransformerConfig.__post_init__()` **mutates fields**: auto-sets `ffn_hidden_size=4*hidden_size`, `kv_channels=hidden_size/num_heads`, forces `attention_softmax_in_fp32` if query-key scaling enabled
- Sequence parallelism **reuses the TP group** — it is not a separate parallelism dimension
- Gradient accumulation fusion writes directly to `param.main_grad` inside the GEMM kernel, **bypassing `param.grad`** entirely
- `deallocate_output_tensor()` replaces `.data` with a 1-element scalar; `custom_backward()` calls C++ autograd engine directly because standard `.backward()` validates shapes
- `share_embeddings_and_output_weights`: embedding (PP stage 0) and output projection (last PP stage) sync via `_EMBEDDING_GROUP` all-reduce in `finalize_model_grads()`
- Expert params need `param.allreduce = False` or they silently get all-reduced across the wrong group
- `MoEAuxLossAutoScaler` injects aux loss gradient via a **global class variable** side-channel, not the computation graph
- Post-training: `--manual-gc` causes memory leak per fwd-bwd pass; `--tp-comm-overlap` incompatible with distillation; interleaved PP not supported for distillation
- NeMo checkpoint field translations: `encoder_seq_length` → `seq_length`, `share_embeddings_and_output_weights` is **inverted**

## Sub-Guides

Read the relevant sub-guide when working on a subsystem:

| Location | Covers |
|----------|--------|
| `megatron/core/CLAUDE.md` | Core patterns, TP, PP, DP, FSDP, optimizer, CUDA graphs |
| `megatron/core/transformer/CLAUDE.md` | Transformer layers, MoE, model architectures |
| `megatron/core/dist_checkpointing/CLAUDE.md` | Distributed checkpointing, resharding |
| `megatron/training/CLAUDE.md` | Training loop, arguments, initialization |
| `megatron/core/inference/CLAUDE.md` | Inference engines, text generation |
| `tests/CLAUDE.md` | Test patterns, fixtures, markers |

## Quick Reference

```bash
# Tests
pytest tests/unit_tests/
pytest --cov=megatron.core tests/unit_tests/

# Training
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# Arguments (3500+ lines — search with grep)
python pretrain_gpt.py --help | grep <topic>
```
