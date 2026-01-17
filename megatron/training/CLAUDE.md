# CLAUDE.md - training

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Training orchestration: CLI arguments, training loop, checkpointing, init
> **Parent**: [../CLAUDE.md](../../CLAUDE.md)

---

## Overview

`megatron/training/` is the training orchestration layer that glues together `megatron/core/`
components into end-to-end training scripts. It handles CLI arguments, the main training loop,
checkpoint management, tokenization, and training initialization.

**Problem Solved**: Core provides composable building blocks (models, parallelism, data), but
training requires orchestration across these: parsing hundreds of CLI args, managing distributed
training state, handling checkpoints, providing tokenizers, and driving the training loop with
metrics/logging.

**Key Concepts**:
- **Arguments Layer**: 25+ argument groups mapping to different subsystems (core, parallelism,
  training, etc.)
- **Global State Pattern**: Singleton accessors (`get_args()`, `get_tokenizer()`) for
  cluster-wide state
- **Training Loop**: Main loop drives forward/backward/optimizer steps; handles pipeline
  scheduling
- **Checkpointing**: Distributed checkpoint save/load with resume-from-checkpoint support
- **Initialization**: Sets up distributed groups, seeds, and device placement before training

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| arguments.py | 3551 | CLI argument parser (25+ groups) | `parse_args()`, `validate_args()` |
| training.py | 3139 | Main training loop & entry points | `pretrain()`, `train()`, `train_step()` |
| checkpointing.py | 1912 | Checkpoint save/load with resume | `save_checkpoint()`, `load_checkpoint()` |
| utils.py | 692 | Rank-aware printing, timing utilities | `print_rank_0()`, `is_last_rank()` |
| initialize.py | 553 | Distributed setup, seed initialization | `initialize_megatron()` |
| one_logger_utils.py | 469 | OneLogger integration for metrics | `one_logger_log()` |
| yaml_arguments.py | 451 | YAML config file parsing | `load_yaml_config()` |
| theoretical_memory_usage.py | 367 | Memory estimation utilities | `compute_theoretical_memory()` |
| ft_integration.py | 367 | Fault tolerance integration | `FTIntegration` |
| global_vars.py | 308 | Singleton accessors for cluster state | `get_args()`, `get_tokenizer()`, `get_timers()` |
| argument_utils.py | 250 | Argument parsing utilities | `add_common_args()` |
| inprocess_restart.py | 159 | In-process restart for fault recovery | `inprocess_restart()` |
| config.py | 116 | Training config dataclass | `TrainingConfig` |

### Subdirectories

| Directory | Files | Purpose | Key Exports |
|-----------|-------|---------|-------------|
| tokenizer/ | 6 | Tokenizer implementations | `build_tokenizer()` |
| - tokenizer.py | 923 | Main tokenizer factory | GPT2, BERT, SentencePiece, HF |
| - bert_tokenization.py | 431 | BERT tokenizer | `BertTokenizer` |
| - multimodal_tokenizer.py | 340 | Multimodal tokenizer | `MultimodalTokenizer` |
| - gpt2_tokenization.py | 324 | GPT-2 tokenizer | `GPT2Tokenizer` |
| - sft_tokenizer.py | 196 | SFT tokenizer wrapper | `SFTTokenizer` |
| datasets/ | 4 | Data samplers & datasets | |
| - fim_dataset.py | 308 | Fill-in-the-middle dataset | `FIMDataset` |
| - data_samplers.py | 275 | Distributed samplers | `MegatronPretrainingSampler` |
| - sft_dataset.py | 146 | Supervised fine-tuning dataset | `SFTDataset` |

---

## Architecture

**Initialization Flow**:
```
initialize_megatron() → set_random_seed() → setup_distributed() → parse_args()
→ initialize_megatron_logging() → device placement + parallelism state setup
```

**Training Loop**:
```
pretrain() → build model & optim → for epoch/step:
  ├─ forward pass (with pipeline schedule if PP enabled)
  ├─ backward pass (with loss accumulation if gradient accumulation)
  ├─ optimizer step → reduce gradients (DP/TP communication)
  ├─ log metrics to tensorboard/wandb
  ├─ periodic checkpoint save
  └─ validation/evaluation
```

**Key Design**:
- Arguments are parsed once, stored globally via `set_args()` / `get_args()`
- Tokenizers are loaded once, accessed globally via `set_tokenizer()` / `get_tokenizer()`
- Training loop drives `megatron/core/` forward/backward; core is stateless
- Checkpointing uses distributed sharded state-dict format (not full-model-on-one-rank)

---

## Common Tasks

**Access global training arguments**:
```python
from megatron.training import get_args
args = get_args()
print(args.tensor_model_parallel_size, args.pipeline_model_parallel_size)
```

**Load checkpoint and resume training**:
```python
from megatron.training import load_checkpoint, save_checkpoint
load_checkpoint(model, optimizer, lr_scheduler)
# Training resumes from saved iteration
save_checkpoint(iteration, model, optimizer, lr_scheduler)
```

**Print rank-aware logs**:
```python
from megatron.training import print_rank_0, is_last_rank
print_rank_0("This prints on rank 0 only")
if is_last_rank():
    print("This prints on last rank")
```

**Add new CLI arguments**:
Edit `arguments.py`. Add to appropriate argument group (e.g., `_add_training_args()`,
`_add_parallelism_args()`), then access via `get_args().your_arg_name`.

---

## Dependencies

**Imports From**:
- `megatron/core/` - Models, parallelism, optimization, inference
- `megatron/core/distributed/` - `parallel_state` for distributed setup
- `transformer_engine` - FP8 operations (if enabled)

**Imported By**:
- `pretrain_*.py` - Training entry points call `pretrain()` after importing from here
- Examples in `examples/` - Use `initialize_megatron()` and argument parsing

---

## Gotchas

**Global State Initialization Order**: `initialize_megatron()` MUST be called before
`get_args()` / `get_tokenizer()`. Calling these early returns None/raises error.

**Distributed Checkpoint Format**: Checkpoints are sharded by TP/PP/DP groups, not
single-file. Don't use `torch.load()` directly; use `load_checkpoint()`.

**Argument Groups Are Extensive**: 3500+ lines of args. Use `python pretrain_gpt.py
--help | grep <topic>` to find what you need instead of searching blindly.

**Gradient Accumulation vs. Batch Size**: `micro_batch_size` × `gradient_accumulation_steps`
= effective batch size. Small micro-batch on large model = slow (not memory-efficient).

---

## See Also

- `megatron/core/CLAUDE.md` - Parallelism, optimizers, datasets, core building blocks
- `megatron/core/transformer/CLAUDE.md` - TransformerConfig, attention mechanisms
- `megatron/core/models/CLAUDE.md` - GPT, BERT, Llama model implementations
- `megatron/core/dist_checkpointing/CLAUDE.md` - Distributed checkpoint format details
- `examples/CLAUDE.md` - Training examples using this module

