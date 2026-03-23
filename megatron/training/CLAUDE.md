# Training Orchestration Guide

> Covers: training loop, CLI arguments, initialization, checkpointing
> Parent: [../../CLAUDE.md](../../CLAUDE.md)

## Key Patterns

- **Global singletons**: `get_args()`, `get_tokenizer()`, `get_timers()` provide cluster-wide state. `initialize_megatron()` **must** be called before any of these — calling early returns None or raises.
- **Arguments parsed once**: stored globally via `set_args()`. Not re-parsed per iteration. Core is stateless; the training loop drives it.
- **Training loop**: `pretrain()` → build model & optimizer → for each step: fwd/bwd → gradient reduce → optimizer step → log → checkpoint → eval.
- **Config migration** (v0.16): new `TrainingConfig` dataclass (`training_config.py`), `CheckpointConfig`, `StragglerDetectionConfig`, `ResilienceConfig`. Arguments are being migrated from argparse to config dataclasses. `arguments.py` is still ~3500 lines.

## Gotchas

- **Never use `torch.load()` directly** for checkpoints. They are sharded by TP/PP/DP groups. Use `load_checkpoint()` / `save_checkpoint()`.
- **Batch size formula**: `micro_batch_size * gradient_accumulation_steps = effective batch size`.
- **arguments.py** is massive. Use `python pretrain_gpt.py --help | grep <topic>` to find what you need.
- **Optimizer CPU offload** with checkpointing: the `step` key must be handled correctly (fixed in v0.16, was a bug).
- **Phase transition iterations**: new in v0.16 — allows changing training config at specified iterations.
- **Dgrad logging** (`dgrad_logging.py`): new in v0.16 — gradient distribution logging for debugging.
