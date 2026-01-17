# CLAUDE.md - AI Assistant Guide for Megatron-LM

> **Repository**: NVIDIA Megatron-LM | **Version**: 0.14.0 (Megatron Core)
> **Purpose**: Index and navigation for AI assistants working on this codebase

This is the **root guide**. For detailed context on specific subsystems, see the linked
sub-CLAUDE.md files below. This approach reduces token usage by loading only relevant context.

---

## Project Overview

**Megatron-LM** is NVIDIA's GPU-optimized library for training transformer models at scale.

**Two Components**:
1. **Megatron Core** (`megatron/core/`) - Production-ready composable library
2. **Megatron-LM** (everything else) - Training scripts, examples, utilities

**Key Capabilities**: 2B-671B+ parameter training, 5 parallelism strategies (TP/PP/DP/CP/EP),
47% MFU on H100, MoE support (DeepSeek-V3, Mixtral), multi-modal models, production inference.

---

## Repository Map & Sub-CLAUDE.md Index

### Directory Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Core library (see sub-CLAUDE.md files below)
│   ├── training/                # Training orchestration
│   ├── legacy/                  # Deprecated code (avoid modifying)
│   ├── post_training/           # Quantization, distillation
│   └── rl/                      # RLHF, GRPO training
├── examples/                    # Training examples
├── tests/                       # Unit and functional tests
├── tools/                       # Utilities (preprocessing, inference, formatting)
├── docs/                        # Documentation
└── pretrain_*.py                # Training entry points
```

### Sub-CLAUDE.md Files

For detailed context, read the relevant sub-CLAUDE.md file:

| Location | Purpose | When to Read |
|----------|---------|--------------|
| `megatron/core/CLAUDE.md` | Core library overview, key concepts | Working on any core component |
| `megatron/core/transformer/CLAUDE.md` | Attention, MLP, TransformerConfig | Modifying transformer layers |
| `megatron/core/transformer/moe/CLAUDE.md` | MoE: routers, experts, dispatchers | Working on MoE models |
| `megatron/core/models/CLAUDE.md` | Model architectures (GPT, BERT, etc.) | Adding/modifying models |
| `megatron/core/pipeline_parallel/CLAUDE.md` | PP schedules, p2p communication | Pipeline parallelism work |
| `megatron/core/tensor_parallel/CLAUDE.md` | TP layers, mappings | Tensor parallelism work |
| `megatron/core/distributed/CLAUDE.md` | DDP, FSDP, ZeRO | Data parallelism work |
| `megatron/core/optimizer/CLAUDE.md` | Distributed optimizer, gradients | Optimizer changes |
| `megatron/core/inference/CLAUDE.md` | Inference engine, text generation | Inference work |
| `megatron/core/datasets/CLAUDE.md` | Data pipeline, indexed datasets | Data loading changes |
| `megatron/core/dist_checkpointing/CLAUDE.md` | Checkpoint strategies | Checkpointing work |
| `megatron/training/CLAUDE.md` | Training loop, arguments, init | Training script changes |
| `megatron/rl/CLAUDE.md` | GRPO, PPO, rollout generation | RL training work |
| `megatron/post_training/CLAUDE.md` | ModelOpt, quantization | Post-training work |
| `tests/CLAUDE.md` | Test patterns, fixtures | Writing tests |
| `examples/CLAUDE.md` | Example patterns | Creating examples |

---

## Universal Conventions

### Code Style (Enforced via pre-commit)

```yaml
Tools:
- black: Line length 100, skip-magic-trailing-comma, skip-string-normalization
- isort: Profile black, known_first_party=megatron
- pylint: For megatron/core/ only
- flake8: Max line 100, ignores E203/E501/F401/E402/E714
```

**Import Order**:
```python
from __future__ import annotations  # FUTURE
import os, sys                       # STDLIB
import torch, transformer_engine     # THIRDPARTY
from megatron.core import X          # FIRSTPARTY
from .config import Y                # LOCALFOLDER
```

**Naming**: Classes=`PascalCase`, functions=`snake_case`, constants=`UPPER_SNAKE_CASE`

**File Headers**: All new files need Apache 2.0 license header.

### Development Workflow

**Branch Strategy**:
- `main` - Stable release
- `dev` - Experimental features
- `core_r*` - Release branches

**PR Process** (for `main`):
1. Pre-checks (tests, formatting)
2. Add `Expert Review` label → auto-assigned reviewers
3. After approvals, add `Final Review` label
4. Merged by core-adlr or core-nemo teams

**Commit Messages**: Imperative mood, <72 chars summary, explain the "why"

### Formatting Commands

```bash
./tools/autoformat.sh              # Auto-format all
black --skip-magic-trailing-comma --skip-string-normalization megatron/core/
isort megatron/core/
pylint megatron/core/
```

---

## Quick Start

### Installation

```bash
# Recommended: NGC Docker container
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:25.04-py3

# Or pip install
pip install --no-build-isolation megatron-core[mlm,dev]

# Development setup
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation -e .[mlm,dev]
pip install pre-commit && pre-commit install
```

### Run Tests

```bash
pytest tests/unit_tests/                           # All unit tests
pytest tests/unit_tests/transformer/test_attention.py  # Specific test
pytest --cov=megatron.core tests/unit_tests/       # With coverage
```

### Run Training

```bash
# Simple example
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# GPT pretraining (see examples/ for full scripts)
python pretrain_gpt.py --tensor-model-parallel-size 4 ...
```

---

## Key Architectural Patterns

These patterns are used throughout the codebase:

1. **Config-Driven**: Everything uses dataclass configs (`TransformerConfig`, `ModelParallelConfig`)
2. **Spec-Based Models**: Models built from `ModuleSpec` for easy layer customization
3. **Parallelism-First**: All modules inherit `MegatronModule`, use `parallel_state`
4. **Extension Points**: `extensions/transformer_engine.py`, custom attention backends

---

## Parallelism Quick Reference

| Type | Flag | Use Case |
|------|------|----------|
| **TP** (Tensor) | `--tensor-model-parallel-size N` | Large layers (within node) |
| **PP** (Pipeline) | `--pipeline-model-parallel-size N` | Many layers (100B+ models) |
| **DP** (Data) | Automatic | Scale throughput |
| **CP** (Context) | `--context-parallel-size N` | Long sequences (8K+) |
| **EP** (Expert) | `--expert-model-parallel-size N` | MoE models |

**Formula**: Total GPUs = TP × PP × CP × EP × DP

For detailed parallelism strategies, see `megatron/core/*/CLAUDE.md` files.

---

## Largest/Most Complex Files

When working on these, expect high complexity:

| File | Size | Purpose |
|------|------|---------|
| `megatron/training/arguments.py` | 204KB | CLI argument parser (25+ groups) |
| `megatron/core/optimizer/distrib_optimizer.py` | 126KB | ZeRO-style distributed optimizer |
| `megatron/training/training.py` | 125KB | Main training loop |
| `megatron/core/pipeline_parallel/schedules.py` | 100KB | 1F1B, interleaved schedules |
| `megatron/core/extensions/transformer_engine.py` | 88KB | TE/FP8 integration |
| `megatron/core/parallel_state.py` | 87KB | Process group management |
| `megatron/core/transformer/transformer_config.py` | 77KB | Model configuration |

---

## For AI Assistants: Best Practices

### Do

1. Read the relevant sub-CLAUDE.md before making changes
2. Check `megatron/core/` first - this is the main library
3. Run `./tools/autoformat.sh` before committing
4. Add tests for new functionality
5. Use config-driven patterns (dataclasses, specs)
6. Think parallelism-first when adding features
7. Check existing patterns before inventing new ones

### Don't

- Touch `megatron/legacy/` unless necessary
- Format code outside your changes
- Add features without tests
- Break backward compatibility
- Create large PRs that do multiple things
- Hardcode values instead of using configs

### When in Doubt

1. Check similar implementations in the codebase
2. Read the tests to understand expected behavior
3. Look at `examples/` for usage patterns
4. Consult `MOE_TRAINING_GUIDE.md` for MoE-specific guidance

---

## Resources

**Documentation**:
- Official: https://docs.nvidia.com/Megatron-Core/
- GitHub: https://github.com/NVIDIA/Megatron-LM
- MoE Guide: `/MOE_TRAINING_GUIDE.md`

**Related Projects**:
- Megatron Bridge (HF checkpoint conversion)
- NeMo Framework (enterprise wrapper)
- Transformer Engine (FP8 kernels)
- Megatron Energon (multi-modal data)

**Maintainers**: @jaredcasper, @jon-barker

---

*For detailed context on any subsystem, read the corresponding sub-CLAUDE.md file listed above.*
