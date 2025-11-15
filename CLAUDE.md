# CLAUDE.md - AI Assistant Guide for Megatron-LM

> **Last Updated**: 2025-11-15
> **Repository**: NVIDIA Megatron-LM
> **Version**: 0.14.0 (Megatron Core)

This document provides AI assistants with essential context about the Megatron-LM codebase, development workflows, architectural patterns, and key conventions. Use this as a comprehensive reference when working on tasks in this repository.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Workflow](#development-workflow)
4. [Code Style & Conventions](#code-style--conventions)
5. [Testing Infrastructure](#testing-infrastructure)
6. [Key Architectural Patterns](#key-architectural-patterns)
7. [Parallelism Strategies](#parallelism-strategies)
8. [Common Tasks & Patterns](#common-tasks--patterns)
9. [Important Files & Entry Points](#important-files--entry-points)
10. [Dependencies & Build System](#dependencies--build-system)
11. [CI/CD & Automation](#cicd--automation)
12. [Contributing Guidelines](#contributing-guidelines)

---

## Project Overview

### What is Megatron-LM?

Megatron-LM is NVIDIA's GPU-optimized library for training transformer models at scale. It consists of two main components:

1. **Megatron Core** (`megatron/core/`) - Production-ready composable library with GPU-optimized building blocks
2. **Megatron-LM** (everything else) - Reference implementation with training scripts, examples, and utilities

### Key Capabilities

- **Massive Scale Training**: Efficiently trains models from 2B to 671B+ parameters
- **Advanced Parallelism**: Supports TP, PP, DP, CP, EP with sophisticated scheduling
- **High Performance**: Achieves up to 47% Model FLOP Utilization (MFU) on H100 clusters
- **Production-Ready**: Complete inference engine, checkpointing, and fault tolerance
- **Multi-Modal Support**: Text, vision, video, and multi-modal models
- **MoE Expertise**: State-of-the-art Mixture of Experts implementations (DeepSeek-V3, Mixtral, Qwen)

### Target Use Cases

- Training foundation models (GPT, LLaMA, DeepSeek, Qwen, T5, BERT, Mamba)
- Multi-modal model training (LLaVA, NVLM, MiMo)
- Post-training (RLHF, DPO, distillation)
- High-performance inference at scale
- Research on distributed training techniques

---

## Repository Structure

### Top-Level Layout

```
Megatron-LM/
├── megatron/                   # Main source code
│   ├── core/                   # Megatron Core library (CORE COMPONENT)
│   ├── training/               # Training orchestration
│   ├── legacy/                 # Backward compatibility code
│   ├── post_training/          # Post-training algorithms
│   └── rl/                     # Reinforcement learning framework
├── examples/                   # Ready-to-use training examples
├── tests/                      # Comprehensive test suite
│   ├── unit_tests/             # Unit tests
│   ├── functional_tests/       # End-to-end tests
│   └── test_utils/             # Test utilities and recipes
├── tools/                      # Utility scripts
├── docs/                       # Documentation
├── pretrain_*.py               # Training entry points
├── train_rl.py                 # RL training entry point
├── pyproject.toml              # Package configuration
└── setup.py                    # Build configuration
```

### Megatron Core (`megatron/core/`) - The Heart of the System

**18 major subdirectories, ~200+ Python files**

#### Parallelism Infrastructure

- **`tensor_parallel/`** (8 files) - Tensor parallelism implementation
  - `layers.py` - TP-aware layers (Linear, Embedding)
  - `mappings.py` - Tensor split/gather operations
  - `random.py` - Synchronized random state across TP

- **`pipeline_parallel/`** (6 files) - Pipeline parallelism
  - `schedules.py` (100KB+) - 1F1B, interleaved, and custom schedules
  - `p2p_communication.py` - Point-to-point communication primitives

- **`distributed/`** (8 files + fsdp/) - Data parallelism
  - `param_and_grad_buffer.py` - Gradient/parameter buffer management
  - `distributed_data_parallel.py` - DDP implementation
  - `fsdp/` - Fully Sharded Data Parallel (custom + PyTorch FSDP2)

#### Model Architectures (`models/`)

**10 model architecture families:**

```
models/
├── gpt/                 # GPT models with MoE variants
├── bert/                # BERT models
├── T5/                  # T5 encoder-decoder
├── mamba/               # State-space models (Mamba)
├── retro/               # Retrieval-augmented models
├── vision/              # Vision transformers (CLIP, RADIO, ViT)
├── multimodal/          # LLaVA and multimodal models
├── huggingface/         # HuggingFace integrations
├── mimo/                # Multi-input multi-output (video VLMs)
└── common/              # Shared components
    ├── embeddings/      # RoPE, YaRN, language/vision embeddings
    ├── language_module/ # Base language module
    └── vision_module/   # Base vision module
```

#### Transformer Core (`transformer/`)

**18 files including:**

- `transformer_config.py` (77KB) - Main configuration dataclass
- `attention.py` (48KB) - Multi-head, grouped-query attention
- `transformer_layer.py` (38KB) - Individual transformer layers
- `transformer_block.py` (36KB) - Stacked transformer blocks
- `mlp.py` (16KB) - MLP/FFN layers with various activations
- `multi_latent_attention.py` (38KB) - Multi-latent attention (DeepSeek-V3)
- **`moe/`** subdirectory - Mixture of Experts
  - `experts.py` (54KB) - Expert implementations
  - `token_dispatcher.py` (57KB) - Token routing and dispatching
  - `router.py` (23KB) - Router implementations
  - Supports GroupedGEMM, aux-loss-free balancing, DeepSeek-V3

#### Other Core Components

- **`inference/`** - Production inference engine
  - `engines/` - Dynamic, static, and MCore-specific engines
  - `model_inference_wrappers/` - GPT, T5, multimodal wrappers
  - `text_generation_controllers/` - Generation logic (68KB controller)
  - `text_generation_server/` - Server infrastructure

- **`datasets/`** (19 files) - Data pipeline
  - `indexed_dataset.py` (34KB) - Binary indexed datasets
  - `blended_megatron_dataset_builder.py` (23KB) - Dataset blending
  - Model-specific datasets: `gpt_dataset.py`, `bert_dataset.py`, `t5_dataset.py`
  - C++ helpers for performance (`helpers.cpp`)

- **`optimizer/`** (7 files) - Optimization
  - `distrib_optimizer.py` (126KB) - Distributed optimizer (ZeRO-style)
  - `optimizer.py` (57KB) - Base optimizer wrapper
  - `clip_grads.py` - Gradient clipping utilities
  - `cpu_offloading/` - CPU offload for memory efficiency

- **`dist_checkpointing/`** (14 files) - Distributed checkpointing
  - `strategies/` - fully_parallel, torch, zarr, tensorstore, two_stage
  - Async checkpointing, model resharding capabilities

- **`fusions/`** (12 files) - CUDA kernel fusions
  - Fused LayerNorm, SwiGLU, GELU, cross-entropy, softmax
  - MLA YaRN RoPE fusion

- **`extensions/`** - Third-party integrations
  - `transformer_engine.py` (88KB) - NVIDIA Transformer Engine integration
  - `kitchen.py` (44KB) - Kitchen sink optimizations

- **`export/`** - Model export
  - `trtllm/` - TensorRT-LLM export support

- **`quantization/`** - FP4, FP8 quantization support

### Training Orchestration (`megatron/training/`)

**17 files + subdirectories:**

- `training.py` (125KB) - **Main training loop** (forward/backward, optimizer steps)
- `arguments.py` (204KB) - **Massive CLI argument parser** (all training arguments)
- `checkpointing.py` (83KB) - Training checkpoint management
- `initialize.py` (20KB) - Megatron initialization (distributed setup, seeds)
- `global_vars.py` (9KB) - Global state management
- `yaml_arguments.py` (21KB) - YAML configuration support
- `theoretical_memory_usage.py` (14KB) - Memory estimation
- `tokenizer/` - Tokenizer implementations (GPT2, BERT, SentencePiece, HF)
- `datasets/` - Data samplers and SFT datasets

### Entry Points (Root Level)

**Training scripts:**

- `pretrain_gpt.py` - GPT pretraining
- `pretrain_bert.py` - BERT pretraining
- `pretrain_t5.py` - T5 pretraining
- `pretrain_mamba.py` - Mamba pretraining
- `pretrain_retro.py` - RETRO pretraining
- `pretrain_vlm.py` - Vision-Language Model pretraining
- `pretrain_vision_*.py` - Vision model variants
- `train_rl.py` - RL training (PPO, RLHF)

### Examples Directory

```
examples/
├── gpt3/                    # GPT-3 training examples
├── llama/                   # LLaMA training (including FP8)
├── mixtral/                 # Mixtral MoE training
├── t5/                      # T5 training
├── bert/                    # BERT training
├── mamba/                   # Mamba training
├── retro/                   # RETRO training
├── multimodal/              # Multi-modal training (LLaVA, NVLM)
├── mimo/                    # MiMo video VLM
├── inference/               # Inference server examples
└── rl/                      # RL environment configs
```

### Tools Directory

- `preprocess_data.py` (16KB) - Data preprocessing tool
- `preprocess_mmdata.py` - Multi-modal data preprocessing
- `autoformat.sh` - Code formatting script
- `linter.py` - Linting utilities
- `checkpoint/` - Checkpoint conversion utilities
- `retro/` - RETRO-specific tools
- `run_text_generation_server.py` - Text generation server launcher
- `run_inference_performance_test.py` - Inference benchmarking

---

## Development Workflow

### Internal Development Model

**CRITICAL**: Megatron-LM is developed internally at NVIDIA and synced to GitHub.

1. Development happens in NVIDIA's internal repository
2. PRs from external contributors are pulled into internal repo
3. Changes are tested internally and pushed back to GitHub
4. Contributors receive proper credit in commits

### Branch Strategy

- `main` - Stable release branch
- `dev` - Early access experimental features
- `core_r*` - Release branches (e.g., `core_r0.14`)
- `pull-request/[0-9]+` - PR branches

### Feature Development

When working on features:

1. **Small changes** (bug fixes) - Generally welcomed
2. **Large architectural changes** - Open an issue FIRST to discuss
3. **Stylistic changes** - Discuss first; must align with project direction
4. Work in atomic commits (one feature/fix per commit)
5. Rebase on main/dev before submitting

### Code Review Process

For PRs into `main` (enforced via CODEOWNERS):

```mermaid
flowchart LR
    A[Pre-checks] --> B[PR Tests]
    B --> C1[Expert Review]
    C1 --> C2[Final Review]
    C2 --> D[Merge]
```

1. **Pre-checks** - Tests, typing, documentation, autoformatting
2. **Expert Review** - Domain experts review (auto-assigned by CODEOWNERS)
3. **Final Review** - Final approval from core maintainers
4. **Merge** - Merged by core-adlr or core-nemo teams

For PRs into `dev`:

- One approval from `eharper@nvidia.com` or `zijiey@nvidia.com`
- More lightweight process

### Response Time Expectations

- **Acknowledgement**: Within 1 week
- **Priority**: Bugs/regressions > enhancements > support requests
- **Follow-up ping**: After 1 week if no response
- **Stale policy**: 60 days without activity

---

## Code Style & Conventions

### Formatting Tools

**All enforced via pre-commit hooks:**

```yaml
# .pre-commit-config.yaml
- black (v24.4.2)
  - Line length: 100
  - Skip magic trailing comma
  - Skip string normalization
  - Applies to: megatron/core/, tests/unit_tests/

- isort (v5.13.2)
  - Profile: black-compatible
  - Known first party: megatron
  - Known third party: transformer_engine

- pylint (v3.2.6)
  - Applies to: megatron/core/

- flake8 (v7.1.0)
  - Max line length: 100
  - Ignores: E203, E501, F401, E402, E714
```

### Code Style Guidelines

**Line Length**: 100 characters (strictly enforced)

**Import Organization** (via isort):
```python
# FUTURE
from __future__ import annotations

# STDLIB
import os
import sys

# THIRDPARTY
import torch
import transformer_engine

# FIRSTPARTY
from megatron.core import parallel_state

# LOCALFOLDER
from .config import ModelConfig
```

**Typing**: Add proper type hints to all new code
```python
from typing import Optional, List, Tuple

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    ...
```

**Docstrings**: Google-style docstrings (work in progress)
```python
def my_function(arg1: int, arg2: str) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
```

**Naming Conventions**:
- Classes: `PascalCase` (e.g., `TransformerLayer`)
- Functions/methods: `snake_case` (e.g., `forward_step`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_LEARNING_RATE`)
- Private: Prefix with `_` (e.g., `_internal_method`)

### File Headers

All new files must include Apache 2.0 license header:
```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ...
```

### Running Formatters

```bash
# Auto-format all code
./tools/autoformat.sh

# Or manually
black --skip-magic-trailing-comma --skip-string-normalization megatron/core/ tests/unit_tests/
isort megatron/core/
pylint megatron/core/
```

### What NOT to Do

- Don't format code outside the scope of your PR
- Don't include commented-out code
- Don't iterate excessively on design across multiple commits
- Don't submit code incompatible with Apache 2.0 license
- Don't touch legacy code unless necessary

---

## Testing Infrastructure

### Test Organization

```
tests/
├── unit_tests/              # Fast, isolated unit tests
│   ├── models/              # Model-specific tests
│   ├── transformer/         # Transformer component tests
│   ├── dist_checkpointing/  # Checkpointing tests
│   ├── inference/           # Inference engine tests
│   ├── pipeline_parallel/   # Pipeline parallelism tests
│   ├── fusions/             # Kernel fusion tests
│   ├── data/                # Data pipeline tests
│   └── ...
├── functional_tests/        # End-to-end integration tests
│   ├── test_cases/          # Functional test definitions
│   ├── python_test_utils/   # Python test utilities
│   └── shell_test_utils/    # Shell test utilities
└── test_utils/              # Shared test utilities
    ├── recipes/             # Test configuration recipes (YAML)
    └── python_scripts/      # Helper scripts
```

### Running Tests

**Unit tests** (via pytest):
```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test file
pytest tests/unit_tests/transformer/test_attention.py

# Run with coverage
pytest --cov=megatron.core tests/unit_tests/

# Parallel execution
pytest -n auto tests/unit_tests/
```

**Functional tests**:
```bash
# Run from functional_tests directory
cd tests/functional_tests/
pytest test_cases/
```

### Test Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
addopts = "--durations=15 -s -rA -x"
markers = [
    "internal: mark a test as internal/private",
    "flaky: mark flaky tests for LTS environment",
    "flaky_in_dev: mark flaky tests for DEV environment",
]
```

### Writing Tests

**Example unit test pattern**:
```python
import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig

class TestTransformerLayer:
    def setup_method(self, method):
        """Setup before each test method."""
        parallel_state.initialize_model_parallel()

    def teardown_method(self, method):
        """Cleanup after each test method."""
        parallel_state.destroy_model_parallel()

    @pytest.mark.parametrize("hidden_size", [256, 512, 1024])
    def test_forward_pass(self, hidden_size):
        """Test basic forward pass with various hidden sizes."""
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_attention_heads=8,
        )
        # ... test implementation
```

### Test Recipes

Configuration-driven tests using YAML recipes (`tests/test_utils/recipes/`):

```yaml
# gpt.yaml example
model_type: gpt
tensor_model_parallel_size: 2
pipeline_model_parallel_size: 1
num_layers: 12
hidden_size: 768
num_attention_heads: 12
```

---

## Key Architectural Patterns

### 1. Config-Driven Architecture

**Everything uses configuration dataclasses:**

```python
@dataclass
class TransformerConfig:
    """Configuration for transformer models."""
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_layers: int = 12
    # ... 50+ configuration options
```

Key config classes:
- `TransformerConfig` - Main model configuration (77KB!)
- `ModelParallelConfig` - Parallelism settings
- Training args via `argparse` (204KB argument parser!)

### 2. Spec-Based Model Construction

**Models built from "specs" (layer specifications):**

```python
# Define layer spec
def get_gpt_layer_spec() -> ModuleSpec:
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
            ),
            mlp=ModuleSpec(module=MLP),
        ),
    )

# Build model from spec
model = GPTModel(config=config, transformer_layer_spec=get_gpt_layer_spec())
```

**Why specs?** Allows easy customization of layer implementations without changing model code.

### 3. Parallelism-First Design

**All modules are parallelism-aware:**

- Inherit from `MegatronModule` base class
- Use `parallel_state` for process group management
- Support TP via `tensor_parallel/` utilities
- Support PP via `pipeline_parallel/` schedules
- Integrate with distributed optimizer

```python
from megatron.core import parallel_state
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

# Tensor parallelism is built-in
class MLP(MegatronModule):
    def __init__(self, config):
        self.linear_fc1 = ColumnParallelLinear(...)  # Split columns
        self.linear_fc2 = RowParallelLinear(...)     # Split rows
```

### 4. Modular & Composable

**Building blocks can be mixed and matched:**

```python
# Custom transformer with different attention
custom_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        self_attention=ModuleSpec(module=MultiLatentAttention),  # Use MLA
        mlp=ModuleSpec(module=SwitchMLP),                        # Use MoE
    ),
)
```

### 5. Extension Points

**Easy to integrate external libraries:**

- `extensions/transformer_engine.py` - Transformer Engine integration
- `extensions/kitchen.py` - Additional optimizations
- Models can use custom attention backends via `attention_backend` arg

### 6. Backward Compatibility

**Old code isolated in `legacy/`:**

- New code in `megatron/core/`
- Old code in `megatron/legacy/`
- Migration guides provided
- Both maintained for smooth transitions

---

## Parallelism Strategies

### Five Types of Parallelism

Megatron supports 5 parallelism dimensions that can be combined:

#### 1. Tensor Parallelism (TP)

**Splits individual layers across GPUs**

```bash
--tensor-model-parallel-size 4    # 4-way TP
--sequence-parallel                # Enable sequence parallelism (recommended)
```

**When to use:**
- Large models that don't fit in single GPU memory
- Typically TP ∈ {1, 2, 4, 8} within a node
- Use with sequence parallelism for better memory efficiency

**Implementation:**
- `ColumnParallelLinear` - Split output dimension
- `RowParallelLinear` - Split input dimension
- `VocabParallelEmbedding` - Split vocabulary

#### 2. Pipeline Parallelism (PP)

**Splits model depth (layers) across GPUs**

```bash
--pipeline-model-parallel-size 8           # 8 pipeline stages
--virtual-pipeline-model-parallel-size 4   # 4 virtual stages (interleaving)
```

**Schedules** (`pipeline_parallel/schedules.py`):
- **1F1B** (One-forward-one-backward) - Default, memory-efficient
- **Interleaved** - Virtual pipeline for better GPU utilization
- **Dynamic** - For inference with variable batch sizes

**When to use:**
- Very large models (100B+ parameters)
- Combine with TP for 2D parallelism
- Use virtual pipeline to reduce bubbles

#### 3. Data Parallelism (DP)

**Replicates model, splits data**

```bash
# Standard DDP
--data-parallel-sharding-strategy no_shard

# Fully Sharded Data Parallel (ZeRO)
--use-custom-fsdp                                    # Megatron FSDP (~15% faster)
--data-parallel-sharding-strategy optim              # ZeRO-1 (shard optimizer)
--data-parallel-sharding-strategy optim_grads        # ZeRO-2 (+ gradients)
--data-parallel-sharding-strategy optim_grads_params # ZeRO-3 (+ parameters)
```

**Communication optimization:**
```bash
--overlap-grad-reduce      # Overlap gradient reduction with backward
--overlap-param-gather     # Overlap parameter gathering
--use-distributed-optimizer # Distributed optimizer for faster checkpointing
```

#### 4. Context Parallelism (CP)

**Splits long sequences across GPUs**

```bash
--context-parallel-size 2                      # 2-way CP
--cp-comm-type p2p                             # Communication: p2p, a2a, allgather
--hierarchical-context-parallel-sizes 2 4      # Hierarchical CP
```

**When to use:**
- Long context lengths (8K+)
- Memory-bound on activations
- Often combined with TP

#### 5. Expert Parallelism (EP)

**For Mixture of Experts (MoE) models**

```bash
--expert-model-parallel-size 4  # 4-way EP
--num-experts 8                 # 8 experts per MoE layer
--moe-grouped-gemm              # Use grouped GEMM optimization
```

**CRITICAL**: When combining EP + TP, **Sequence Parallelism (SP) must be enabled**

### Parallelism Combination Examples

**LLaMA-3 8B (8 GPUs)**:
```bash
TP=1, PP=1, CP=2, DP=4
# Use CP for long sequences (8K)
```

**LLaMA-3 70B (64 GPUs)**:
```bash
TP=4, PP=4, CP=2, DP=2
# 2D parallelism (TP+PP)
```

**LLaMA-3.1 405B (1024 GPUs)**:
```bash
TP=8, PP=8, CP=2, DP=16
# 3D parallelism for massive scale
```

**Mixtral 8x7B (64 GPUs)**:
```bash
TP=1, PP=4, EP=8, DP=2
# EP for MoE experts
```

**DeepSeek-V3 671B (1024 GPUs)**:
```bash
TP=2, PP=16, EP=64, DP=varies
# Large MoE with heavy EP
```

### Parallelism Selection Guidelines

1. **Start with TP** for models that don't fit on 1 GPU
2. **Add PP** for very large models (100B+)
3. **Use CP** for long sequences (8K+)
4. **Use EP** for MoE models
5. **DP fills remaining GPUs** for data throughput
6. **TP should divide num_attention_heads**
7. **PP should divide num_layers**
8. **Total GPUs = TP × PP × CP × EP × DP**

---

## Common Tasks & Patterns

### Task 1: Adding a New Model Architecture

**Pattern to follow:**

1. **Create model directory** in `megatron/core/models/`
2. **Define layer specs** (see `gpt/gpt_layer_specs.py`)
3. **Implement model class** (inherit from `MegatronModule`)
4. **Add configuration** (extend `TransformerConfig` if needed)
5. **Write unit tests** in `tests/unit_tests/models/`
6. **Create training script** in root or `examples/`

**Example structure:**
```
megatron/core/models/my_model/
├── __init__.py
├── my_model_model.py           # Main model class
├── my_model_layer_specs.py     # Layer specifications
└── README.md                    # Model documentation
```

### Task 2: Adding a New Fusion Kernel

**Location**: `megatron/core/fusions/`

**Pattern:**
1. Implement CUDA kernel (or use Triton)
2. Add Python wrapper
3. Add tests in `tests/unit_tests/fusions/`
4. Integrate into relevant module (attention, MLP, etc.)

**Example**: See `fused_layer_norm.py`, `fused_softmax.py`

### Task 3: Modifying Training Arguments

**Files to modify:**
- `megatron/training/arguments.py` (204KB!) - Add argument parsing
- `megatron/core/transformer/transformer_config.py` - If model config
- `megatron/core/model_parallel_config.py` - If parallelism config

**Pattern:**
```python
# In arguments.py
group = parser.add_argument_group(title='my feature')
group.add_argument('--my-new-arg', type=int, default=42,
                   help='Description of my new argument')
```

### Task 4: Adding a New Optimizer

**Location**: `megatron/core/optimizer/`

**Pattern:**
1. Extend `MegatronOptimizer` class
2. Implement required methods
3. Handle distributed optimizer logic
4. Add tests
5. Register in training script

### Task 5: Implementing a New Parallelism Strategy

**Key files:**
- `megatron/core/parallel_state.py` - Process group management
- `megatron/core/process_groups_config.py` - Process group configuration
- Relevant parallelism directory (`tensor_parallel/`, `pipeline_parallel/`, etc.)

**This is complex** - consult with maintainers first!

### Task 6: Data Preprocessing

**For text data:**
```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

**For multimodal data:**
```bash
python tools/preprocess_mmdata.py \
    --input dataset.yaml \
    --output-prefix mm_data \
    ...
```

**Creates binary indexed datasets** (`.bin` and `.idx` files)

### Task 7: Running Inference

**Start text generation server:**
```bash
python tools/run_text_generation_server.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --load /path/to/checkpoint \
    --tokenizer-type GPT2BPETokenizer \
    ...
```

**Or use inference API directly** (see `megatron/core/inference/`)

### Task 8: Checkpointing

**Save checkpoint:**
```python
from megatron.training import checkpointing
checkpointing.save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
```

**Load checkpoint:**
```python
iteration = checkpointing.load_checkpoint(model, optimizer, opt_param_scheduler)
```

**Distributed checkpointing** (for model resharding):
```python
from megatron.core.dist_checkpointing import save, load
save(state_dict, checkpoint_dir, strategy='fully_parallel')
load(state_dict, checkpoint_dir)
```

---

## Important Files & Entry Points

### Largest/Most Complex Files

1. **`megatron/training/arguments.py`** (204KB)
   - All CLI arguments for training
   - Handles argument validation and processing
   - Multiple argument groups (model, optimizer, data, etc.)

2. **`megatron/core/optimizer/distrib_optimizer.py`** (126KB)
   - Distributed optimizer (ZeRO-style)
   - Complex state management across GPUs
   - Critical for large-scale training

3. **`megatron/training/training.py`** (125KB)
   - Main training loop
   - Forward/backward passes
   - Optimizer steps, logging, checkpointing

4. **`megatron/rl/rl_utils.py`** (103KB)
   - RL utilities (PPO, RLHF)
   - Reward computation, advantage estimation

5. **`megatron/core/pipeline_parallel/schedules.py`** (100KB)
   - Pipeline schedules (1F1B, interleaved)
   - Complex scheduling logic

6. **`megatron/core/extensions/transformer_engine.py`** (88KB)
   - Transformer Engine integration
   - FP8 support, optimized kernels

7. **`megatron/core/parallel_state.py`** (87KB)
   - Process group management
   - Parallelism state (TP, PP, DP, CP, EP)

8. **`megatron/core/transformer/transformer_config.py`** (77KB)
   - Main transformer configuration dataclass
   - 50+ configuration options

### Key Configuration Files

- **`pyproject.toml`** - Package configuration, dependencies, dev tools
- **`.pre-commit-config.yaml`** - Pre-commit hooks
- **`.flake8`** - Flake8 linting configuration
- **`.pylintrc`** - Pylint configuration
- **`setup.py`** - Build configuration (C++ extensions)

### Example Configurations

- **`examples/gpt3/gpt_config.yaml`** - GPT-3 training config
- **`examples/multimodal/pretrain_dataset.yaml`** - Multimodal dataset config
- **`examples/rl/environment_configs/*.yaml`** - RL environment configs
- **`tests/test_utils/recipes/*.yaml`** - Test recipe configs

---

## Dependencies & Build System

### Core Dependencies

**Required:**
- `torch` - PyTorch (main framework)
- `numpy<2.0.0` - Numerical computing
- `packaging>=24.2` - Version parsing

**Optional (but recommended):**
- `transformer-engine[pytorch]>=2.9.0` - FP8 and optimized kernels
- `nvidia-resiliency-ext` - Fault-tolerant training
- `megatron-energon[av_decode]~=6.0` - Multi-modal data loading
- `mamba-ssm~=2.2` - State-space models
- `causal-conv1d~=1.5` - Mamba dependency
- `nv-grouped-gemm~=1.1` - MoE optimization
- `tensorstore~=0.1` - Distributed checkpointing
- `einops~=0.8` - Tensor operations

### Installation Variants

**Megatron Core only:**
```bash
pip install megatron-core
```

**With development dependencies:**
```bash
pip install --no-build-isolation megatron-core[dev]
```

**With Megatron-LM (training scripts):**
```bash
pip install --no-build-isolation megatron-core[mlm,dev]
```

**LTS (Long-term support, NGC PyTorch 24.01):**
```bash
pip install --no-build-isolation megatron-core[mlm,lts]
```

### Build System

**Uses setuptools with optional C++ extensions:**

```python
# setup.py
ext_modules=[
    Extension(
        "megatron.core.datasets.helpers_cpp",
        sources=["megatron/core/datasets/helpers.cpp"],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17"],
        optional=True,  # Won't fail build if compilation fails
    )
]
```

**Build from source:**
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

### UV Support (New!)

**Faster dependency resolution:**
```bash
uv pip install --no-build-isolation megatron-core[dev]
```

**Packages that require `--no-build-isolation`:**
- causal-conv1d
- nv-grouped-gemm
- flash_mla
- mamba-ssm
- transformer-engine

### Docker Recommendation

**Strongly recommended**: Use NGC PyTorch containers
```bash
docker run --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

**Why Docker?**
- Pre-configured CUDA, cuDNN, NCCL
- Compatible PyTorch version
- FP8 support on Hopper/Ada/Blackwell
- Tested configurations

---

## CI/CD & Automation

### GitHub Actions Workflows

**Main CI/CD** (`.github/workflows/cicd-main.yml`):
- Runs on: push to `main`/`dev`, PRs, scheduled (daily), manual dispatch
- Checks: membership, tests, linting
- Environment: `nemo-ci`
- Concurrency control to avoid redundant runs

**Other workflows:**
- `auto-assign-milestone.yml` - Auto-assign milestones to PRs
- `auto-swap-labels.yml` - Label management
- `auto-update-copy-pr-bot.yml` - PR copying automation
- `cherry-pick-release-commit.yml` - Cherry-pick to release branches
- `copyright-check.yml` - Copyright header validation
- `install-test.yml` - Installation testing
- `build-test-publish-wheel.yml` - Wheel building and publishing

### Pre-commit Hooks

**Automatically run** (if set up):
```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**Hooks configured:**
- `black` - Code formatting
- `isort` - Import sorting
- `pylint` - Linting

### Stale PR/Issue Bot

- Marks PRs/issues as stale after **60 days** of inactivity
- Helps maintain backlog
- Can be overridden with activity

### Automated Testing

**Unit tests** run on every PR:
```bash
pytest tests/unit_tests/
```

**Functional tests** run on schedule/main:
```bash
pytest tests/functional_tests/
```

---

## Contributing Guidelines

### Before You Start

1. **Check if your change aligns with project direction**
   - Bug fixes: Generally welcomed
   - Small enhancements: Usually fine
   - Large architectural changes: **Open an issue first to discuss**

2. **Search existing issues/PRs** to avoid duplicates

3. **Set up development environment**
   ```bash
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   pip install --no-build-isolation -e .[mlm,dev]
   pip install pre-commit
   pre-commit install
   ```

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] **I want this in a versioned release** and have added appropriate Milestone (e.g., `Core 0.14`)
- [ ] **I have added relevant unit tests** (in `tests/unit_tests/`)
- [ ] **I have added relevant functional tests** (if applicable)
- [ ] **I have added proper typing** to my code
- [ ] **I have added relevant documentation** (docstrings, README updates)
- [ ] **I have run `./tools/autoformat.sh`** on my PR
- [ ] **Commits are atomic** (one feature/fix per commit)
- [ ] **Commits are rebased on main**
- [ ] **Commit messages are in imperative mood** ("Add feature X", not "Added feature X")

### PR Labels

**For PRs into `main` branch:**

1. Add `Expert Review` label when ready for review
2. Wait for expert reviewers (auto-assigned via CODEOWNERS)
3. After approvals, add `Final Review` label
4. Wait for final reviewers (auto-assigned)
5. PR will be merged by `core-adlr` or `core-nemo` teams

**Don't proceed to Final Review until:**
- All expert reviewers have approved
- Merge conflicts are resolved
- CI is passing

### Commit Message Guidelines

**Format:**
```
Brief summary line (imperative mood, <72 chars)

Longer explanation if needed. Explain the motivation, what changed,
and why it changed (not just what changed).

Fixes #123
```

**Examples:**
- ✅ "Add support for multi-latent attention"
- ✅ "Fix gradient accumulation in distributed optimizer"
- ✅ "Optimize MoE routing with fused kernels"
- ❌ "Updated the code"
- ❌ "Fixed a bug"

### What Gets Reviewed

**Expert reviewers look for:**
- Correctness and performance
- Code quality and style
- Test coverage
- Documentation completeness
- Compatibility with existing features

**Final reviewers look for:**
- Overall design alignment
- Impact on project direction
- Release readiness

### After Your PR is Merged

**Congratulations!** Your changes will:
1. Be pulled into NVIDIA's internal repository
2. Undergo internal testing
3. Be pushed back to GitHub in next sync
4. Appear in next release (if milestone set)

**Cherry-picking to release branches:**
- If your PR needs to go into a release branch (`core_r*`)
- After merge to main, use "Cherry-pick" option
- Opens new PR to release branch

### Contact

**Project maintainers:**
- @jaredcasper
- @jon-barker

**Response times:**
- Initial acknowledgement: Within 1 week
- Follow-up ping: After 1 week if no response

---

## Quick Reference: Key Commands

### Training

```bash
# Simple training example
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# GPT-3 pretraining
python pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 100000 \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab \
    --merge-file /path/to/merges \
    --split 949,50,1 \
    --fp16

# LLaMA-3 with FP8
./examples/llama/train_llama3_8b_h100_fp8.sh
```

### Data Preprocessing

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### Inference

```bash
python tools/run_text_generation_server.py \
    --tensor-model-parallel-size 1 \
    --load /path/to/checkpoint \
    --tokenizer-type GPT2BPETokenizer
```

### Testing

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test
pytest tests/unit_tests/transformer/test_attention.py -v

# Run with coverage
pytest --cov=megatron.core tests/unit_tests/
```

### Formatting

```bash
# Auto-format all code
./tools/autoformat.sh

# Or manually
black --skip-magic-trailing-comma --skip-string-normalization megatron/core/
isort megatron/core/
pylint megatron/core/
```

---

## Additional Resources

### Documentation

- **Official Docs**: https://docs.nvidia.com/Megatron-Core/
- **GitHub**: https://github.com/NVIDIA/Megatron-LM
- **Issues**: https://github.com/NVIDIA/Megatron-LM/issues
- **Changelog**: `/CHANGELOG.md`
- **Contributing**: `/CONTRIBUTING.md`

### Related Projects

- **Megatron Bridge**: https://github.com/NVIDIA-NeMo/Megatron-Bridge
  - HuggingFace ↔ Megatron checkpoint conversion
- **NeMo Framework**: https://docs.nvidia.com/nemo-framework/
  - Enterprise framework built on Megatron Core
- **NeMo RL**: https://github.com/NVIDIA-NeMo/RL
  - RLHF and DPO post-training
- **Megatron Energon**: https://github.com/NVIDIA/Megatron-Energon
  - Multi-modal data loading
- **Transformer Engine**: https://github.com/NVIDIA/TransformerEngine
  - FP8 and optimized kernels

### Performance Guides

- **NeMo Performance Guide**: https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html
- **NeMo Performance Summary**: https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html

### Roadmaps

- **MoE Q3-Q4 2025 Roadmap**: https://github.com/NVIDIA/Megatron-LM/issues/1729
- **GPT-OSS Implementation Tracker**: https://github.com/NVIDIA/Megatron-LM/issues/1739

---

## For AI Assistants: Best Practices

### When Working on This Codebase

1. **Always check `megatron/core/` first** - This is the main library
2. **Don't touch `megatron/legacy/`** unless absolutely necessary
3. **Run `./tools/autoformat.sh`** before committing
4. **Add tests** for all new functionality
5. **Add type hints** to all new code
6. **Write docstrings** for public APIs
7. **Respect the 100-character line limit** (strictly enforced)
8. **Use config-driven patterns** (dataclasses, specs)
9. **Think parallelism-first** when adding new features
10. **Check existing patterns** before inventing new ones

### Common Pitfalls to Avoid

- ❌ Formatting code outside your changes
- ❌ Adding features without tests
- ❌ Breaking backward compatibility
- ❌ Ignoring type hints
- ❌ Long commit messages without context
- ❌ Large PRs that do multiple things
- ❌ Modifying generated files
- ❌ Hardcoding values instead of using configs

### When in Doubt

1. **Check similar existing implementations** in the codebase
2. **Read the tests** to understand expected behavior
3. **Look at `examples/`** for usage patterns
4. **Ask for clarification** via GitHub issues
5. **Propose design first** for large changes

---

**Document Version**: 1.0
**Last Updated**: 2025-11-15
**Maintainers**: NVIDIA Megatron-LM Team

For questions or updates to this document, please open an issue or PR.
