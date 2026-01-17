# CLAUDE.md - examples

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Training examples: GPT, LLaMA, Mixtral, multimodal, and inference scripts
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

The `examples/` directory contains production-ready training and inference scripts demonstrating
how to use Megatron-LM and Megatron Core. These examples range from simple training loops to
complex multimodal and MoE setups.

**Key Resource**: `run_simple_mcore_train_loop.py` - Start here for the simplest training
example. This is the recommended entry point for understanding Megatron Core.

---

## Example Categories

| Category | Directory | Purpose |
|----------|-----------|---------|
| Simple Training | `run_simple_mcore_train_loop.py` | Minimal training loop using Megatron Core |
| Language Models | `gpt3/`, `llama/`, `t5/`, `bert/` | LLM pretraining scripts |
| MoE Models | `mixtral/` | Mixture-of-Experts training |
| Alternative Architectures | `mamba/`, `retro/` | SSM and retrieval-augmented training |
| Multimodal | `multimodal/` | Vision-language models (LLaVA, NVLM) |
| Video VLMs | `mimo/` | MiMo video VLM training and inference |
| Inference | `inference/` | Serving and text generation examples |
| RL | `rl/` | Environment configs for RL training |

---

## Directory Structure

```
examples/
├── run_simple_mcore_train_loop.py        # START HERE: simplest example
├── gpt3/                                 # GPT-3 pretraining configs
├── llama/                                # LLaMA training (includes FP8)
├── mixtral/                              # Mixtral MoE training
├── multimodal/                           # Vision-language training
│   ├── train.py                         # Main training script
│   ├── model.py                         # Model definition
│   ├── layer_specs.py                   # Layer specifications
│   ├── config.py                        # Configuration
│   ├── dataset_helpers.py               # Data loading
│   ├── dataloader_provider.py           # Dataloader setup
│   ├── image_processing.py              # Image preprocessing
│   └── run_text_generation.py           # Inference script
├── mimo/                                 # Video VLM
│   ├── train.py                         # Training script
│   └── avlm_inference.py                # Inference script
├── inference/                            # Inference server setup
├── t5/, bert/, mamba/, retro/          # Other architectures
└── rl/                                   # RL training configs
```

---

## Quick Start

### Run Simplest Example
```bash
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### Run GPT-3 Pretraining
```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --gpt-model-type gpt3 \
  --train-samples 10000
```

### Run Multimodal Training
```bash
python examples/multimodal/train.py \
  --config-path configs/multimodal_config.yaml
```

### Run Inference (Multimodal)
```bash
python examples/multimodal/run_text_generation.py \
  --model-path checkpoint.pt \
  --prompt "Describe this image"
```

---

## Common Patterns

**1. Config-Driven Setup**: All examples use dataclass configs
  - GPT: GPT3Config (defined in megatron/core/models/)
  - Multimodal: Config dataclass in config.py
  - See megatron/core/transformer/transformer_config.py for TransformerConfig

**2. Spec-Based Models**: Models constructed from ModuleSpec layers
  - layer_specs.py defines the architecture
  - get_model_provider() function returns a spec builder
  - Enables parallelism without code changes

**3. Data Loaders**: Custom dataloaders with Megatron parallelism awareness
  - multimodal/dataloader_provider.py shows the pattern
  - Use indexed datasets for efficiency
  - Broadcast dataloaders across ranks

**4. Parallelism Setup**: All examples initialize parallel_state
  - Initialize GPU groups before model creation
  - Use ModelParallelConfig to control TP/PP/DP/CP/EP
  - See megatron/core/distributed/ for implementation

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_simple_mcore_train_loop.py` | Minimal working example - read first |
| `multimodal/train.py` | Complex example: VLM with TP/PP |
| `multimodal/model.py` | Model architecture (attention + MLP) |
| `multimodal/layer_specs.py` | Layer customization pattern |
| `gpt3/*.py` | Full GPT-3 setup with configs |
| `llama/` | LLaMA + FP8 quantization patterns |
| `mixtral/` | MoE training setup |
| `mimo/train.py` | Video VLM with temporal modeling |

---

## Dependencies & Setup

**Installation**:
```bash
pip install --no-build-isolation -e .[mlm,dev]
pip install torch tensorboard
```

**For Multimodal**:
- PIL, torchvision (image loading)
- transformers (HuggingFace models)

**For Inference**:
- Tensor Parallelism aware text generation
- Optional: vLLM-style optimizations in inference/

**For Video (MiMo)**:
- av, ffmpeg (video decoding)
- spatial-temporal model dependencies

---

## Common Tasks

### Add a New Example
1. Create new directory: `examples/my_model/`
2. Copy structure from similar example (e.g., llama/)
3. Define config dataclass in config.py
4. Implement model in model.py
5. Use layer_specs.py pattern for architecture
6. Update pretrain_*.py at repo root or use standalone script

### Use Custom Attention or MLP
1. Modify layer_specs.py to use custom spec
2. See multimodal/layer_specs.py for pattern
3. Use TransformerConfig.use_custom_backend or extensions

### Enable FP8 Quantization
1. Set fp8=True in config
2. See llama/ examples for setup
3. Requires Tensor Engine and compatible GPU

### Add Parallelism (TP/PP/CP/EP)
1. Update model_parallel_config in config
2. Specify --tensor-model-parallel-size, --pipeline-model-parallel-size flags
3. Models automatically adapt (spec-based)
4. See gpt3/ for full parallelism examples

---

## Gotchas & Best Practices

**1. Parallelism Initialization**:
- ALWAYS initialize parallel_state before creating models
- Wrong rank assignment = silent data corruption
- Initialize_model_parallel(tensor_model_parallel_size, ...)

**2. Dataloader Broadcasting**:
- Dataloaders must broadcast batch size across ranks
- Use megatron.core.datasets utilities
- Off-by-one errors in data split = training hangs

**3. Gradient Accumulation**:
- Works with TP/PP but requires careful bookkeeping
- Verify gradient shapes match model sharding
- Test with small batch sizes first

**4. Checkpointing**:
- Distributed checkpointing required for TP/PP
- See megatron/core/dist_checkpointing/ for format
- Old single-GPU checkpoints won't load with parallelism

**5. Configuration Compatibility**:
- Not all config combinations are valid (e.g., CP + PP)
- Start simple: DP only, then add TP, then PP
- Use pretrain_*.py flags rather than manual init

**6. Testing Your Example**:
- Run on 1 GPU first to catch obvious bugs
- Then scale to 2 GPUs with TP=2
- Finally test with DP across nodes

---

## For AI Assistants

### When Modifying Examples
1. Read the relevant sub-CLAUDE.md first (e.g., transformer/CLAUDE.md for attention changes)
2. Check similar examples for patterns before inventing
3. Test locally before pushing: `python -m pytest tests/`
4. Run formatter: `./tools/autoformat.sh examples/`

### When Adding a New Model Type
1. Copy from similar example (llama -> your_model)
2. Define config in config.py
3. Implement model.py with proper specs
4. Add to pretrain_*.py entry point
5. Add unit test in tests/

### For Multimodal/Complex Examples
- Start simple, then add parallelism
- Use run_simple_mcore_train_loop.py as single-GPU reference
- Verify gradients flow before adding TP/PP
- Profile before and after parallelism changes

---

## Resources

- **Official Docs**: https://docs.nvidia.com/Megatron-Core/
- **Parent Guide**: [../CLAUDE.md](../CLAUDE.md)
- **Core Library**: [../megatron/core/CLAUDE.md](../megatron/core/CLAUDE.md)
- **Training Guide**: [../megatron/training/CLAUDE.md](../megatron/training/CLAUDE.md)
- **MoE Training**: [../MOE_TRAINING_GUIDE.md](../MOE_TRAINING_GUIDE.md)

---

*For questions about specific components (transformers, parallelism, optimization),
consult the relevant megatron/core/*/CLAUDE.md file.*
