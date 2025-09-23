# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Megatron-LM repository containing both Megatron-LM (research framework) and Megatron-Core (production library) for training large-scale transformer models. Megatron-LM is NVIDIA's research framework for training large language models with optimized GPU techniques, while Megatron-Core provides the core building blocks as a production-ready library.

## Common Development Commands

### Testing
- `pytest tests/` - Run all tests
- `pytest tests/unit_tests/` - Run unit tests only
- `pytest tests/functional_tests/` - Run functional tests only
- `pytest -m "not internal"` - Run tests excluding internal markers
- `pytest -x` - Stop on first failure

### Code Quality and Linting
- `python tools/linter.py` - Run custom linting (checks line length, docstrings, imports)
- `pylint megatron/` - Run pylint with project configuration (.pylintrc)
- `flake8 megatron/` - Run flake8 linting with project configuration (.flake8)
- `mypy megatron/` - Run type checking with mypy (mypy.ini configuration)
- `python tools/autoformat.sh` - Auto-format code (uses black/isort from pyproject.toml)

### Data Preprocessing
- `python tools/preprocess_data.py` - Preprocess training data into binary format
- `python tools/merge_datasets.py` - Merge multiple datasets

### Training Commands
- Single GPU training: Use scripts in `examples/` directory
- Multi-GPU training: Use `torchrun` with distributed training scripts
- Key training scripts:
  - `pretrain_gpt.py` - GPT model pretraining
  - `pretrain_bert.py` - BERT model pretraining
  - `pretrain_t5.py` - T5 model pretraining
  - `pretrain_vlm.py` - Vision-Language model pretraining
  - `pretrain_mamba.py` - Mamba model pretraining

### Inference and Text Generation
- `python tools/run_text_generation_server.py` - Start text generation server
- `python tools/text_generation_cli.py` - CLI client for text generation server
- See `examples/inference/` for inference examples

### Checkpoint Management
- `python tools/checkpoint/convert.py` - Convert between checkpoint formats and model parallelism configurations

## Code Architecture

### Core Structure
- **`megatron/core/`** - Megatron-Core library with production-ready components
- **`megatron/legacy/`** - Legacy Megatron-LM code (older model implementations)
- **`megatron/training/`** - Training utilities and argument parsing
- **`megatron/inference/`** - Inference and text generation components

### Key Model Classes
- **`megatron/core/models/gpt/gpt_model.py`** - Main GPT model implementation
- **`megatron/core/models/bert/bert_model.py`** - BERT model implementation
- **`megatron/core/models/t5/t5_model.py`** - T5 model implementation
- **`megatron/core/models/multimodal/`** - Vision-language models
- **`megatron/core/models/mamba/`** - Mamba architecture models

### Parallelism Components
- **`megatron/core/tensor_parallel/`** - Tensor model parallelism
- **`megatron/core/pipeline_parallel/`** - Pipeline model parallelism
- **`megatron/core/distributed/`** - Data parallelism and distributed training
- **`megatron/core/transformer/moe/`** - Mixture of Experts parallelism

### Transformer Building Blocks
- **`megatron/core/transformer/transformer_block.py`** - Core transformer block
- **`megatron/core/transformer/attention/`** - Attention mechanisms
- **`megatron/core/transformer/mlp/`** - Feed-forward network implementations
- **`megatron/core/models/common/embeddings/`** - Embedding layers

### Configuration
- Use `TransformerConfig` class for model configuration
- Training arguments handled by `megatron/training/arguments.py`
- See example scripts in `examples/` for typical configurations

## Development Environment

### Docker Setup
The project is designed to work with NVIDIA PyTorch containers:
```bash
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/megatron:/workspace/megatron nvcr.io/nvidia/pytorch:xx.xx-py3
```

### Dependencies
- Check `requirements/pytorch_24.10` for PyTorch container requirements
- `requirements_ci.txt` for CI-specific dependencies
- Install with `pip install -e .` for development

### Example Training Commands
For single-GPU training (debugging):
- `bash examples/gpt3/train_gpt3_175b_distributed.sh`
- `bash examples/bert/train_bert_340m_distributed.sh`
- `bash examples/t5/train_t5_220m_distributed.sh`

For multi-GPU distributed training:
- Use `torchrun` launcher with appropriate tensor/pipeline parallelism flags
- Set `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`, `--sequence-parallel`

## Key Development Notes

### Model Parallelism
- Tensor parallelism: splits individual transformer layers across GPUs
- Pipeline parallelism: splits model depth across GPUs with micro-batching
- Sequence parallelism: works with tensor parallelism for memory efficiency
- Expert parallelism: for Mixture of Experts models

### Memory Optimization
- Use `--recompute-activations` for selective activation checkpointing
- Use `--use-distributed-optimizer` for optimizer state distribution
- FlashAttention available with `--use-flash-attn`

### Reproducibility
- Use `--deterministic-mode` for bitwise reproducible training
- Set `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` when using Transformer Engine
- Avoid FlashAttention for deterministic runs

### Data Format
- Training data should be in loose JSON format, one sample per line
- Use `tools/preprocess_data.py` to convert to binary format
- Different preprocessing for BERT (sentence splitting) vs GPT (document level)

## Testing Strategy

### Test Organization
- `tests/unit_tests/` - Fast unit tests for individual components
- `tests/functional_tests/` - End-to-end training and convergence tests
- Use markers: `internal` for private functions, `flaky` for unstable tests

### Running Tests
- Run full test suite: `pytest tests/`
- Run specific test categories: `pytest -m "not flaky"`
- For CI environments, tests are configured with `pytest.ini` markers

## Code Style

### Formatting
- Line length: 100 characters (configured in .flake8, .pylintrc, pyproject.toml)
- Use black formatter with configuration in pyproject.toml
- Import sorting with isort (black-compatible profile)

### Documentation
- Pylint enforces docstrings for classes (C0115) and functions (C0116)
- Follow existing code style when making changes
- Use proper type hints where applicable (mypy configured in mypy.ini)