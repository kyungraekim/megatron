# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Megatron-LM** is NVIDIA's GPU-optimized library for training transformer models at scale. The project consists of:

- **Megatron Core** (`megatron/core/`): Production-ready composable library with GPU-optimized building blocks, parallelism strategies, and model architectures
- **Megatron-LM** (root): Reference implementation with end-to-end training scripts, examples, and utilities

This codebase is designed for distributed training of large language models (billions to trillions of parameters) across multiple GPUs and nodes.

## Architecture & Key Components

### Directory Structure

```
megatron/
├── core/                          # Megatron Core library
│   ├── models/                    # Model architectures (GPT, LLaMA, Mixtral, Mamba, etc.)
│   ├── transformer/               # Transformer building blocks (attention, MLP, layer specs)
│   ├── tensor_parallel/           # Tensor parallelism implementation
│   ├── pipeline_parallel/         # Pipeline parallelism schedules
│   ├── distributed/               # Distributed training (FSDP, DDP, process groups)
│   ├── optimizer/                 # Distributed optimizers and schedulers
│   ├── datasets/                  # High-performance dataset loaders
│   ├── inference/                 # Inference engines
│   └── export/                    # Model export utilities
├── training/                      # Training infrastructure (checkpointing, arguments, logging)
├── inference/                     # Inference server implementation
├── post_training/                 # RLHF and post-training methods
└── legacy/                        # Legacy components (pre-MCore)

examples/                          # Ready-to-use training examples
tools/                             # Utilities (data preprocessing, checkpoint conversion)
tests/                             # Test suite
├── unit_tests/                    # Unit tests for Megatron Core
└── functional_tests/              # End-to-end functional tests
```

### Core Concepts

**Parallelism Strategies**: Megatron implements multiple parallelism dimensions that can be combined:
- **Tensor Parallelism (TP)**: Splits individual layers across GPUs. When using TP with MoE models, Sequence Parallelism must be enabled.
- **Pipeline Parallelism (PP)**: Splits model depth across GPUs with microbatch pipelining
- **Data Parallelism (DP)**: Replicates model across GPUs. Supports FSDP/ZeRO for sharding optimizer states, gradients, and parameters.
- **Context Parallelism (CP)**: Splits sequence dimension for long context training
- **Expert Parallelism (EP)**: Distributes MoE experts across GPUs. Must use Sequence Parallelism when combined with TP.

**Model Architectures**: Core abstractions in `megatron/core/models/`:
- Each model (GPT, LLaMA, Mixtral, etc.) is built from composable transformer blocks
- Models use "layer specs" to define their architecture (e.g., `get_gpt_layer_local_spec()`)
- TransformerConfig holds model hyperparameters and training configurations

**Distributed State**:
- `parallel_state.py` manages process group initialization and distributed state
- Must call `parallel_state.initialize_model_parallel()` before creating models
- Process groups are created for each parallelism dimension

**Checkpointing**:
- Distributed checkpointing in `megatron/core/dist_checkpointing/` handles saving/loading across parallel dimensions
- Checkpoint format: `--ckpt-format torch_dist` for PyTorch distributed format

## Common Commands

### Environment Setup

```bash
# Install Megatron Core for development
pip install megatron-core[dev]
pip install --no-build-isolation transformer-engine[pytorch]

# Or install from source with uv (for Mamba/hybrid models)
export UV_VERSION=0.7.2
curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
export UV_PROJECT_ENVIRONMENT=./venv
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
bash docker/common/install.sh --environment dev  # or lts
```

### Testing

```bash
# Run unit tests (requires distributed setup)
# Single test file
torchrun --nproc_per_node=8 -m pytest tests/unit_tests/test_file.py -xvs

# Run with coverage
torchrun --nproc_per_node=8 -m pytest \
    --cov=megatron/core \
    --cov-report=term \
    tests/unit_tests/

# Run specific test
torchrun --nproc_per_node=8 -m pytest tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_forward -xvs

# CI test script (used in CI pipeline)
cd tests/unit_tests
bash run_ci_test.sh --tag latest --environment dev --bucket tests/unit_tests/ --log-dir ./logs
```

### Data Preprocessing

```bash
# Preprocess JSONL data for training
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

### Training

```bash
# Simple training example with mock data
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# LLaMA-3 8B training with FP8 (8 GPUs)
bash examples/llama/train_llama3_8b_h100_fp8.sh /path/to/checkpoints /path/to/logs

# GPT training with custom parallelism
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --context-parallel-size 2 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --data-path /path/to/data \
    # ... additional training args
```

### Linting & Formatting

```bash
# Format code (black, isort, pylint run on megatron/core only)
pre-commit run --all-files

# Or manually:
black --skip-magic-trailing-comma --skip-string-normalization megatron/core/
isort megatron/core/
pylint megatron/core/

# Check with flake8
flake8 megatron/core/
```

## Development Guidelines

### Code Style
- **Formatting**: Code in `megatron/core/` follows Black formatting with custom flags (`--skip-magic-trailing-comma --skip-string-normalization`)
- **Imports**: Use isort for import ordering. Megatron is first-party, transformer-engine is third-party.
- **Line length**: 100 characters
- **Type hints**: Encouraged for new code
- **Docstrings**: Google-style docstrings are required for public APIs

### Testing Requirements
- All new features in Megatron Core require unit tests in `tests/unit_tests/`
- Tests must work with distributed training setup (use `torchrun`)
- Use pytest markers:
  - `@pytest.mark.internal` for internal/private function tests
  - `@pytest.mark.flaky` for tests that are flaky in LTS environment
  - `@pytest.mark.flaky_in_dev` for tests that are flaky in DEV environment
- Test coverage is measured and reported via pytest-cov

### Parallelism Rules
- **TP + MoE**: When combining Tensor Parallelism with Expert Parallelism, Sequence Parallelism MUST be enabled
- **Communication overlap**: Use `--overlap-grad-reduce`, `--overlap-param-gather` for DP, `--tp-comm-overlap` for TP
- **FSDP**: Megatron's custom FSDP (`--use-custom-fsdp`) is ~15% faster than PyTorch FSDP2

### Model Development
- New models should be added to `megatron/core/models/`
- Define model architecture using TransformerConfig and layer specs
- Inherit from appropriate base classes and implement required methods
- Add checkpoint converters in `tools/checkpoint/` for HuggingFace compatibility

### Distributed Training Patterns
```python
# Initialize distributed training
from megatron.core import parallel_state

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=2
)

# Build model with parallelism
model = GPTModel(
    config=transformer_config,
    transformer_layer_spec=get_gpt_layer_local_spec(),
    vocab_size=50257,
    parallel_output=True
)

# Get forward-backward function for pipeline parallelism
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
forward_backward_func = get_forward_backward_func()
```

### Checkpointing Patterns
```python
from megatron.core import dist_checkpointing

# Save checkpoint
dist_checkpointing.save(
    sharded_state_dict=model.sharded_state_dict(),
    checkpoint_dir=checkpoint_path
)

# Load checkpoint
dist_checkpointing.load(
    sharded_state_dict=model.sharded_state_dict(),
    checkpoint_dir=checkpoint_path
)
```

## Common Issues & Solutions

### NaN/Inf in Training
- Check `--check-for-nan-in-loss-and-grad` flag is enabled
- For FP8 training issues, refer to [Transformer Engine docs](https://github.com/NVIDIA/TransformerEngine)
- Adjust learning rate and initialization for stability

### Memory Issues
- Enable activation checkpointing: `--recompute-activations`
- For extreme cases: `--recompute-granularity full --recompute-method uniform`
- Use distributed optimizer: `--use-distributed-optimizer`
- Use FSDP with full sharding: `--data-parallel-sharding-strategy optim_grads_params`

### Performance Optimization
- Enable communication overlap flags
- Use FP8 training on Hopper/Ada/Blackwell GPUs: `--fp8-hybrid`
- Use FlashAttention (default via Transformer Engine)
- Enable grouped GEMM for MoE: `--moe-grouped-gemm`
- Use virtual pipeline parallelism for better load balancing: `--virtual-pipeline-model-parallel-size`

### Multi-Node Training
- Set NCCL environment variables appropriately
- Use `--distributed-backend nccl`
- Configure master address and port for torch.distributed
- Be aware that `CUDA_MAX_CONNECTIONS=1` should NOT be set when using FSDP2

## Related Documentation

- [Megatron Core Documentation](https://docs.nvidia.com/Megatron-Core/)
- [NeMo Framework Performance Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)
- [MoE Training Guide](megatron/core/transformer/moe/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Project Maintenance

- Development happens internally at NVIDIA and is synced to GitHub
- PRs from external contributors are first merged internally, then pushed to GitHub with credit
- Small bug fixes are welcomed; large architectural changes should be discussed in issues first
- Maintainers: @jaredcasper, @jon-barker
- Response time: ~1 week for acknowledgment, prioritizing bug fixes and regressions
