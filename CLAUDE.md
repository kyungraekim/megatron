# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Megatron-LM** is NVIDIA's GPU-optimized library for training transformer models at scale. The repository contains two main components:

1. **Megatron Core** (`megatron/core/`) - Production-ready library with composable building blocks for custom training frameworks
2. **Megatron-LM** (top-level) - Reference implementation with complete training scripts and examples

## Development Commands

### Testing

```bash
# Run all unit tests (requires distributed setup)
pytest tests/unit_tests/ -v

# Run specific test file
pytest tests/unit_tests/transformer/test_attention.py -v

# Run tests with experimental features enabled
pytest tests/unit_tests/ -v --experimental

# Run single test with proper distributed environment
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 pytest tests/unit_tests/test_basic.py -v

# Run simple training example (requires 2 GPUs)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

### Code Quality

```bash
# Format code (black is configured in pyproject.toml)
black megatron/core/ tests/unit_tests/ --skip-magic-trailing-comma --skip-string-normalization

# Sort imports
isort megatron/core/

# Lint with pylint
pylint megatron/core/

# Lint with flake8
flake8

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Installation

```bash
# Install from source with dev dependencies
pip install -e .[dev]

# Install for development with uv (preferred in CI)
bash docker/common/install.sh --environment dev

# Install minimal version (PyTorch only)
pip install -e .
```

## Architecture Overview

### Core Architectural Pattern: ModuleSpec

Megatron uses a **declarative module construction pattern** throughout the codebase:

1. **ModuleSpec** - Defines what module to create and its submodules
2. **Submodules dataclass** - Specifies structure of nested components
3. **build_module()** - Instantiates modules from specs with configuration
4. **BackendSpecProvider** - Abstracts backend differences (Transformer Engine vs local PyTorch)

**Example:**
```python
# Define submodules structure
@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None

# Create spec with backend provider
mlp_spec = ModuleSpec(
    module=MLP,
    submodules=MLPSubmodules(
        linear_fc1=backend.column_parallel_linear(),  # TE or local
        linear_fc2=backend.row_parallel_linear(),
    ),
)

# Use build_module() to instantiate
self.mlp = build_module(mlp_spec, config=self.config)
```

### Key Directories

- `megatron/core/transformer/` - Transformer building blocks (attention, MLP, layers)
- `megatron/core/models/` - Model implementations (GPT, LLaMA, T5, Mamba, multimodal, etc.)
- `megatron/core/tensor_parallel/` - Tensor parallelism implementation
- `megatron/core/pipeline_parallel/` - Pipeline parallelism and schedules
- `megatron/core/distributed/` - Distributed training (FSDP, DDP, process groups)
- `megatron/core/datasets/` - High-performance dataloaders
- `megatron/core/optimizer/` - Distributed optimizers

### Backend Abstraction

The codebase supports two backend implementations:

1. **Transformer Engine (TE)** - High-performance kernels with FP8 support (`megatron/core/extensions/transformer_engine_spec_provider.py`)
2. **Local PyTorch** - Pure PyTorch fallback (`megatron/core/models/backends.py`)

Backend providers expose methods like:
- `column_parallel_linear()` - Returns TE or local linear layer spec
- `row_parallel_linear()`
- `layernorm()`

### TransformerLayer Architecture

The `TransformerLayer` (`megatron/core/transformer/transformer_layer.py`) is the core building block:

```
TransformerLayer
├── input_layernorm (ModuleSpec)
├── self_attention (ModuleSpec → Attention or MultiLatentAttention)
├── pre_mlp_layernorm (ModuleSpec or IdentityOp)
├── mlp (ModuleSpec → MLP or MOE)
├── per_layer_embedding (ModuleSpec or IdentityOp)  # Optional feature
└── Various layernorms for post-attention/pre-cross-attention
```

All submodules follow the ModuleSpec pattern for consistency and backend flexibility.

### Model Construction Flow

1. Define `TransformerConfig` with model hyperparameters
2. Create layer specs using `get_gpt_layer_with_transformer_engine_spec()` or `get_gpt_layer_local_spec()`
3. Build model (e.g., `GPTModel`) with config and layer spec
4. Use `build_module()` to recursively instantiate all components

### Parallelism Strategies

Megatron implements multiple parallelism dimensions:

- **Tensor Parallelism (TP)** - Split layers across GPUs (with sequence parallelism)
- **Pipeline Parallelism (PP)** - Split model depth with microbatch pipelining
- **Data Parallelism (DP)** - Replicate model with gradient synchronization
- **Context Parallelism (CP)** - Split long sequences across GPUs
- **Expert Parallelism (EP)** - For Mixture-of-Experts models

Configuration via `parallel_state.initialize_model_parallel(tp_size, pp_size)` and process group setup.

## Important Development Patterns

### Adding New Transformer Components

When adding new transformer components (like attention mechanisms, MLP variants):

1. **Create the module** in `megatron/core/transformer/` following MegatronModule pattern
2. **Define Submodules dataclass** specifying internal components as ModuleSpec
3. **Create spec helper** in relevant `*_layer_specs.py` file (e.g., `get_my_component_spec_for_backend()`)
4. **Update layer spec functions** to conditionally include your component
5. **Add to TransformerLayerSubmodules** if it's a layer-level component
6. **Use build_module()** for instantiation, not direct construction

### Layer Spec Functions

When modifying layer spec functions (`get_gpt_layer_with_transformer_engine_spec()`, `get_gpt_layer_local_spec()`):

- These functions return `TransformerLayerSubmodules` dataclass
- Add parameters to control which components are included
- Use `IdentityOp` for disabled/optional components (zero overhead)
- Keep TE and local specs in sync

### Configuration Validation

When adding config options to `TransformerConfig`:

- Add validation logic in `_validate_config()` method
- Check for incompatible option combinations
- Provide clear error messages
- Document requirements in docstrings

### Conditional kwargs Pattern

When passing module-specific parameters through `build_module()`:

```python
additional_kwargs = {}
if isinstance(submodules.my_module, ModuleSpec):
    if submodules.my_module.module == MyModule:
        additional_kwargs["special_param"] = value

self.my_module = build_module(
    submodules.my_module,
    config=self.config,
    **additional_kwargs
)
```

This allows graceful handling when module is `IdentityOp` or different type.

## Code Style Guidelines

### Formatting
- Black with `--skip-magic-trailing-comma --skip-string-normalization` (line length 100)
- isort for import sorting
- Pre-commit hooks enforce formatting on `megatron/core/` and `tests/unit_tests/`

### Naming Conventions
- Module files: snake_case (e.g., `transformer_layer.py`)
- Class names: PascalCase (e.g., `TransformerLayer`)
- Functions/methods: snake_case (e.g., `get_gpt_layer_spec()`)
- Constants: UPPER_SNAKE_CASE

### Documentation
- Comprehensive docstrings for all public APIs (Google style)
- Architecture diagrams in dataclass docstrings when helpful
- Inline comments explaining non-obvious design decisions

## Testing Patterns

### Unit Test Structure

Tests follow pytest conventions with distributed setup:

```python
class TestMyFeature:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)  # TP, PP sizes
        self.config = TransformerConfig(...)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_my_feature(self):
        # Test implementation
        pass
```

### Running Tests with Distribution

Most tests require proper distributed environment variables:
```bash
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 pytest path/to/test.py -v
```

### Test Data

Test data is automatically downloaded to `/opt/data` via pytest fixtures in `conftest.py`.

### GPU Tests

Use `@pytest.mark.skipif(not torch.cuda.is_available())` for GPU-only tests.

## Common Gotchas

### Per-Layer Embeddings (PLE)
- Must set `add_bias_linear=False` when using `use_per_layer_embeddings=True`
- Requires `hidden_size_per_layer_input` to be set
- Access layer embeddings via `self.per_layer_embedding.layer_embedding`, not directly

### Sequence Parallelism
- Required when combining Expert Parallelism (EP) with Tensor Parallelism (TP)
- Enable with `--sequence-parallel` flag

### Recompute/Checkpointing
- FP8 models use `te_checkpoint()` instead of `tensor_parallel.checkpoint()`
- Pass `distribute_saved_activations` and RNG tracker for proper determinism

### Mixed Precision
- FP8 requires Hopper/Ada/Blackwell GPUs and Transformer Engine
- BF16 recommended over FP16 for large models
- Configure via `config.fp16`, `config.bf16`, or `config.fp8`

### IdentityOp Pattern
- Used for disabled optional components (returns input unchanged)
- Follows same forward() signature as actual implementation
- No performance overhead when component disabled

## Recent Refactorings

### Per-Layer Embeddings Refactoring (Oct 2025)
- Converted from direct instantiation to ModuleSpec pattern
- New module: `megatron/core/transformer/per_layer_embedding.py`
- Updated `transformer_layer.py` to use `build_module()` for PLE
- See `PLE_REFACTORING_SUMMARY.md` for detailed documentation

This refactoring exemplifies the preferred pattern for all transformer components.

## CI/CD and Version Control

### Pre-commit Hooks
- Black formatting on `megatron/core/` and `tests/unit_tests/`
- Pylint on `megatron/core/`
- isort on `megatron/core/`

### GitLab CI
- Primary CI runs on GitLab (`.gitlab-ci.yml`)
- Functional tests in `tests/functional_tests/`
- GPU-based testing infrastructure

### Branching
- Main development branch: `main`
- Feature branches follow standard naming
- PRs first go to NVIDIA internal repo, then pushed to public GitHub

## Additional Resources

- Main README: `README.md` - Installation, performance benchmarks, parallelism strategies
- Contributing guide: `CONTRIBUTING.md` - Issue/PR policies, code submission guidelines
- Megatron Core README: `megatron/core/README.md` - Core library overview
- API Documentation: https://docs.nvidia.com/Megatron-Core/
- Examples: `examples/` directory with model-specific training scripts
