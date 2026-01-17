# CLAUDE.md - post_training

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Post-training: ModelOpt integration, quantization, distillation
> **Parent**: [../CLAUDE.md](../../CLAUDE.md)

## Overview

`post_training/` provides high-level post-training utilities for NVIDIA Megatron-LM models,
with emphasis on **ModelOpt** (NVIDIA's model optimization toolkit) integration.

**Primary Use Cases**:
- Quantization (FP8, FP4, INT8) for inference optimization
- Knowledge distillation (teacher-student training)
- Checkpoint format conversion (NeMo, MLM, torch)
- Token generation with speculative decoding
- Loss computation with distillation support

**Key Dependency**: `modelopt.torch` (NVIDIA's optimization library)

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `model_builder.py` | 341 | ModelOpt model provider; builds GPT/Mamba with quantization specs |
| `checkpointing.py` | 196 | Load/save ModelOpt states; handle sharded & unsharded formats |
| `arguments.py` | 122 | CLI arguments for quantization, distillation, model type |
| `generate.py` | 153 | Token generation without KV-cache; speculative decoding |
| `utils.py` | 90 | Memory monitoring, quantization summary, dataset loading |
| `loss_func.py` | 72 | Loss computation with KD support and loss masking |
| `non_loss_data_func.py` | 75 | Speculative decoding metrics (acceptance length) |

---

## Architecture & Patterns

### Model Building (model_builder.py)

Entry point: `modelopt_gpt_mamba_builder(args, pre_process, post_process, config, ...)`

**Flow**:
1. Parse config from args via `core_transformer_config_from_args()`
2. Select layer spec based on model type (GPT vs Mamba) and quantization
3. Create model (MCoreGPTModel or MCoreMambaModel)
4. Load ModelOpt state if checkpoint exists (via `load_modelopt_state()`)
5. Register state_dict hooks for checkpoint compatibility
6. Enable distillation if `--export-kd-teacher-load` is set

**Key Logic**:
- Teacher model loaded in distillation mode: `_load_teacher_model_config()` +
  `_load_teacher_model()` parse NEMO-format config and load checkpoint
- ModelOpt state loaded **before** optimizer creation (important for param registration)
- Heterogeneous block specs supported for hybrid models
- GPT-OSS YaRN RoPE configuration available

### Checkpointing (checkpointing.py)

Two main workflows:

**Sharded Checkpoints** (MLM iter_XXXXXXX/ or NeMo model_weights/):
```python
load_modelopt_checkpoint(model, load_arg='load')
# -> get_sharded_load_dir() finds checkpoint type
# -> dist_checkpointing.load() for model state
# -> restore_sharded_modelopt_state() for quantization params
```

**Unsharded Checkpoints** (torch format):
```python
load_modelopt_checkpoint(model)
# -> _load_base_checkpoint() loads single file
# -> extract modelopt_state from state_dict
# -> mto.restore_from_modelopt_state()
```

Helper: `has_modelopt_state(path)` checks if checkpoint contains quantization state

### Loss Computation (loss_func.py)

`loss_func(loss_mask, output_tensor, model)` handles:
- Standard LM loss with masking
- Knowledge distillation losses (logits + intermediate) if enabled
- Tensor parallel all-reduce if needed
- Returns: (loss, num_tokens, report_dict)

### Generation (generate.py)

**simple_generate()**: Forward-only loop, no KV-cache, pads sequence to mult of 32
**simple_speculative_generate()**: Wraps simple_generate + draft acceptance checking

---

## Common Tasks

### Task: Enable Quantization on a Model

1. Add `--export-quant-cfg` or `--export-kv-cache-quant` to CLI args
2. ModelOpt spec is applied in `modelopt_gpt_mamba_builder()` via layer specs
3. Checkpoint loading via `load_modelopt_checkpoint()` restores quantization state

**Quantization Types** (from arguments.py):
- `--export-quant-cfg`: FP8, FP4, INT8 (full model)
- `--export-kv-cache-quant`: fp8, fp8_affine, nvfp4, nvfp4_affine, nvfp4_rotate
- `--export-real-quant-cfg`: FP8 with online quantization during training

### Task: Run Knowledge Distillation

1. Prepare teacher checkpoint (NeMo or MLM format)
2. Pass `--export-kd-teacher-load <path>` + `--export-kd-cfg <yaml>`
3. Distillation enabled in `modelopt_gpt_mamba_builder()`:
   - Loads teacher via `_load_teacher_model()`
   - Wraps student model via `mtd.convert(..., mode=[("kd_loss", config)])`
   - `loss_func()` computes combined KD loss

**Config Format** (yaml): See ModelOpt docs for distillation config schema

### Task: Load ModelOpt Checkpoint

```python
from megatron.post_training.checkpointing import load_modelopt_checkpoint

# In training loop after model creation:
load_modelopt_checkpoint(model, optimizer, opt_param_scheduler)
# Handles both sharded and unsharded formats transparently
```

### Task: Convert NeMo Checkpoint

Sharded NeMo checkpoints use `model_weights/` or `weights/` directory:
```python
# get_sharded_load_dir() auto-detects:
# - MLM: looks for latest_checkpointed_iteration.txt + iter_XXXXXXX/
# - NeMo: looks for model_weights/ or weights/ dirs
```

---

## Dependencies

**External**:
- `modelopt.torch` (quantization, distillation)
- `modelopt.torch.opt` (ModelOpt state management)
- `modelopt.torch.quantization` (MTQ utilities)
- `modelopt.torch.distill` (knowledge distillation)

**Internal** (Megatron):
- `megatron.core.models.gpt.GPTModel` (base model)
- `megatron.core.models.mamba.MambaModel` (Mamba support)
- `megatron.core.post_training.modelopt.*` (layer specs, hooks)
- `megatron.training.arguments` (arg parsing, config conversion)
- `megatron.training.checkpointing` (base checkpoint loading)

---

## Gotchas & Constraints

1. **ModelOpt state MUST be loaded before optimizer creation**
   - ModelOpt can add trainable params (e.g., quantization scales, PEFT)
   - Called in `modelopt_gpt_mamba_builder()` before returning to `get_model()`

2. **Distillation incompatibilities** (in model_builder.py):
   - `--manual-gc` causes memory leak per fwd-bwd pass
   - `--tp-comm-overlap` incompatible with distillation
   - Interleaved PP (`--virtual-pipeline-model-parallel-size`) not supported

3. **Sequence padding for sequence parallelism**
   - `generate.py` pads sequences to mult of 32 for sequence-parallel compatibility
   - Affects output length (may include padding tokens)

4. **Teacher config translation**
   - NeMo format fields translated to MCore in `_load_teacher_model_config()`
   - E.g., `encoder_seq_length` -> `seq_length`, `share_embeddings_and_output_weights` inverted

5. **Sharded checkpoint format detection**
   - MLM tracker file: `latest_checkpointed_iteration.txt`
   - NeMo dirs: `model_weights/` or `weights/`
   - Raises ValueError if format not recognized

6. **Checkpoint strictness**
   - ModelOpt adds/removes params during quantization
   - Load with `strict=False` to allow state_dict mismatches

---

## Extension Points

- **Custom quantization specs**: Modify `megatron.core.post_training.modelopt.*.model_specs`
- **Custom loss functions**: Extend `loss_func.py` with additional loss terms
- **Generation backends**: Replace `simple_generate()` with KV-cache variant
- **Distillation modes**: ModelOpt supports multiple distillation strategies via config yaml

---

## Testing

Unit tests in `tests/unit_tests/post_training/` (if applicable)

Key test areas:
- Checkpoint loading (sharded vs unsharded)
- Quantization state preservation
- Knowledge distillation loss computation
- Generation token validation

---

## References

- ModelOpt: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- Megatron Core: [../core/CLAUDE.md](../core/CLAUDE.md)
- Knowledge Distillation: ModelOpt distillation plugin docs
- Quantization: https://docs.nvidia.com/DeepLearning/Megatron-Core/

---

*For parallelism details, see parent CLAUDE.md. For transformer architecture, see
[../core/transformer/CLAUDE.md](../core/transformer/CLAUDE.md).*
