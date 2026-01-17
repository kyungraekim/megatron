# CLAUDE.md - models

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Model architectures (GPT, BERT, T5, Mamba, vision, multimodal)
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/models/` contains production-ready model architectures built from
composable transformer blocks and layer specs. Each model family (GPT, BERT, T5, Mamba,
etc.) demonstrates how to use `TransformerConfig`, `ModuleSpec`, and `MegatronModule` to
build scalable LLMs from the core library.

**Problem Solved**: Building a 671B parameter model requires careful composition of:
parallelism-aware layers, distributed embeddings, activation hooks, checkpoint mapping.
This directory provides tested reference implementations and patterns for all major
architectures.

**Key Concepts**:
- **Layer Specs Pattern**: Each model defines `{model}_layer_specs.py` with `ModuleSpec`
  blueprints for layer composition (attention type, MLP, activations, norms)
- **Backend Abstraction**: `backends.py` provides swappable implementations (Local,
  Transformer Engine, Inference-optimized)
- **Embeddings**: `common/embeddings/` has RoPE, YaRN, relative position embeddings
  shared across all models
- **LanguageModule/VisionModule**: Base classes for text/vision models with unified
  forward signature

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `gpt/gpt_model.py` | 797 | GPT transformer with MoE, MTP, seq-parallel | `GPTModel` |
| `gpt/gpt_layer_specs.py` | 705 | Layer specs for GPT (attention, MLP types) | `get_gpt_layer_local_spec()` |
| `gpt/fine_grained_callables.py` | 612 | Custom forward/backward funcs for MoE | `get_fine_grained_func()` |
| `T5/t5_model.py` | 536 | T5 encoder-decoder architecture | `T5Model` |
| `T5/t5_spec.py` | 251 | Encoder/decoder layer specs for T5 | `get_t5_encoder_spec()` |
| `bert/bert_model.py` | 388 | BERT masked language model | `BertModel` |
| `bert/bert_layer_specs.py` | 118 | BERT layer specs (no decoder) | `get_bert_layer_spec()` |
| `mamba/mamba_model.py` | 307 | State-space model (SSM) architecture | `MambaModel` |
| `mamba/mamba_layer_specs.py` | 148 | Mamba SSM layer specs | `get_mamba_layer_spec()` |
| `multimodal/llava_model.py` | 1029 | LLaVA vision-language model | `LLaVAModel` |
| `vision/clip_vit_model.py` | 261 | CLIP Vision Transformer | `CLIPVisionTransformer` |
| `vision/radio.py` | 380 | RADIO vision backbone | `RADIOModel` |
| `retro/model.py` | 107 | RETRO retrieval-augmented model | `RETROModel` |
| `retro/decoder_spec.py` | 195 | RETRO decoder layer specs | `get_retro_decoder_spec()` |
| `retro/encoder_spec.py` | 171 | RETRO encoder layer specs | `get_retro_encoder_spec()` |
| `backends.py` | 182 | Backend abstraction (Local, TE, Inference) | `LocalSpecProvider`, `InferenceSpecProvider` |
| `common/embeddings/rotary_pos_embedding.py` | 261 | RoPE positional embeddings | `RotaryEmbedding` |
| `common/embeddings/yarn_rotary_pos_embedding.py` | 251 | YaRN scaling for RoPE | `YarnRotaryEmbedding` |
| `common/embeddings/language_model_embedding.py` | - | Token + positional embeddings | `LanguageModelEmbedding` |
| `common/language_module/language_module.py` | - | Base class for text models | `LanguageModule` |
| `common/vision_module/vision_module.py` | - | Base class for vision models | `VisionModule` |
| `common/model_chunk_schedule_plan.py` | 557 | Pipeline scheduling for models | `ModelChunkSchedulePlan` |

---

## Architecture

**Model Composition Pattern**:

```
┌─────────────────────────────────────┐
│ TransformerConfig                   │ Num layers, hidden size, vocab, etc
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ BackendSpecProvider                 │ Backend choice (LocalSpecProvider, TE, etc)
│ └─ Returns layer templates (spec)   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ {Model}LayerSpecs.py                │ E.g., get_gpt_layer_spec()
│ └─ ModuleSpec(TransformerLayer)     │ Defines attention, MLP, activation
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ {Model}Model                        │ E.g., GPTModel, BertModel
│ └─ Embedding + TransformerBlock×N  │ Wraps shared embeddings + layers
└─────────────────────────────────────┘
```

**Key Data Flows**:

1. **Forward Pass**: Input tokens → Embedding → TransformerBlocks → Unembedding → Logits
   - Each block uses layer spec to instantiate Attention + MLP with correct backend
   - Backend (TE, Local) transparent to model code

2. **Backward Pass**: Loss → Gradient through layers → Grad accumulation in DP
   - Fine-grained MoE routing backwards for expert load balancing
   - TP/PP synchronization automatic via collective ops

3. **Distributed Checkpoint**: Model state → ShardedStateDict (schema-free) → Save
   - Can load with different TP/PP/DP configuration via mapping

---

## Common Tasks

### 1. Create a New Model Architecture

Start with `gpt_model.py` as template. Three files needed:

```python
# my_model/my_model.py
from megatron.core.models.common.language_module import LanguageModule
from megatron.core.transformer import TransformerBlock

class MyModel(LanguageModule):
    def __init__(self, config, transformer_layer_spec, ...):
        super().__init__(config)
        self.embedding = LanguageModelEmbedding(config)
        self.transformer = TransformerBlock(
            config, transformer_layer_spec, ...
        )

# my_model/my_model_layer_specs.py
def get_my_layer_spec(config, backend_provider):
    return ModuleSpec(
        module_class=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend_provider.layer_norm(),
            self_attention=SelfAttention(...),
            mlp=MLP(...)
        )
    )
```

### 2. Switch Between Backends (Local vs Transformer Engine)

```python
from megatron.core.models.backends import LocalSpecProvider, TESpecProvider

# Use TE optimizations
backend = TESpecProvider() if use_te else LocalSpecProvider()
spec = get_gpt_layer_spec(config, backend)

# Backend transparently swaps:
# - ColumnParallelLinear → TEColumnParallelLinear
# - LayerNorm → TENorm (FP8-aware)
# - Attention → TEDotProductAttention (flash attention)
```

### 3. Access Rotary Embeddings

```python
from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding

# RoPE with optional YaRN scaling
embedding = RotaryEmbedding(
    dim=config.hidden_size // config.num_attention_heads,
    seq_len_interpolation_factor=config.rope_scaling_factor,  # RoPE
)

# Or with YaRN
embedding = YarnRotaryEmbedding(
    dim=..., seq_len=context_length, attn_factor=attn_factor
)
```

### 4. Use Layer Specs for Customization

```python
# Change attention type in layer spec
spec = get_gpt_layer_spec(config, backend)
spec.submodules.self_attention = MultiLatentAttention  # Swap in different attention

# Now all layers use custom attention automatically
model = GPTModel(config, transformer_layer_spec=spec)
```

---

## Dependencies

**Depends On** (imports from):
- `megatron/core/transformer/` - TransformerConfig, ModuleSpec, TransformerLayer/Block
- `megatron/core/tensor_parallel/` - ColumnParallelLinear, RowParallelLinear
- `megatron/core/pipeline_parallel/` - Pipeline communication
- `megatron/core/dist_checkpointing/` - ShardedStateDict for saving/loading
- `megatron/core/extensions/transformer_engine.py` - TE-optimized kernels (optional)

**Used By** (imported by):
- `megatron/training/training.py` - Training loop creates models
- `megatron/training/pretrain_gpt.py` - Example scripts
- `megatron/rl/` - RLHF model wrappers
- `megatron/inference/` - Inference engine for decoding
- Custom user models extending this library

---

## Gotchas & Tips

1. **Layer Spec vs ModuleSpec**: Layer spec is a ModuleSpec with `submodules` defined.
   Use `backend_provider.{module_type}()` to get the right implementation.

2. **Embedding Dtype**: Language model embedding defaults to config.params_dtype. For
   mixed-precision training, explicitly cast after embedding layer.

3. **Rotary Embedding Dimension**: Must be `hidden_size / num_attention_heads`. If using
   GQA, base dimension on GQA heads, not num_attention_heads.

4. **Backend Thread Safety**: BackendSpecProvider methods are called during model init.
   Ensure TE/Apex imports happen at module level, not inside specs.

5. **Checkpoint Format**: ShardedStateDict is opaque to model code. Use `save()` / `load()`
   from `dist_checkpointing/`. Avoid manual state_dict reshaping across PP configs.

6. **Vision Models**: CLIPVisionTransformer, RADIOModel use VisionModule base (no language
   embedding). For multimodal, compose LanguageModule + VisionModule.

---

## See Also

- `../transformer/CLAUDE.md` - TransformerConfig, ModuleSpec, attention mechanisms
- `../tensor_parallel/CLAUDE.md` - TP layers used by models
- `../dist_checkpointing/CLAUDE.md` - Checkpoint save/load format
- `megatron/training/CLAUDE.md` - Training loop that creates these models
- `MOE_TRAINING_GUIDE.md` - MoE-specific config and debugging
