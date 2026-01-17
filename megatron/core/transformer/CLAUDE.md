# CLAUDE.md - transformer

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Transformer building blocks: attention, MLP, layers, and TransformerConfig
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/transformer/` provides the core building blocks for transformer-based models:
attention mechanisms, feed-forward networks, layer configurations, and complete transformer
stacks. These modules are designed to work seamlessly with all parallelism strategies (TP,
PP, DP, CP, EP).

**Problem Solved**: Transformers require careful implementation of attention (multi-head,
grouped-query, multi-latent), layer normalization, residual connections, and MLP layers
across distributed systems. This directory provides production-grade implementations with
support for FP8 quantization, CUDA graphs, and advanced attention backends.

**Key Concepts**:
- **TransformerConfig**: Single source of truth for all model hyperparameters (1857 lines)
- **MegatronModule**: Base class ensuring checkpoint compatibility across parallelism
  configs
- **ModuleSpec**: Dataclass-based layer specification for easy customization
- **Attention Variants**: Multi-head (MHA), grouped-query (GQA), multi-latent (MLA)
- **CUDA Graphs**: Automatic graph capture for ~10% inference speedup
- **MoE Integration**: Token dispatchers, experts, routers for mixture-of-experts models

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `transformer_config.py` | 1857 | Model hyperparameter dataclass | `TransformerConfig`, `MLATransformerConfig` |
| `attention.py` | 1499 | Multi-head/GQA/MLA attention | `ParallelAttention`, `CoreAttention` |
| `cuda_graphs.py` | 1976 | CUDA graph capture & caching | `CudaGraphBuilder`, `is_graph_capturing()` |
| `transformer_layer.py` | 1116 | Attention + MLP + LayerNorm | `TransformerLayer`, `TransformerLayerSubmodules` |
| `transformer_block.py` | 847 | Stacked transformer layers | `TransformerBlock` |
| `module.py` | 467 | Base MegatronModule class | `MegatronModule`, `GraphableMegatronModule` |
| `mlp.py` | 352 | Gated/regular MLPs with fusions | `MLP` |
| `multi_token_prediction.py` | 1081 | Multi-token prediction head | `MultiTokenPredictionHead` |
| `multi_latent_attention.py` | 1046 | Multi-latent attention (DeepSeek-V3) | `MultiLatentAttention` |
| `spec_utils.py` | 126 | Module spec dataclass & builder | `ModuleSpec`, `build_module()` |
| `utils.py` | 467 | Layer utilities & helpers | `make_sharded_tensors_for_checkpoint()` |
| `dot_product_attention.py` | 271 | Fused dot-product attention | `DotProductAttention` |
| `enums.py` | 79 | Attention/layer enums | `AttnMaskType`, `LayerType`, `AttnBackend` |
| `moe/token_dispatcher.py` | 1520 | Token routing to experts | `TokenDispatcher`, `MoEDroplessDispatcher` |
| `moe/moe_utils.py` | 1281 | MoE-specific utilities | `check_load_balance()`, `ExpertDispatcher` |
| `moe/experts.py` | 985 | Expert FFN modules | `Experts`, `MoEExpertBlock` |
| `moe/router.py` | 593 | Top-k routing logic | `TopKRouter`, `NoisyTopKRouter` |
| `moe/moe_layer.py` | 388 | MoE wrapper layer | `MoELayer` |
| `moe/fused_a2a.py` | 518 | All-to-all for expert routing | `FusedAllToAllDispatcher` |
| `moe/shared_experts.py` | 294 | Shared expert pools | `SharedExpertsLayer` |
| `moe/upcycling_utils.py` | 359 | MoE checkpoint conversion | `upcycle_to_moe()` |
| `fsdp_dtensor_checkpoint.py` | 476 | Distributed tensor checkpointing | `get_dtensor_from_tensor()` |
| `pipeline_parallel_layer_layout.py` | 308 | PP stage layout specification | `PipelineParallelLayerLayout` |
| `heterogeneous/heterogeneous_config.py` | 267 | Heterogeneous TP config | `HeterogeneousTransformerConfig` |
| `custom_layers/batch_invariant_kernels.py` | 1006 | Custom CUDA kernels | Batch-invariant attention backends |

---

## Architecture

**Module Hierarchy**:

```
MegatronModule (base class, abstract interface)
├── TransformerLayer
│   ├── ParallelAttention (attention implementation)
│   └── MLP (feed-forward)
├── TransformerBlock (wrapper for stacked layers)
├── ParallelEmbedding
├── VocabParallelEmbedding
└── [Custom models: GPT, BERT, Llama, etc.]

TransformerConfig (dataclass)
└── Controls all hyperparameters for above modules
```

**Key Data Flows**:

1. **Training Forward Pass**:
   - Input tokens → Embedding → Loop over `TransformerLayer` →
   - Each layer: Attention(query, key, value) + Add & Norm + MLP + Add & Norm →
   - Output logits → Loss

2. **Attention Computation**:
   - Split heads: [batch, seq_len, hidden] → [batch, num_heads, seq_len, head_dim]
   - Scores: Q @ K^T / sqrt(d)
   - Apply mask, softmax, dropout
   - Attention @ V
   - Merge heads → output

3. **CUDA Graph Capture** (inference):
   - On first token: Capture full forward pass to a graph
   - On subsequent tokens: Replay cached graph (faster)

4. **MoE Routing**:
   - Router selects top-k experts per token
   - Dispatcher gathers tokens → routes to expert shards → scatters results

---

## Common Tasks

### 1. Create a Transformer Layer

```python
from megatron.core.transformer import TransformerLayer, TransformerConfig

config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    ffn_hidden_size=4096,
)
layer = TransformerLayer(config=config, layer_number=0)
output = layer(hidden_states)
```

### 2. Customize Attention Backend

```python
from megatron.core.transformer.enums import AttnBackend

config = TransformerConfig(
    attn_backend=AttnBackend.FLASH_ATTN,  # or TRITON, TE, etc.
)
# Automatically uses Flash Attention in ParallelAttention
```

### 3. Build Model from ModuleSpec

```python
from megatron.core.transformer import ModuleSpec, build_module

spec = ModuleSpec(
    module=TransformerLayer,
    params={'config': config, 'layer_number': 0},
)
layer = build_module(spec)
```

### 4. Enable CUDA Graphs for Inference

```python
from megatron.core.transformer.cuda_graphs import CudaGraphScope

config = TransformerConfig(cuda_graph_mode=CudaGraphScope.FULL)
# Now all fwd passes are captured to graph on first call
```

### 5. Add Multi-Token Prediction Head

```python
from megatron.core.transformer import MultiTokenPredictionHead

mtp_head = MultiTokenPredictionHead(
    config=config,
    vocab_size=50257,
    num_predict=4,  # predict 4 tokens ahead
)
```

---

## Dependencies

**Depends On** (imports from):
- `megatron.core.parallel_state` - Process group management
- `megatron.core.tensor_parallel` - TP layer utilities
- `megatron.core.dist_checkpointing` - Checkpoint mappings
- `torch` - PyTorch core
- `transformer_engine` (optional) - FP8, fusions
- `flash_attn` (optional) - Flash attention backend
- `einops` (optional) - Tensor reshaping

**Used By** (imported by):
- `megatron.core.models.*` - Model definitions (GPT, BERT, Llama)
- `megatron.core.inference.*` - Text generation engine
- `megatron.training.*` - Training scripts
- Custom user models

**Subdirectories**:
- `moe/` - Mixture of Experts (routers, experts, dispatchers)
- `heterogeneous/` - Heterogeneous tensor parallel support
- `custom_layers/` - Custom CUDA kernels (batch-invariant attention)

---

## Gotchas & Tips

1. **TransformerConfig is Massive**: 1857 lines with 50+ settings. Check docstrings for
   each field. Use IDE autocompletion.

2. **Attention Mask Shapes**: Mask should be [batch, 1, seq_len, seq_len] or broadcastable.
   Causal masks are created internally, don't double-apply.

3. **MegatronModule Checkpoints**: All state dicts go through
   `state_dict_for_save_checkpoint()`. Override this carefully—it's used for distributed
   saves.

4. **CUDA Graphs Limitations**: Cannot capture graphs with dynamic shapes. Disable for
   variable-length sequences or adaptive batching.

5. **MoE Load Balancing**: Unbalanced routing (e.g., all tokens to expert 0) kills
   throughput. Check `moe_utils.check_load_balance()` during training.

6. **Attention Backend Fallback**: If Flash Attn is unavailable, code falls back to
   standard PyTorch attention. Verify `HAVE_FA*` flags in attention.py.

7. **LayerNorm Variants**: TransformerConfig supports RMSNorm, LayerNorm, etc. via
   `norm_type` field. Ensure normalization is initialized correctly.

---

## See Also

- `megatron/core/CLAUDE.md` - Parallelism, optimizers, datasets
- `megatron/core/models/CLAUDE.md` - GPT, BERT, Llama model implementations
- `megatron/core/inference/CLAUDE.md` - Text generation engine
- `megatron/training/CLAUDE.md` - Training loop integration
- `MOE_TRAINING_GUIDE.md` - Mixture of Experts detailed guide
