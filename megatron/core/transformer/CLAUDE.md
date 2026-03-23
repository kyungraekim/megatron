# Transformer & MoE Guide

> Covers: transformer layers, attention, MLP, ModuleSpec, MoE routing/dispatch, model architectures
> Parent: [../CLAUDE.md](../CLAUDE.md)

## Transformer Gotchas

- `TransformerConfig.__post_init__()` **mutates fields**: auto-sets `ffn_hidden_size = 4 * hidden_size`, `kv_channels = hidden_size / num_attention_heads`, `num_query_groups = num_attention_heads` (MHA default, not GQA). Also forces `attention_softmax_in_fp32 = True` if `apply_query_key_layer_scaling` is set. Sets `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` **globally** if `disable_bf16_reduced_precision_matmul` is True — a process-wide side effect.
- **Attention mask shape**: must be `[batch, 1, seq_len, seq_len]` or broadcastable. Causal masks are created internally — don't double-apply.
- **Attention backend fallback**: if Flash Attn unavailable, falls back to standard PyTorch. Check `HAVE_FA*` flags in `attention.py`.
- **RoPE dimension**: must be `hidden_size / num_attention_heads`. For GQA, base on `num_query_groups`, not `num_attention_heads`. Wrong value = silent wrong results.
- **Backend thread safety**: TE/Apex imports must happen at module level, not inside spec functions.
- **DSA** (`experimental_attention_variant/dsa.py`): new in v0.16 — experimental attention variant with dedicated module specs.

## ModuleSpec Patterns

- `ModuleSpec.module` can be: a **class** (direct instantiation), a `(module_path, ClassName)` **tuple** (dynamic import at build time), or a bare **function** (returned as-is, not called).
- The `submodules` field is automatically injected as `submodules=` kwarg into the module's `__init__`. If you forget `submodules` in the spec, the module silently receives `None`, causing a crash deep inside the layer (not at construction time).
- Layer specs: each model defines `get_{model}_layer_spec()` returning `ModuleSpec` with `TransformerLayerSubmodules`.
- Backend providers (`LocalSpecProvider`, `TESpecProvider`) swap implementations transparently: `ColumnParallelLinear` → `TEColumnParallelLinear`, `LayerNorm` → `TENorm`.

## MoE Gotchas

- **EP_SIZE must evenly divide num_experts** — mismatch causes assertion failure at init.
- Expert params need `param.allreduce = False` set during construction, or they silently get all-reduced across the full DP group instead of the expert-data-parallel group, producing **wrong gradients**.
- `MoEAuxLossAutoScaler` injects aux loss gradient via a **global class variable** (`main_loss_backward_scale`), not the computation graph. The scale is set by the schedule before `loss.backward()`. If backward runs before the scale is set, it uses the stale value from the previous step. Same pattern used for `MTPLossAutoScaler`.
- **Token dropping**: when `moe_expert_capacity_factor < topk`, excess tokens are dropped. Monitor drop rate — it silently hurts convergence.
- **Dispatchers**: All-to-All (less memory, more communication), All-Gather (more memory, less communication), Flex (auto-chooses). `GroupedGEMM` requires careful memory alignment; reduce `capacity_factor` if OOM.
- **Load balancing options**: `aux_loss` (per-microbatch), `seq_aux_loss` (per-sequence), `global_aux_loss` (per-step EMA). See `MOE_TRAINING_GUIDE.md`.
- **Router replay** (`moe/router_replay.py`): new in v0.16 — replay routing decisions for MoE models.
- `upcycling_utils.upcycle_to_moe()` converts dense FFN checkpoint to MoE by mapping weights to all experts.

## MoE Data Flow (stable pattern)

```
Route:    TopKRouter → gating linear → top-k selection → aux loss
Dispatch: MoETokenDispatcher → sort by expert → all-to-all/all-gather to EP ranks
Compute:  Experts → GroupedGEMM per local expert
Combine:  Weighted sum by routing probs + optional shared expert output
```

## Models

- **Composition pattern**: `TransformerConfig` → `get_{model}_layer_spec()` → `ModuleSpec` with submodules → Model class
- Old single-GPU checkpoints won't load with parallelism — need `dist_checkpointing` format.
- **DeepSeek V3.2**: new model architecture support in v0.16.
- **Gated Delta Net** (`ssm/gated_delta_net.py`): new SSM variant in v0.16.
- **MTP in standalone PP stages**: new in v0.16 — Multi-Token Prediction layers can be placed in standalone pipeline stages.
