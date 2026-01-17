# CLAUDE.md - moe

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Mixture of Experts: routers, experts, token dispatching, and load balancing
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

`megatron/core/transformer/moe/` implements Mixture of Experts (MoE) layers for scaling
models like DeepSeek-V3 and Mixtral. Each token is routed to a subset of experts rather
than processing through a single feed-forward layer, enabling parameter scaling without
proportional compute increases.

**Problem Solved**: Traditional dense FFN layers scale poorly beyond 100B parameters.
MoE activates only a subset of experts per token, allowing models with 300B+ parameters
while maintaining similar per-token compute. This directory handles token routing decisions,
cross-GPU expert dispatching, load balancing, and grouped computation.

**Key Concepts**:
- **Routing**: TopKRouter selects top-k experts per token using gating network
- **Dispatching**: MoETokenDispatcher routes tokens to expert-holding GPUs (all-to-all or
  all-gather)
- **Expert Parallelism (EP)**: Experts distributed across EP_SIZE ranks, each rank owns
  num_experts / EP_SIZE local experts
- **Load Balancing**: Auxiliary loss (Switch Transformer, Global Load Balancing) prevents
  token clustering
- **Shared Experts**: Optional low-traffic expert bypass (not routed, always computed)

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `token_dispatcher.py` | 1520 | Token routing strategies (all-to-all, all-gather, flex) | `MoETokenDispatcher`, `MoEAlltoAllTokenDispatcher`, `MoEAllGatherTokenDispatcher` |
| `moe_utils.py` | 1281 | Routing utilities, load balancing loss, permutation | `switch_load_balancing_loss_func`, `topk_routing_with_score_function`, `sinkhorn` |
| `experts.py` | 985 | Expert FFN modules with groupedGEMM optimization | `Experts`, `DenseExpertMLP`, `SequentialExpertMLP` |
| `router.py` | 593 | Top-k routing gate with aux loss variants | `TopKRouter`, `Router` (abstract) |
| `fused_a2a.py` | 518 | Fused all-to-all for expert dispatching | `fused_dispatch`, `fused_combine`, `hybrid_ep_dispatch` |
| `moe_layer.py` | 388 | Main MoE layer orchestrating route→dispatch→compute→combine | `MoELayer`, `MoESubmodules` |
| `shared_experts.py` | 294 | Shared expert pool (always-on experts, not routed) | `SharedExpertMLP` |
| `upcycling_utils.py` | 359 | Checkpoint conversion for MoE models | `upcycle_to_moe` |
| `grouped_gemm_util.py` | 22 | Grouped GEMM constant mappings | `GROUPED_GEMM_*` constants |

---

## Architecture

**Data Flow**: Tokens → Route → Dispatch → Compute → Combine

```
Input [B*S, H]
    ↓
┌─ ROUTE (TopKRouter) ──────────────────────┐
│ Input → Gating Linear → Logits            │
│ Apply Z-Loss, input jitter (training)     │
│ Top-K selection → probs, routing_map      │
│ Apply aux loss (Switch, Global, etc.)     │
└───────────────────────────────────────────┘
    ↓ probs [B*S, K], routing_map [B*S, E]
┌─ DISPATCH (MoETokenDispatcher) ───────────┐
│ Preprocess: reshape, sort tokens by expert│
│ All-to-All / All-Gather: send to EP ranks │
│ Postprocess: extract tokens for each      │
│   local expert, track tokens_per_expert   │
└───────────────────────────────────────────┘
    ↓ tokens_per_expert, dispatched tokens
┌─ COMPUTE (Experts) ───────────────────────┐
│ For each local expert:                    │
│   Gather tokens assigned to that expert   │
│   GroupedGEMM: compute FC1 + act + FC2    │
│   Permute back to original token order    │
└───────────────────────────────────────────┘
    ↓ expert outputs
┌─ COMBINE (MoETokenDispatcher) ─────────────┐
│ All-to-All / All-Gather: scatter outputs  │
│ Weighted sum by routing probs              │
│ (Optional: add shared expert output)      │
└───────────────────────────────────────────┘
    ↓ output [B*S, H]
```

**Expert Parallelism (EP)**: If EP_SIZE=8, num_experts=64, each rank owns 8 experts.
Token routed to expert 0 goes to rank 0, expert 8 to rank 1, etc.

**Load Balancing**: Routing can cause token clustering (all tokens → expert 0). Three
options:
1. **Switch Loss** (`moe_router_load_balancing_type="aux_loss"`): Per-microbatch
2. **Seq Loss**: Per-sequence (reshapes batch as sequences)
3. **Global Loss**: Per-training-step across all DP/CP ranks (tracks exponential moving
   average)

---

## Common Tasks

### 1. Add MoE Layer to Model

```python
from megatron.core.transformer.moe import MoELayer, MoESubmodules
from megatron.core.transformer import TransformerConfig, Experts

config = TransformerConfig(
    num_moe_experts=64,
    moe_router_topk=2,  # each token → top-2 experts
    moe_expert_capacity_factor=1.0,  # tokens_per_expert = (tokens / experts) * factor
    moe_router_load_balancing_type="aux_loss",  # load balancing strategy
    moe_aux_loss_coeff=0.01,  # weight of aux loss
    moe_token_dispatcher_type="alltoall",  # or "allgather", "flex"
)

submodules = MoESubmodules(
    experts=Experts,  # expert module class
    shared_experts=SharedExpertMLP,  # optional, for always-on experts
)

moe_layer = MoELayer(
    config=config,
    submodules=submodules,
    layer_number=12,
    pg_collection=pg_collection,  # from parallel_state
)

output, _ = moe_layer(hidden_states)
```

### 2. Inspect Token Routing Distribution

```python
# In training loop, after forward pass:
from megatron.core.transformer.moe.moe_utils import check_load_balance

# Optional: log per-expert token counts
tokens_per_expert = routing_map.sum(dim=0)  # [num_experts]
check_load_balance(routing_map, layer_number=12)
```

### 3. Use Different Token Dispatcher

```python
config = TransformerConfig(
    moe_token_dispatcher_type="flex",  # auto-choose all-to-all or all-gather
    moe_alltoall_group_tp_overlap=True,  # overlap TP + EP all-to-all
)
# MoELayer automatically selects MoEFlexTokenDispatcher
```

### 4. Enable Shared Experts (Optional Low-Traffic Path)

```python
config = TransformerConfig(
    moe_shared_expert_intermediate_size=8192,  # gate size for shared experts
    moe_shared_expert_overlap=True,  # overlap shared expert with dispatch
)
# Shared experts always activated, not routed to specific ranks
```

### 5. Tune Load Balancing Hyperparameters

```python
config = TransformerConfig(
    moe_router_topk=2,  # increase for higher capacity
    moe_expert_capacity_factor=1.25,  # allow overflow, drop excess tokens
    moe_z_loss_coeff=0.001,  # penalize large logits for stability
    moe_input_jitter_eps=0.01,  # add noise to router input
    moe_router_load_balancing_type=[
        "aux_loss",
        "seq_aux_loss",
        "global_aux_loss",
    ],  # multiple loss types
    moe_aux_loss_coeff=[0.01, 0.01, 0.001],  # per-loss weight
)
```

---

## Dependencies

**Depends On** (imports from):
- `megatron.core.parallel_state` - EP group management
- `megatron.core.tensor_parallel` - All-to-all, all-gather collectives
- `megatron.core.transformer.module` - MegatronModule base class
- `megatron.core.extensions.transformer_engine` - Fused kernels (optional, TE)
- `torch`, `torch.distributed` - PyTorch core, collectives

**Used By** (imported by):
- `megatron.core.transformer.transformer_layer` - TransformerLayer optionally includes
  MoE
- `megatron.core.models.*` - MoE model definitions (DeepSeek-V3, Mixtral)
- `megatron.training.training` - Training loop handles MoE aux loss

---

## Gotchas & Tips

1. **Expert Parallelism Setup**: EP requires EP_SIZE to divide num_experts evenly. If
   num_experts=64 and EP_SIZE=8, each rank gets 64/8=8 local experts. Mismatch causes
   assertion failure.

2. **Aux Loss Scaling**: Aux loss is scaled by coefficient and merged into model output.
   For per-token loss scaling, Megatron automatically divides by num_tokens. Check
   `MoEAuxLossAutoScaler.apply()`.

3. **Token Dropping**: If `moe_expert_capacity_factor < topk`, excess tokens are dropped
   during training. This reduces compute but can hurt convergence. Monitor token drop
   rate via logging.

4. **All-to-All vs All-Gather**: All-to-all is faster but uses more communication.
   All-gather is slower but better for small models. Use "flex" to auto-choose.

5. **GroupedGEMM**: Experts compute via grouped batched GEMMs for efficiency. Requires
   careful memory alignment. If OOM, reduce `moe_expert_capacity_factor`.

6. **Load Imbalance Indicators**: High auxiliary loss, or uneven tokens_per_expert
   across ranks → imbalanced routing. Try increasing `moe_z_loss_coeff` or
   `moe_input_jitter_eps`.

7. **Checkpoint Conversion**: Use `upcycling_utils.upcycle_to_moe()` to convert
   dense-FFN checkpoint to MoE (maps FFN weights to all experts).

---

## See Also

- `../CLAUDE.md` - Attention, MLP, TransformerConfig details
- `megatron/core/CLAUDE.md` - Parallelism, parallel_state, EP groups
- `megatron/core/models/CLAUDE.md` - DeepSeek-V3, Mixtral model implementations
- `megatron/MOE_TRAINING_GUIDE.md` - Detailed MoE training guide
- Papers: Switch Transformers (arxiv.org/abs/2101.03961), Global Load Balancing Loss
  (arxiv.org/abs/2501.11873)
