# Mixture of Experts (MoE) Training Guide for Megatron-LM

> **Last Updated**: 2025-11-15
> **Target Audience**: AI Engineers, Researchers implementing autoregressive transformer models with MoE
> **Recommended Reading**: Read CLAUDE.md first for general Megatron-LM concepts

This comprehensive guide covers implementing and training autoregressive transformer models with Mixture of Experts (MoE) in Megatron-LM, with a focus on GPU efficiency and production-ready configurations.

---

## Table of Contents

1. [MoE Architecture Overview](#moe-architecture-overview)
2. [Implementation Deep Dive](#implementation-deep-dive)
3. [Training Configuration](#training-configuration)
4. [GPU Efficiency Optimizations](#gpu-efficiency-optimizations)
5. [Parallelism Strategy for MoE](#parallelism-strategy-for-moe)
6. [Example Configurations](#example-configurations)
7. [Performance Tuning](#performance-tuning)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Advanced Topics](#advanced-topics)

---

## MoE Architecture Overview

### What is Mixture of Experts?

Mixture of Experts (MoE) is a neural network architecture that uses **sparse activation** to scale model capacity while keeping computational cost manageable. Instead of processing every token through the same dense FFN layers, MoE routes each token to a subset of specialized "expert" networks.

**Key Concepts:**

- **Experts**: Individual FFN networks (typically 4-128+ experts per layer)
- **Router/Gating Network**: Determines which experts process each token
- **Top-K Routing**: Each token is routed to K experts (typically K=2)
- **Sparse Activation**: Only K out of N experts are activated per token
- **Load Balancing**: Ensures tokens are distributed evenly across experts

### MoE Benefits

1. **Scalable Capacity**: Train models with 10x-100x more parameters than dense models
2. **Constant Compute**: Computational cost stays roughly 2x-3x a dense model
3. **Improved Quality**: Better performance on diverse tasks due to specialization
4. **Memory Efficiency**: With expert parallelism, memory is distributed

### Architecture Variants in Megatron-LM

Megatron-LM supports several MoE architectures:

| Model | Experts | Top-K | Router Type | Special Features |
|-------|---------|-------|-------------|------------------|
| **Mixtral 8x7B** | 8 | 2 | Softmax + Aux Loss | Standard MoE |
| **DeepSeek-V2** | 160 | 6 | Group-limited | Multi-head latent attention (MLA) + Shared experts |
| **DeepSeek-V3** | 256 | 8 | Group-limited + Aux-loss-free | Shared experts + Dynamic expert bias |
| **Qwen-MoE** | 64 | 4 | Softmax + Aux Loss | Shared experts |

---

## Implementation Deep Dive

### Core Components

The MoE implementation in Megatron-LM consists of several key modules:

#### 1. MoE Layer (`megatron/core/transformer/moe/moe_layer.py`)

The main MoE layer that replaces standard FFN in transformer layers.

**Key Architecture:**

```python
class MoELayer(BaseMoELayer):
    """
    Components:
    - Router: Computes expert assignments and routing probabilities
    - Token Dispatcher: Handles token routing to experts (AlltoAll/AllGather/Flex)
    - Experts: The expert networks (GroupedMLP or SequentialMLP)
    - Shared Experts (optional): Dense experts that process all tokens
    """

    def forward(self, hidden_states):
        # 1. Router: Compute routing decisions
        scores, indices = self.router(hidden_states)

        # 2. Dispatch: Send tokens to experts
        (expert_input, expert_tokens_per_rank,
         expert_weight) = self.token_dispatcher.dispatch(
            hidden_states, routing_map, probs
        )

        # 3. Expert computation (parallel across experts)
        expert_output = self.experts(expert_input, expert_tokens_per_rank)

        # 4. Combine: Gather outputs from experts
        output = self.token_dispatcher.combine(
            expert_output, expert_weight
        )

        # 5. Shared experts (optional)
        if self.use_shared_expert:
            shared_output = self.shared_experts(hidden_states)
            output = output + shared_output

        return output
```

#### 2. Router (`megatron/core/transformer/moe/router.py`)

The router determines which experts process each token.

**Router Types:**

**A. TopKRouter (Default)**
```python
class TopKRouter(Router):
    """
    Standard top-k routing:
    1. Compute routing logits: logits = input @ W_gate
    2. Apply routing score function (softmax or sigmoid)
    3. Select top-k experts per token
    4. Apply load balancing loss (optional)
    """

    def forward(self, hidden_states):
        # Routing computation
        logits = self.gating(hidden_states)  # [S*B, num_experts]

        # Score function (softmax by default)
        scores = self.score_function(logits)

        # Top-k selection
        top_logits, top_indices = torch.topk(scores, k=self.topk, dim=1)

        # Compute load balancing loss
        if self.moe_aux_loss_func:
            aux_loss = self.moe_aux_loss_func(logits, top_indices)

        return scores, top_indices, aux_loss
```

**B. Group-Limited Router** (DeepSeek-V2/V3)
```python
"""
Group-limited routing for reducing communication:
1. Divide experts into groups
2. Select top groups based on aggregated scores
3. Select top-k experts within selected groups

Benefits:
- Reduces cross-node communication
- Better locality for multi-node training
"""
```

**Router Configuration Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--moe-router-topk` | 2 | Number of experts per token |
| `--moe-router-load-balancing-type` | `aux_loss` | Load balancing strategy |
| `--moe-aux-loss-coeff` | 0.01 | Auxiliary loss coefficient |
| `--moe-router-score-function` | `softmax` | Softmax or sigmoid |
| `--moe-router-num-groups` | None | Number of groups for group-limited routing |
| `--moe-router-group-topk` | None | Number of groups to select |
| `--moe-router-dtype` | None | Router computation dtype (fp32/fp64 for stability) |
| `--moe-router-fusion` | False | Enable router fusion (TE 2.7+) |

**Load Balancing Strategies:**

1. **Auxiliary Loss (aux_loss)**: GShard/Switch Transformer style
   - Encourages balanced expert utilization
   - Loss = α × Σ(fraction_tokens × fraction_capacity)

2. **Sequence Auxiliary Loss (seq_aux_loss)**: DeepSeek-V2 style
   - Computes loss per sequence instead of per batch
   - Better for variable-length sequences

3. **Global Auxiliary Loss (global_aux_loss)**: Cross-device balance
   - Balances experts globally across all devices

4. **Aux-Loss-Free (via expert bias)**: DeepSeek-V3 style
   - Uses dynamic expert bias instead of auxiliary loss
   - `--moe-router-enable-expert-bias`
   - `--moe-router-bias-update-rate` (default: 1e-3)

5. **Sinkhorn**: S-BASE algorithm
   - Iterative balancing algorithm

#### 3. Token Dispatcher (`megatron/core/transformer/moe/token_dispatcher.py`)

Handles communication between tokens and experts.

**Dispatcher Types:**

**A. AlltoAll Dispatcher (Recommended for most cases)**
```python
class MoEAlltoAllTokenDispatcher:
    """
    Uses AlltoAll collective for expert parallelism

    Dispatch:
    - Permute tokens by expert assignment
    - AlltoAll scatter to expert-parallel ranks

    Combine:
    - AlltoAll gather from expert-parallel ranks
    - Unpermute tokens to original order

    Best for: Balanced expert assignments
    """
```

**B. AllGather Dispatcher**
```python
class MoEAllGatherTokenDispatcher:
    """
    Uses AllGather collective

    Dispatch:
    - AllGather tokens to all expert-parallel ranks
    - Each rank processes its local experts

    Combine:
    - Local unpermute
    - ReduceScatter to original ranks

    Best for: Imbalanced expert assignments (variable capacity)
    """
```

**C. Flex Dispatcher** (Experimental)
```python
class MoEFlexTokenDispatcher:
    """
    Flexible dispatcher that adapts communication pattern

    Best for: Specific optimization scenarios
    """
```

**Communication Optimization:**

- **Fused All2All**: Combines permutation and communication
  - Enabled via `--moe-token-dispatcher-type alltoall` (default)
  - See `megatron/core/transformer/moe/fused_a2a.py`

- **Expert Parallel A2A Overlapping**:
  - Overlap expert computation with All2All communication
  - Significant speedup for large-scale MoE
  - Automatically enabled when conditions are met

#### 4. Experts (`megatron/core/transformer/moe/experts.py`)

The actual expert networks.

**Expert Implementations:**

**A. GroupedMLP (Recommended)**
```python
class GroupedMLP(MegatronModule):
    """
    Efficient grouped matrix multiplication for experts

    Uses nv-grouped-gemm library for optimized computation:
    - Single CUDA kernel for all experts
    - Better GPU utilization
    - Lower memory overhead

    Requirements:
    - pip install nv-grouped-gemm~=1.1
    - --moe-grouped-gemm flag
    - --disable-bias-linear (bias not supported yet)

    Performance: ~2-3x faster than SequentialMLP
    """
```

**B. SequentialMLP (Fallback)**
```python
class SequentialMLP(MegatronModule):
    """
    Sequential execution of experts

    Falls back when:
    - grouped-gemm not available
    - Bias is required
    - Specific activation functions

    Slower but more flexible
    """
```

**Expert Configuration:**

| Argument | Description |
|----------|-------------|
| `--num-experts` | Total number of experts (e.g., 8, 64, 256) |
| `--moe-grouped-gemm` | Use grouped GEMM for experts (recommended) |
| `--moe-use-legacy-grouped-gemm` | Use legacy grouped GEMM implementation |
| `--ffn-hidden-size` | Hidden size for each expert's FFN |
| `--disable-bias-linear` | Required for grouped GEMM |

#### 5. Shared Experts (Optional)

Some architectures (DeepSeek, Qwen) include shared experts that process all tokens.

```python
class SharedExpertMLP(MegatronModule):
    """
    Dense FFN layer that processes all tokens

    Benefits:
    - Provides baseline knowledge
    - Reduces expert load imbalance impact
    - Can overlap with routed expert computation

    Configuration:
    - --moe-shared-expert-intermediate-size (hidden size)
    - --moe-shared-expert-overlap (computation overlap)
    """
```

**Shared Expert Overlap:**
- When `--moe-shared-expert-overlap` is enabled:
  - Shared expert computation runs in parallel stream
  - Overlaps with routed expert communication
  - Improves GPU utilization

---

## Training Configuration

### Minimal MoE Configuration

Starting from the Mixtral 8x7B example:

```bash
#!/bin/bash
# Minimal Mixtral-style MoE training

# Basic model arguments
MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 32768
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
)

# MoE-specific arguments
MOE_ARGS=(
    --num-experts 8                              # 8 experts per MoE layer
    --moe-router-topk 2                          # Route each token to 2 experts
    --moe-router-load-balancing-type aux_loss    # Use auxiliary loss for balance
    --moe-aux-loss-coeff 1e-2                    # Aux loss coefficient
    --moe-grouped-gemm                           # Use efficient grouped GEMM
    --moe-token-dispatcher-type alltoall         # AlltoAll communication
)

# Parallelism (CRITICAL for MoE)
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8              # Expert parallelism
    --sequence-parallel                          # Required with EP+TP
    --use-distributed-optimizer
)

# Training args
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 500000
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

torchrun --nproc_per_node=8 pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]}
```

### Advanced MoE Configuration (DeepSeek-V3 Style)

```bash
# Advanced MoE with all optimizations

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 64
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --num-query-groups 128
    --seq-length 4096
    --max-position-embeddings 163840
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    # Basic MoE
    --num-experts 256                           # 256 experts (DeepSeek-V3 scale)
    --moe-router-topk 8                         # Route to 8 experts
    --moe-grouped-gemm                          # Grouped GEMM
    --moe-token-dispatcher-type alltoall

    # Group-limited routing (reduce cross-node communication)
    --moe-router-num-groups 64                  # Divide into 64 groups
    --moe-router-group-topk 4                   # Select from 4 groups

    # Aux-loss-free load balancing
    --moe-router-enable-expert-bias             # Use dynamic expert bias
    --moe-router-bias-update-rate 1e-3          # Bias update rate
    --moe-router-load-balancing-type none       # No auxiliary loss

    # Router optimization
    --moe-router-dtype fp32                     # FP32 for stability
    --moe-router-fusion                         # Fused router (TE 2.7+)

    # Shared experts
    --moe-shared-expert-intermediate-size 18432 # Shared expert hidden size
    --moe-shared-expert-overlap                 # Overlap computation

    # Memory optimization
    --overlap-param-gather
    --overlap-grad-reduce
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 16
    --expert-model-parallel-size 64            # Heavy EP for 256 experts
    --context-parallel-size 2                  # For long context
    --sequence-parallel                        # Required
    --use-distributed-optimizer
)

# FP8 training (Hopper+ GPUs)
FP8_ARGS=(
    --fp8-hybrid                               # FP8 hybrid mode
    --fp8-amax-compute-algo max                # FP8 amax computation
    --fp8-amax-history-len 1024                # FP8 history
)

# Activation checkpointing for memory
CKPT_ARGS=(
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe shared_experts moe_act  # Recompute MoE modules
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4096                   # Large batch for stability
    --lr 3e-4
    --min-lr 3e-5
    --train-iters 1000000
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16                                     # Or use FP8
)

# Communication optimizations
COMM_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
    --use-distributed-optimizer
    --overlap-param-gather-with-optimizer-step  # Advanced overlap
)
```

### MoE Layer Frequency

Control which layers are MoE vs dense:

```bash
# Every layer is MoE
--moe-layer-freq 1

# Every other layer is MoE (1 MoE : 1 Dense)
--moe-layer-freq 2

# Custom pattern: [MoE, Dense, Dense, MoE, Dense, Dense, ...]
--moe-layer-freq "[1,0,0]"

# DeepSeek-V3 pattern: MoE every 3 layers starting from layer 1
--moe-layer-freq "[0,1,0,0]"
```

---

## GPU Efficiency Optimizations

### 1. Grouped GEMM (Most Important)

**Impact**: 2-3x speedup for expert computation

```bash
# Enable grouped GEMM
--moe-grouped-gemm
--disable-bias-linear  # Required for grouped GEMM

# Install dependency
pip install nv-grouped-gemm~=1.1
```

**How it works**:
- Single fused CUDA kernel for all expert FFN computations
- Better memory coalescing and cache utilization
- Reduced kernel launch overhead

**When NOT to use**:
- If you need bias in linear layers
- Legacy models that require sequential expert computation

### 2. Router Fusion

**Impact**: 10-20% speedup for routing computation

```bash
# Enable router fusion (requires Transformer Engine 2.7+)
--moe-router-fusion
```

**What it fuses**:
- Router gating computation
- Top-k selection
- Auxiliary loss computation
- All in a single optimized kernel

### 3. Communication Overlap

**Impact**: 20-40% speedup by hiding communication latency

**A. Expert Parallel A2A Overlapping**
```bash
# Automatically enabled when conditions are met:
# - AlltoAll token dispatcher
# - Multiple EP ranks
# - Sufficient experts per rank

# Manual control (if needed):
--overlap-grad-reduce
--overlap-param-gather
```

**B. Shared Expert Overlap**
```bash
# Run shared experts in parallel with routed experts
--moe-shared-expert-overlap
```

**How it works**:
```
Without overlap:
|--- Router ---|--- A2A Send ---|--- Expert Compute ---|--- A2A Recv ---|--- Combine ---|

With overlap:
|--- Router ---|--- A2A Send ---|
               |--- Expert Compute ---|
               |--- Shared Expert ---|  (parallel stream)
                                    |--- A2A Recv ---|--- Combine ---|
```

### 4. FP8 Training (Hopper/Ada/Blackwell GPUs)

**Impact**: 1.5-2x throughput with minimal accuracy loss

```bash
FP8_ARGS=(
    --fp8-hybrid                              # FP8 E4M3/E5M2 hybrid
    --fp8-amax-compute-algo max               # Amax computation
    --fp8-amax-history-len 1024               # Scaling factor history
    --fp8-margin 0                            # Scaling margin
)

# FP8 specific for MoE
--moe-router-dtype fp32                       # Keep router in FP32 for stability
```

**What runs in FP8**:
- Expert FFN forward/backward
- Attention (if enabled)
- Gradients

**What stays in FP32/BF16**:
- Router computation (for numerical stability)
- LayerNorm
- Master weights

### 5. Activation Checkpointing

**Impact**: 2-4x memory reduction, ~20% slower

```bash
# Selective recomputation for MoE
--recompute-activations
--recompute-granularity selective
--recompute-modules moe shared_experts moe_act layernorm

# Full recomputation (more memory savings, slower)
--recompute-activations
--recompute-granularity full
```

**What to recompute**:
- `moe`: Entire MoE layer (dispatch + experts + combine)
- `shared_experts`: Shared expert computation
- `moe_act`: Only MoE activation functions (faster, less memory savings)
- `layernorm`: LayerNorm operations

**Trade-off**:
- More recomputation = Less memory, slower training
- Selective recomputation of only expensive operations is often best

### 6. Optimizer Offloading (For very large models)

```bash
# CPU offloading for optimizer states
--use-distributed-optimizer
--cpu-offload-optimizer

# Or use ZeRO-style sharding
--use-distributed-optimizer
--data-parallel-sharding-strategy optim_grads_params
```

### 7. CUDA Graphs (Inference)

```bash
# For inference workloads
--use-cuda-graph
--cuda-graph-num-batches 4  # Number of batch sizes to capture
```

---

## Parallelism Strategy for MoE

### Understanding MoE Parallelism

MoE models introduce **Expert Parallelism (EP)** on top of standard parallelism:

```
Total GPUs = TP × PP × EP × DP × CP

Where:
- TP = Tensor Parallelism
- PP = Pipeline Parallelism
- EP = Expert Parallelism (unique to MoE)
- DP = Data Parallelism
- CP = Context Parallelism
```

### Expert Parallelism (EP)

**How it works**:
- Experts are sharded across EP ranks
- Each rank holds `num_experts / EP` local experts
- AlltoAll communication routes tokens to experts

**Example**: 8 experts, EP=4
- Rank 0: Experts 0-1
- Rank 1: Experts 2-3
- Rank 2: Experts 4-5
- Rank 3: Experts 6-7

**EP Configuration**:
```bash
--expert-model-parallel-size 8  # 8-way expert parallelism
```

**EP Best Practices**:
1. **EP should divide num_experts evenly**
   - ✅ 64 experts, EP=8 → 8 experts per rank
   - ❌ 64 experts, EP=7 → Imbalanced

2. **EP size considerations**:
   - Small models (8-16 experts): EP = num_experts (one expert per GPU)
   - Medium models (32-64 experts): EP = 8-16
   - Large models (128-256 experts): EP = 32-64

3. **EP affects communication**:
   - Larger EP → More AlltoAll communication
   - Prefer EP within a node when possible
   - Use group-limited routing for multi-node EP

### Expert Tensor Parallelism (Expert TP)

Some models use different TP size for experts vs attention:

```bash
--tensor-model-parallel-size 4        # Attention TP
--expert-tensor-parallel-size 2       # Expert TP (lower to reduce comm)
```

**When to use**:
- Very wide expert FFNs
- Want to reduce expert communication overhead
- Have enough GPUs for additional parallelism dimension

### Critical: Sequence Parallelism with EP+TP

**IMPORTANT**: When using EP + TP together, Sequence Parallelism is **REQUIRED**:

```bash
--expert-model-parallel-size 8
--tensor-model-parallel-size 2
--sequence-parallel              # REQUIRED!
```

**Why**: Without SP, experts would need to gather full sequence length, causing OOM.

### MoE Parallelism Examples

#### Example 1: Mixtral 8x7B (64 GPUs)

```bash
# Configuration
Total GPUs: 64
Experts: 8
TP: 1, PP: 4, EP: 8, DP: 2

# Layout
- 4 pipeline stages (PP=4)
- Within each stage: 8 expert parallel ranks, 2 data parallel replicas (16 GPUs per stage)
- No tensor parallelism (small enough to fit)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --sequence-parallel
)
```

#### Example 2: DeepSeek-V2 Lite (256 GPUs)

```bash
# Configuration
Total GPUs: 256
Experts: 160
TP: 2, PP: 8, EP: 16, DP: 1

# Layout
- 8 pipeline stages
- Within each stage: TP=2, EP=16
- 160 experts / 16 = 10 experts per EP rank

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 16
    --sequence-parallel
    --use-distributed-optimizer
)
```

#### Example 3: DeepSeek-V3 671B (1024 GPUs)

```bash
# Configuration
Total GPUs: 1024
Experts: 256
TP: 2, PP: 16, EP: 64, DP: varies

# Layout
- 16 pipeline stages (for 671B parameters)
- Heavy expert parallelism (EP=64)
- 256 experts / 64 = 4 experts per EP rank
- Group-limited routing to reduce cross-node communication

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 16
    --expert-model-parallel-size 64
    --context-parallel-size 2
    --sequence-parallel
    --use-distributed-optimizer
)

# Group-limited routing
MOE_ARGS=(
    --num-experts 256
    --moe-router-num-groups 64      # Equal to EP size
    --moe-router-group-topk 4        # Select from 4 groups only
)
```

### Parallelism Selection Guidelines for MoE

**Step 1: Determine EP**
- Start with EP = num_experts for small models
- For large models, aim for 2-8 experts per EP rank
- Keep EP as a power of 2 if possible

**Step 2: Check Memory**
- If experts fit in GPU memory with EP alone → TP=1
- If not → Add TP (2 or 4 typically)

**Step 3: Add PP**
- For models > 100B parameters, use PP
- Divide layers evenly across PP stages

**Step 4: Fill remaining GPUs with DP**
- DP = Total_GPUs / (TP × PP × EP × CP)

**Step 5: Verify**
- Check: TP × PP × EP × DP × CP = Total GPUs
- Check: num_experts % EP == 0
- Check: num_layers % PP == 0
- Check: num_attention_heads % TP == 0

---

## Example Configurations

### 1. Mixtral 8x7B (Standard MoE)

**Model Details**:
- 8 experts per MoE layer
- Top-2 routing
- 46.7B total parameters, 12.9B active per token

**8 GPU Configuration**:
```bash
#!/bin/bash
# Mixtral 8x7B on 8 GPUs

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --num-query-groups 8
    --seq-length 4096
    --max-position-embeddings 32768
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 1000000
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
)

MOE_ARGS=(
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8    # One expert per GPU
    --sequence-parallel
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --lr 1e-4
    --train-iters 500000
    --lr-decay-style cosine
    --min-lr 1e-5
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

OPTIMIZATION_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path ${DATA_PATH}
    --split 99990,8,2
)

torchrun --nproc_per_node=8 pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]} \
    ${DATA_ARGS[@]}
```

### 2. DeepSeek-V2 Style (64 Experts, Shared Experts)

**Model Details**:
- 64 routed experts
- 2 shared experts
- Top-6 routing
- Group-limited routing

**64 GPU Configuration**:
```bash
#!/bin/bash
# DeepSeek-V2 style on 64 GPUs

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 40
    --hidden-size 5120
    --ffn-hidden-size 12288
    --num-attention-heads 128
    --num-query-groups 16
    --seq-length 4096
    --max-position-embeddings 163840
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    # Routed experts
    --num-experts 64
    --moe-router-topk 6
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall

    # Group-limited routing
    --moe-router-num-groups 16
    --moe-router-group-topk 2

    # Shared experts
    --moe-shared-expert-intermediate-size 12288
    --moe-shared-expert-overlap

    # Load balancing
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-2

    # Stability
    --moe-router-dtype fp32
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --sequence-parallel
    --use-distributed-optimizer
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --lr 2e-4
    --min-lr 2e-5
    --train-iters 500000
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

OPTIMIZATION_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe shared_experts
)
```

### 3. Large-Scale MoE (256 Experts, 256 GPUs)

**For research/production large-scale training**:

```bash
#!/bin/bash
# Large-scale MoE on 256 GPUs

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 64
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --num-query-groups 128
    --seq-length 4096
    --max-position-embeddings 163840
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 10000
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    # Routed experts
    --num-experts 256
    --moe-router-topk 8
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall

    # Group-limited routing (reduce cross-node comm)
    --moe-router-num-groups 64
    --moe-router-group-topk 4

    # Shared experts
    --moe-shared-expert-intermediate-size 18432
    --moe-shared-expert-overlap

    # Aux-loss-free balancing
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-load-balancing-type none

    # Router optimization
    --moe-router-dtype fp32
    --moe-router-fusion
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 16
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
)

FP8_ARGS=(
    --fp8-hybrid
    --fp8-amax-compute-algo max
    --fp8-amax-history-len 1024
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4096
    --lr 3e-4
    --min-lr 3e-5
    --train-iters 1000000
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

OPTIMIZATION_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-param-gather-with-optimizer-step
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe shared_experts moe_act
)

# For multi-node
DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 32
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]}
```

---

## Performance Tuning

### Profiling MoE Models

**1. Enable profiling**:
```bash
--profile
--profile-step-start 10
--profile-step-end 20
--tensorboard-dir ./tensorboard
```

**2. Use NVIDIA tools**:
```bash
# Nsight Systems
nsys profile -o profile.qdrep \
    python pretrain_gpt.py ...

# NVIDIA DCGM for multi-GPU monitoring
dcgmi profile --pause
```

**3. Key metrics to monitor**:
- **Expert utilization**: Check if all experts are being used evenly
- **Communication time**: AlltoAll should be < 20% of total time
- **GPU utilization**: Should be > 80%
- **Memory usage**: Leave 10-20% headroom for spikes

### Common Performance Bottlenecks

**1. Expert Load Imbalance**

**Symptoms**:
- Some GPUs idle while others compute
- Low overall GPU utilization
- High variance in expert assignment

**Solutions**:
```bash
# Use auxiliary loss
--moe-router-load-balancing-type aux_loss
--moe-aux-loss-coeff 1e-2

# Or use expert bias (aux-loss-free)
--moe-router-enable-expert-bias
--moe-router-bias-update-rate 1e-3

# Increase top-k (more uniform distribution)
--moe-router-topk 4  # Instead of 2
```

**2. Communication Overhead**

**Symptoms**:
- High AlltoAll time in profiling
- Communication > 30% of iteration time

**Solutions**:
```bash
# Use group-limited routing
--moe-router-num-groups 16
--moe-router-group-topk 2

# Reduce EP size (keep experts local)
--expert-model-parallel-size 8  # Instead of 16

# Enable communication overlap
--overlap-grad-reduce
--overlap-param-gather
```

**3. Memory Issues**

**Symptoms**:
- OOM errors
- Frequent CUDA malloc/free

**Solutions**:
```bash
# Enable activation checkpointing
--recompute-activations
--recompute-granularity selective
--recompute-modules moe shared_experts

# Use distributed optimizer
--use-distributed-optimizer

# Reduce micro-batch size
--micro-batch-size 1

# Use gradient accumulation
--global-batch-size 256
--micro-batch-size 1
# Implies 256 gradient accumulation steps
```

**4. Router Instability**

**Symptoms**:
- NaN losses
- Experts collapse (all tokens to one expert)
- High gradient norms

**Solutions**:
```bash
# Use FP32 for router
--moe-router-dtype fp32

# Reduce auxiliary loss coefficient
--moe-aux-loss-coeff 1e-3  # Instead of 1e-2

# Use gradient clipping
--clip-grad 1.0

# Warmup router learning rate
--router-lr-warmup-iters 1000
```

### Optimal Hyperparameters

**Learning Rates**:
```bash
# Dense model baseline: 1e-4
# MoE models: 1.5-3x higher

Small MoE (< 50B): --lr 1.5e-4
Medium MoE (50-200B): --lr 2e-4
Large MoE (> 200B): --lr 3e-4

# Longer warmup for MoE
--lr-warmup-iters 2000  # Dense: 500-1000
```

**Batch Sizes**:
```bash
# MoE benefits from larger batches
# Helps load balancing and stability

Small MoE: --global-batch-size 256-512
Medium MoE: --global-batch-size 1024-2048
Large MoE: --global-batch-size 2048-4096
```

**Auxiliary Loss Coefficients**:
```bash
# Standard (Mixtral/Switch): 1e-2
--moe-aux-loss-coeff 1e-2

# Sensitive models: 1e-3
--moe-aux-loss-coeff 1e-3

# With many experts (>128): 5e-3
--moe-aux-loss-coeff 5e-3
```

---

## Common Issues & Solutions

### Issue 1: OOM with MoE

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**Solutions**:

**A. Reduce memory footprint**:
```bash
# 1. Activation checkpointing
--recompute-activations
--recompute-granularity selective
--recompute-modules moe shared_experts moe_act

# 2. Reduce batch size
--micro-batch-size 1

# 3. Use distributed optimizer
--use-distributed-optimizer
--overlap-param-gather
```

**B. Increase parallelism**:
```bash
# Increase EP (distribute experts)
--expert-model-parallel-size 16  # Was 8

# Add TP (shard expert FFNs)
--tensor-model-parallel-size 2
--expert-tensor-parallel-size 2
```

**C. Enable CPU offloading** (last resort):
```bash
--cpu-offload-optimizer
--cpu-offload-optimizer-ratio 0.5
```

### Issue 2: Expert Collapse

**Symptoms**:
- All tokens routed to 1-2 experts
- Other experts receive no tokens
- Loss stops decreasing

**Root causes**:
- Auxiliary loss too weak
- Learning rate too high
- Router initialized poorly

**Solutions**:

**A. Strengthen load balancing**:
```bash
# Increase aux loss
--moe-aux-loss-coeff 5e-2  # Was 1e-2

# Or use expert bias
--moe-router-enable-expert-bias
--moe-router-bias-update-rate 5e-3  # Higher = stronger
```

**B. Adjust learning rate**:
```bash
# Reduce overall LR
--lr 1e-4  # Was 3e-4

# Or add router-specific LR (if supported)
--router-lr-scale 0.1  # 10x lower than main LR
```

**C. Better initialization**:
```bash
# Use smaller init std for router
--init-method-std 0.006  # Smaller = less extreme initial routing
```

### Issue 3: Slow AlltoAll Communication

**Symptoms**:
- Communication takes > 30% of iteration time
- Profiling shows high AlltoAll time
- Poor scaling with more GPUs

**Solutions**:

**A. Group-limited routing** (reduce communication):
```bash
--moe-router-num-groups 16      # Limit to groups
--moe-router-group-topk 2       # Select from subset
```

**B. Optimize EP layout**:
```bash
# Keep EP within node (8 GPUs per node)
--expert-model-parallel-size 8  # Not 16 or 32

# Or use hierarchical CP
--hierarchical-context-parallel-sizes 2 4
```

**C. Use faster interconnect**:
- Ensure GPUs are on high-speed interconnect (NVLink, InfiniBand)
- Check `NCCL_DEBUG=INFO` for communication topology

**D. Enable communication overlap**:
```bash
--overlap-grad-reduce
--overlap-param-gather
--moe-shared-expert-overlap  # If using shared experts
```

### Issue 4: NaN Loss

**Symptoms**:
- Loss becomes NaN during training
- Usually happens early in training

**Solutions**:

**A. Numerical stability**:
```bash
# FP32 router
--moe-router-dtype fp32

# FP32 LayerNorm
--apply-layernorm-1p  # 1+x instead of x in LayerNorm

# Lower precision for overflow
--initial-loss-scale 4096  # Start with lower loss scaling
```

**B. Reduce learning rate**:
```bash
--lr 1e-4              # Lower LR
--lr-warmup-iters 5000 # Longer warmup
--clip-grad 1.0        # Gradient clipping
```

**C. Check initialization**:
```bash
--init-method-std 0.01  # Smaller initialization
```

### Issue 5: Poor Convergence

**Symptoms**:
- Loss plateaus early
- Val loss doesn't improve
- MoE performs worse than dense baseline

**Solutions**:

**A. Hyperparameter adjustment**:
```bash
# Larger batch size
--global-batch-size 2048  # Was 1024

# Higher learning rate
--lr 3e-4  # MoE often needs higher LR

# Longer warmup
--lr-warmup-iters 2000

# Lower weight decay
--weight-decay 0.05  # Was 0.1
```

**B. Load balancing**:
```bash
# Ensure experts are being used
--moe-aux-loss-coeff 1e-2

# Monitor expert utilization in logs
```

**C. Routing strategy**:
```bash
# Try different top-k
--moe-router-topk 4  # Instead of 2

# Try different score function
--moe-router-score-function sigmoid  # Instead of softmax
```

### Issue 6: Checkpoint Loading Failures

**Symptoms**:
- Can't load checkpoint after changing parallelism
- Shape mismatch errors

**Solutions**:

**A. Use distributed checkpointing**:
```bash
# Save with dist checkpoint
--use-dist-ckpt
--auto-detect-ckpt-format

# Allows loading with different parallelism
```

**B. Checkpoint conversion**:
```python
# Convert checkpoint for different parallelism
python tools/checkpoint/convert_checkpoint.py \
    --model-type GPT \
    --loader mcore \
    --saver mcore \
    --target-tensor-parallel-size 4 \
    --target-expert-parallel-size 16 \
    --load-dir /old/checkpoint \
    --save-dir /new/checkpoint
```

---

## Advanced Topics

### 1. Custom Router Implementation

For research, you might want to implement custom routing logic:

```python
# megatron/core/transformer/moe/custom_router.py

from megatron.core.transformer.moe.router import Router

class MyCustomRouter(Router):
    """Custom router with specialized routing logic."""

    def __init__(self, config, pg_collection=None):
        super().__init__(config, pg_collection)
        # Add custom initialization

    def forward(self, hidden_states):
        """
        Custom routing logic.

        Returns:
            scores: Routing probabilities [num_tokens, num_experts]
            indices: Expert indices [num_tokens, top_k]
            aux_loss: Auxiliary loss for load balancing
        """
        # 1. Compute routing logits
        logits = self.gating(hidden_states)

        # 2. Custom routing logic here
        # Example: entropy-based routing
        entropy = -torch.sum(F.softmax(logits, dim=-1) *
                           F.log_softmax(logits, dim=-1), dim=-1)

        # Adjust logits based on entropy
        adjusted_logits = logits + self.entropy_weight * entropy.unsqueeze(-1)

        # 3. Top-k selection
        scores = F.softmax(adjusted_logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, k=self.config.moe_router_topk)

        # 4. Compute load balancing loss
        aux_loss = self.compute_load_balancing_loss(logits, top_indices)

        return scores, top_indices, aux_loss
```

**Register custom router**:
```python
# In your training script
from megatron.core.transformer.moe.moe_layer import MoELayer
from custom_router import MyCustomRouter

# Monkey-patch or use custom spec
moe_layer.router = MyCustomRouter(config, pg_collection)
```

### 2. Adaptive Top-K Routing

Dynamically adjust K based on token difficulty:

```python
class AdaptiveTopKRouter(Router):
    """Router with adaptive top-k selection."""

    def forward(self, hidden_states):
        logits = self.gating(hidden_states)
        scores = F.softmax(logits, dim=-1)

        # Compute routing confidence
        max_scores, _ = torch.max(scores, dim=-1)

        # Easy tokens (high confidence): K=1
        # Hard tokens (low confidence): K=4
        k_per_token = torch.where(
            max_scores > 0.5,
            torch.ones_like(max_scores, dtype=torch.long),
            torch.ones_like(max_scores, dtype=torch.long) * 4
        )

        # Implement adaptive top-k selection
        # (requires custom token dispatcher)
        ...
```

### 3. Expert Specialization Analysis

Analyze what each expert learns:

```python
# Analysis script
import torch
from collections import defaultdict

def analyze_expert_specialization(model, dataloader):
    """
    Analyze which types of tokens each expert processes.
    """
    expert_token_stats = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Forward pass
            output = model(batch['input_ids'])

            # Extract routing decisions from each MoE layer
            for layer_idx, layer in enumerate(model.layers):
                if hasattr(layer, 'mlp') and isinstance(layer.mlp, MoELayer):
                    routing_indices = layer.mlp.router.last_routing_indices

                    # Group tokens by expert
                    for expert_idx in range(num_experts):
                        mask = (routing_indices == expert_idx).any(dim=-1)
                        expert_tokens = batch['input_ids'][mask]
                        expert_token_stats[f'layer_{layer_idx}_expert_{expert_idx}'].extend(
                            expert_tokens.cpu().tolist()
                        )

    # Analyze token distributions
    for key, tokens in expert_token_stats.items():
        vocab_dist = Counter(tokens)
        print(f"{key}: Top tokens = {vocab_dist.most_common(10)}")
```

### 4. Mixture of Depths (MoD)

Combine MoE with dynamic depth (skip layers adaptively):

```python
class MoDLayer(nn.Module):
    """
    Mixture of Depths: Tokens can skip layers dynamically.
    Combines with MoE for compound sparsity.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.moe_layer = MoELayer(config)

        # Skip router
        self.skip_router = nn.Linear(config.hidden_size, 2)  # skip or process

    def forward(self, hidden_states):
        # Decide which tokens skip this layer
        skip_logits = self.skip_router(hidden_states)
        skip_probs = F.softmax(skip_logits, dim=-1)
        skip_decisions = torch.argmax(skip_probs, dim=-1)  # 0=skip, 1=process

        # Process only non-skipped tokens through MoE
        process_mask = (skip_decisions == 1)
        if process_mask.any():
            processed = self.moe_layer(hidden_states[process_mask])
            hidden_states[process_mask] = processed

        return hidden_states
```

### 5. Fine-Grained Expert Parallelism

For very large expert counts, use fine-grained EP:

```bash
# Example: 1024 experts on 256 GPUs
--num-experts 1024
--expert-model-parallel-size 256

# Each GPU holds 4 experts
# But communication is huge!

# Solution: Hierarchical EP
--expert-model-parallel-size 64        # Coarse EP
--expert-tensor-parallel-size 4        # Fine-grained TP within expert

# Now: 64 * 4 = 256 GPUs
# Each expert is sharded across 4 GPUs (TP=4)
# 16 experts per EP rank (1024 / 64)
```

### 6. Expert Pruning & Merging

Remove redundant experts post-training:

```python
def compute_expert_similarity(expert1, expert2):
    """Compute cosine similarity between expert weights."""
    w1 = torch.cat([p.flatten() for p in expert1.parameters()])
    w2 = torch.cat([p.flatten() for p in expert2.parameters()])
    return F.cosine_similarity(w1.unsqueeze(0), w2.unsqueeze(0))

def merge_similar_experts(model, similarity_threshold=0.95):
    """Merge experts that are too similar."""
    for layer in model.layers:
        if hasattr(layer, 'mlp') and isinstance(layer.mlp, MoELayer):
            experts = layer.mlp.experts.experts

            # Compute pairwise similarities
            for i in range(len(experts)):
                for j in range(i+1, len(experts)):
                    sim = compute_expert_similarity(experts[i], experts[j])

                    if sim > similarity_threshold:
                        # Merge expert j into expert i
                        for p1, p2 in zip(experts[i].parameters(),
                                        experts[j].parameters()):
                            p1.data = (p1.data + p2.data) / 2

                        # Mark expert j for removal
                        experts[j] = experts[i]  # Point to same weights
```

### 7. Instruction Tuning for MoE

Special considerations for fine-tuning:

```bash
# Use lower learning rate for fine-tuning
FINETUNE_ARGS=(
    --lr 1e-5                          # Much lower than pretraining
    --min-lr 1e-6
    --lr-warmup-iters 100              # Shorter warmup
    --train-iters 10000                # Fewer iterations
    --lr-decay-style constant          # No decay for instruction tuning
)

# Freeze some experts to preserve knowledge
--freeze-experts "0,1,2,3"             # Freeze first 4 experts
--num-trainable-experts 4              # Only train 4 experts

# Use sequence auxiliary loss for variable-length instructions
--moe-router-load-balancing-type seq_aux_loss
--moe-aux-loss-coeff 1e-3              # Lower coefficient
```

### 8. MoE Inference Optimization

Special optimizations for inference:

```bash
INFERENCE_ARGS=(
    # Use static expert assignment (no routing computation)
    --static-expert-assignment         # Pre-compute routing

    # CUDA graphs for repeated batch sizes
    --use-cuda-graph
    --cuda-graph-num-batches 4

    # KV cache optimization
    --use-flash-attn                   # Fast attention
    --max-tokens-to-oom 8192          # KV cache budget

    # Reduced precision
    --fp8-hybrid                       # FP8 inference
    --moe-router-dtype fp16            # FP16 router (faster)
)

# Export to TensorRT-LLM for max performance
python tools/export/trtllm_export.py \
    --model-type gpt \
    --moe \
    --checkpoint-dir /path/to/checkpoint \
    --output-dir /path/to/trtllm
```

---

## Checklist: Launching MoE Training

Use this checklist before starting a large-scale MoE training run:

### Pre-Training Checklist

**Model Configuration**:
- [ ] `--num-experts` is set and divides evenly by `--expert-model-parallel-size`
- [ ] `--moe-router-topk` is set (typically 2-8)
- [ ] `--ffn-hidden-size` is appropriate for expert size
- [ ] `--moe-layer-freq` defines MoE vs dense layer pattern

**Parallelism**:
- [ ] `TP × PP × EP × DP × CP = Total GPUs`
- [ ] `--sequence-parallel` is enabled if using EP+TP
- [ ] `--use-distributed-optimizer` is enabled
- [ ] Parallelism matches hardware topology (EP within node if possible)

**Optimizations**:
- [ ] `--moe-grouped-gemm` is enabled (with `--disable-bias-linear`)
- [ ] `--overlap-grad-reduce` and `--overlap-param-gather` are enabled
- [ ] `--recompute-activations` is configured for memory management
- [ ] Load balancing strategy is chosen (aux_loss, expert_bias, etc.)

**Stability**:
- [ ] `--moe-router-dtype fp32` for numerical stability
- [ ] `--clip-grad 1.0` is set
- [ ] Learning rate warmup is sufficient (1000-2000 iters)
- [ ] Global batch size is large enough (512-4096)

**Data**:
- [ ] Data is preprocessed and indexed
- [ ] Tokenizer is compatible with model
- [ ] Data split is set (train/val/test)

**Checkpointing**:
- [ ] `--save` directory is set and has sufficient space
- [ ] `--save-interval` is reasonable (every 1000-5000 iters)
- [ ] `--use-dist-ckpt` for flexible checkpoint loading

**Monitoring**:
- [ ] `--tensorboard-dir` is set
- [ ] `--log-interval` is reasonable (every 1-100 iters)
- [ ] Wandb/MLflow integration configured (if using)

### During Training Checklist

**Monitor these metrics**:
- [ ] Loss is decreasing smoothly
- [ ] Expert utilization is balanced (check logs)
- [ ] GPU utilization > 80%
- [ ] Communication time < 30% of iteration time
- [ ] No frequent OOM errors
- [ ] Gradient norms are stable (< 10)

**If issues occur**:
- [ ] Check expert load balance (all experts used?)
- [ ] Profile communication (is AlltoAll fast?)
- [ ] Check memory usage (stable or growing?)
- [ ] Verify router stability (aux loss not exploding?)

---

## Summary

**Key Takeaways for MoE Training**:

1. **MoE Architecture**:
   - Router assigns tokens to K experts
   - Only K experts activated per token (sparse)
   - Load balancing is critical

2. **Critical Configurations**:
   - Enable `--moe-grouped-gemm` for 2-3x speedup
   - Use `--sequence-parallel` with EP+TP
   - Set appropriate `--expert-model-parallel-size`

3. **GPU Efficiency**:
   - Grouped GEMM (biggest impact)
   - Communication overlap
   - FP8 training on Hopper+
   - Activation checkpointing for memory

4. **Parallelism**:
   - EP distributes experts across GPUs
   - Group-limited routing reduces communication
   - Balance EP/TP/PP/DP for your hardware

5. **Stability**:
   - Use FP32 for router
   - Larger batch sizes
   - Proper load balancing (aux loss or expert bias)
   - Longer warmup

6. **Common Pitfalls**:
   - Expert collapse → Strengthen load balancing
   - High communication → Group-limited routing
   - OOM → Activation checkpointing + higher EP
   - NaN loss → FP32 router + gradient clipping

**Start Simple, Scale Up**:
- Begin with Mixtral-style config (8 experts, top-2, aux_loss)
- Validate on small scale (8-64 GPUs)
- Add optimizations incrementally
- Profile before scaling to 100s-1000s of GPUs

**For Production**:
- Use distributed checkpointing
- Enable all communication overlaps
- Monitor expert utilization continuously
- Have fallback checkpoints every few hours

---

**Next Steps**:
1. Start with a Mixtral-style 8x7B configuration
2. Validate convergence on small dataset
3. Profile and optimize bottlenecks
4. Scale up gradually with appropriate parallelism
5. Consider advanced features (shared experts, group-limited routing) once baseline works

**Useful References**:
- Mixtral paper: https://arxiv.org/abs/2401.04088
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- Switch Transformers: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668

For questions or issues, see:
- GitHub Issues: https://github.com/NVIDIA/Megatron-LM/issues
- Megatron Docs: https://docs.nvidia.com/Megatron-Core/
- MoE Roadmap: https://github.com/NVIDIA/Megatron-LM/issues/1729
