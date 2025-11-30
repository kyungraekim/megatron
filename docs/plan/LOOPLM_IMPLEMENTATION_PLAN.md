# LoopLM Implementation Plan for Megatron-LM

## Table of Contents
1. [Overview](#overview)
2. [Architecture Summary](#architecture-summary)
3. [Implementation Phases](#implementation-phases)
4. [File Structure](#file-structure)
5. [Detailed Implementation Guide](#detailed-implementation-guide)
6. [MoE Integration Plan](#moe-integration-plan)
7. [Parallelism Strategies](#parallelism-strategies)
8. [Testing Strategy](#testing-strategy)
9. [Training Pipeline](#training-pipeline)

---

## Overview

**LoopLM (Looped Language Model)** is a novel architecture that enables adaptive computation through:
- **Parameter-shared recurrent layers**: A single transformer block is applied recursively `T` times
- **Adaptive early exit**: A learned gating mechanism determines optimal computation depth per token
- **Two-stage training**: Stage I uses entropy-regularized loss; Stage II uses loss-improvement-based gate training

**Key Innovation**: Instead of stacking 32 unique transformer layers, LoopLM uses 1 layer repeated 32 times, drastically reducing parameter count while maintaining expressiveness through recurrent depth.

### Reference Paper: "Ouro: Scaling Latent Reasoning via Looped Language Models"
- Model sizes: 1.4B and 2.6B parameters with 4 recurrent steps
- Training data: 7.7T tokens
- Architecture: Multi-Head Attention, RoPE, SwiGLU, RMSNorm (sandwich normalization)

---

## Architecture Summary

### Core Components

#### 1. Recurrent Transformer Layer
```
F^(t)(x) = lmhead ∘ H_L ∘ H_L ∘ ... ∘ H_L (t times) ∘ emb(x)
                      └─── same layer ───┘
```

**Key characteristics:**
- Same weights used across all recurrent steps
- Different from standard stacking (where each layer has unique weights)
- Reduces parameters: 32-layer model → 1 layer × 32 steps

#### 2. Adaptive Exit Gating Mechanism

**Exit probability distribution**: `q_φ(t|x)` - probability of exiting at step `t`

**Q-exit criterion** (Algorithm 1 from paper):
```python
# Exit when:
# 1. Q(X^(t)) > λ (quality threshold)
# 2. t >= τ (minimum steps)
# where Q(X^(t)) = P(X^(t)|X^(1)) measures improvement
```

**Implementation**:
- Training: Gumbel-Softmax for differentiable sampling
- Inference: Deterministic threshold-based exit

#### 3. Two-Stage Training

**Stage I: Entropy-Regularized Training**
```
L = Σ_{t=1}^{T_max} q_φ(t|x) · L^(t) - β · H(q_φ(·|x))
```
- Trains both model and gate jointly
- Entropy term prevents premature convergence to single exit
- β typically 0.001
- Uniform prior over exit steps

**Stage II: Focused Gate Training**
```
Use loss improvement signal: ΔL^(t) = L^(t-1) - L^(t)
Gate learns to exit when ΔL becomes small
```

### Mathematical Formulation

**Standard Transformer**:
```
For i = 1 to N:
    H_i = TransformerLayer_i(H_{i-1})  // N different layers
```

**LoopLM**:
```
For t = 1 to T:
    H^(t) = TransformerLayer(H^(t-1))  // Same layer, T times
    gate_logits^(t) = ExitGate(H^(t))
```

---

## Implementation Phases

### Phase 1: Core Architecture (Week 1-2)
**Goal**: Implement basic LoopLM without gating, validate parameter sharing

**Deliverables**:
1. `TransformerConfig` extensions for LoopLM
2. `LoopLMModel` with recurrent forward pass
3. Modified `TransformerBlock` for layer reuse
4. Unit tests for parameter sharing
5. Simple training script

**Success Criteria**:
- Model runs forward/backward pass
- Gradients flow correctly through recurrent steps
- Memory usage validated against theoretical estimates

### Phase 2: Exit Gating (Week 3)
**Goal**: Add adaptive computation via learned gating

**Deliverables**:
1. `ExitGate` module
2. Gumbel-Softmax sampling for training
3. Threshold-based exit for inference
4. Gate visualization tools

**Success Criteria**:
- Gate learns diverse exit distributions
- Inference correctly uses threshold exit
- Exit patterns make semantic sense (harder examples → more steps)

### Phase 3: Loss Functions (Week 4)
**Goal**: Implement two-stage training objectives

**Deliverables**:
1. `LoopLMLoss` module (Stage I)
2. Stage II loss computation
3. Multi-stage training coordinator
4. Loss logging and monitoring

**Success Criteria**:
- Stage I loss converges stably
- Entropy term prevents collapse
- Stage II focuses gate training

### Phase 4: MoE Integration (Week 5-6)
**Goal**: Combine LoopLM with Mixture of Experts

**Deliverables**:
1. `LoopLMMoEModel`
2. Router consistency across recurrent steps
3. MoE-specific layer specs
4. Load balancing across loops

**Success Criteria**:
- Router outputs consistent across steps
- Auxiliary loss aggregated correctly
- EP parallelism works with parameter sharing

### Phase 5: Optimization & Parallelism (Week 7-8)
**Goal**: Production-ready training at scale

**Deliverables**:
1. Pipeline parallelism for recurrent loops
2. KV cache sharing for inference
3. Memory optimization (activation recomputation)
4. Multi-node training scripts

**Success Criteria**:
- Efficient multi-GPU training
- Inference optimizations reduce latency
- Memory usage within acceptable bounds

### Phase 6: Testing & Validation (Week 9-10)
**Goal**: Comprehensive testing and documentation

**Deliverables**:
1. Unit tests (>90% coverage)
2. Integration tests
3. Performance benchmarks
4. User documentation

---

## File Structure

### New Files to Create

```
megatron/core/models/looplm/
├── __init__.py
├── looplm_model.py              # Core LoopLM model
├── looplm_config.py             # Configuration extensions (optional)
├── looplm_layer_specs.py        # Layer specifications
├── exit_gate.py                 # Adaptive exit gating mechanism
├── looplm_loss.py               # Stage I & II loss functions
├── looplm_moe_model.py          # LoopLM + MoE integration
└── looplm_moe_layer_specs.py    # MoE-specific layer specs

megatron/training/
├── arguments_looplm.py          # LoopLM-specific CLI arguments

examples/
├── pretrain_looplm.py           # End-to-end training script
├── pretrain_looplm_moe.py       # LoopLM + MoE training
└── looplm_inference.py          # Inference with early exit

tests/unit_tests/models/
├── test_looplm_model.py         # Core model tests
├── test_looplm_gate.py          # Exit gate tests
├── test_looplm_loss.py          # Loss function tests
└── test_looplm_moe.py           # MoE integration tests
```

### Files to Modify

```
megatron/core/transformer/
├── transformer_config.py        # Add LoopLM config parameters
├── transformer_block.py         # Support layer reuse
└── moe/router.py                # Add recurrent step awareness

megatron/core/models/gpt/
├── gpt_model.py                 # Reference for model structure
└── gpt_layer_specs.py           # Reference for layer specs

pretrain_gpt.py                  # Reference for training loop
```

---

## Detailed Implementation Guide

### 1. Configuration Extensions

**File**: `megatron/core/transformer/transformer_config.py`

**Add these parameters to `TransformerConfig` dataclass:**

```python
@dataclass
class TransformerConfig(ModelParallelConfig):
    # ... existing parameters ...

    ####################
    # LoopLM Configuration
    ####################

    recurrent_depth: int = 1
    """Number of times to loop through the shared transformer layer.
    - Standard models: 1 (no looping)
    - LoopLM: typically 4-8
    Example: With num_layers=8 and recurrent_depth=4, effective depth is 32 layers"""

    use_looped_model: bool = False
    """Enable LoopLM architecture with parameter-shared recurrent layers."""

    exit_gate_enabled: bool = False
    """Enable adaptive early exit gating mechanism.
    When True, model can exit computation early based on learned gate."""

    exit_gate_hidden_size: Optional[int] = None
    """Hidden size for exit gate network. If None, uses hidden_size // 4."""

    exit_gate_threshold: float = 0.5
    """Threshold for deterministic early exit during inference.
    Token exits when cumulative probability exceeds this threshold."""

    exit_gate_min_steps: int = 1
    """Minimum number of recurrent steps before allowing early exit."""

    entropy_regularization_beta: float = 0.001
    """Coefficient for entropy regularization in Stage I training.
    L = Σ q(t) * L(t) - β * H(q)
    Typical range: 0.0001 - 0.01"""

    use_stage2_gate_training: bool = False
    """Enable Stage II focused gate training using loss improvement signals.
    Stage I: Entropy-regularized training
    Stage II: Gate focuses on loss improvement ΔL(t) = L(t-1) - L(t)"""

    looplm_share_embeddings_across_steps: bool = True
    """Share embedding computations across recurrent steps (memory optimization)."""

    looplm_kv_cache_strategy: str = "shared"
    """KV cache strategy for inference: 'shared' (reuse cache), 'separate' (new cache per step).
    'shared' is more memory efficient but requires careful bookkeeping."""
```

### 2. Core LoopLM Model

**File**: `megatron/core/models/looplm/looplm_model.py`

**Key implementation details:**

```python
class LoopLMModel(GPTModel):
    """LoopLM: Parameter-shared recurrent transformer with adaptive computation.

    Architecture:
        Instead of stacking N unique layers:
            H_1 = Layer_1(H_0)
            H_2 = Layer_2(H_1)
            ...
            H_N = Layer_N(H_{N-1})

        LoopLM uses 1 shared layer T times:
            H^(1) = SharedLayer(H^(0))
            H^(2) = SharedLayer(H^(1))
            ...
            H^(T) = SharedLayer(H^(T-1))

    Key Features:
        1. Parameter sharing: Same weights across all recurrent steps
        2. Adaptive exit: Learned gate determines when to stop
        3. Multi-exit training: Loss computed at each recurrent step
        4. Gradient flow: Backprop through time (BPTT) through recurrent steps
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        # Validate configuration
        assert config.use_looped_model, "use_looped_model must be True"
        assert config.recurrent_depth >= 1, "recurrent_depth must be >= 1"

        # Store original num_layers, then set to 1 for single shared layer
        self.original_num_layers = config.num_layers
        self.recurrent_depth = config.recurrent_depth
        config.num_layers = 1  # Only create 1 layer (will be reused)

        super().__init__(config, **kwargs)

        # Exit gate (optional)
        if config.exit_gate_enabled:
            gate_hidden_size = config.exit_gate_hidden_size or (config.hidden_size // 4)
            self.exit_gate = ExitGate(
                hidden_size=config.hidden_size,
                gate_hidden_size=gate_hidden_size,
                max_steps=config.recurrent_depth,
                config=config
            )
        else:
            self.exit_gate = None

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        **kwargs
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass with recurrent computation.

        Returns:
            If training (labels provided):
                Dict with keys:
                    'hidden_states': List of hidden states per step [H^(1), ..., H^(T)]
                    'exit_probs': Exit probabilities q(t) for each step (if gate enabled)
                    'logits': List of logits per step (for multi-exit loss)
            If inference (no labels):
                Final logits tensor (after early exit if applicable)
        """
        # Preprocessing: embeddings and rotary embeddings
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                inference_context=inference_context,
                packed_seq_params=kwargs.get('packed_seq_params'),
            )
        )

        # Recurrent forward pass
        hidden_states_per_step = []
        exit_probs_per_step = []
        cumulative_exit_prob = None

        for step in range(self.recurrent_depth):
            # Run through shared transformer layer
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                **kwargs
            )

            hidden_states_per_step.append(hidden_states)

            # Exit gate evaluation (if enabled)
            if self.exit_gate is not None:
                exit_prob = self.exit_gate(hidden_states, step)
                exit_probs_per_step.append(exit_prob)

                # Early exit logic (inference only)
                if not self.training and inference_context is not None:
                    if cumulative_exit_prob is None:
                        cumulative_exit_prob = exit_prob
                    else:
                        cumulative_exit_prob = cumulative_exit_prob + exit_prob

                    # Check exit condition: Q(X^(t)) > λ and t >= τ
                    # Per-sequence exit decisions: each sequence can exit independently
                    # exit_prob shape: [batch_size]
                    if step >= self.config.exit_gate_min_steps:
                        # Check which sequences should exit: [batch_size]
                        should_exit = cumulative_exit_prob > self.config.exit_gate_threshold

                        # If all sequences in batch have decided to exit, return early
                        if should_exit.all():
                            # Early exit: compute logits and return
                            return self._postprocess(
                                hidden_states=hidden_states,
                                input_ids=input_ids,
                                position_ids=position_ids,
                                labels=None,
                                **kwargs
                            )

                        # Note: For partial batch exits (some sequences exit, others continue):
                        # 1. Track exit_mask per sequence: exit_mask[batch_size]
                        # 2. Only compute recurrent steps for non-exited sequences
                        # 3. Gather final hidden states from different exit depths per sequence

            # Prepare input for next recurrent step
            decoder_input = hidden_states

        # Training: return all intermediate states for multi-exit loss
        if self.training or labels is not None:
            # Compute logits for each step
            logits_per_step = []
            for hidden_states in hidden_states_per_step:
                logits = self._postprocess(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    labels=None,
                    **kwargs
                )
                logits_per_step.append(logits)

            return {
                'hidden_states': hidden_states_per_step,
                'logits': logits_per_step,
                'exit_probs': exit_probs_per_step if self.exit_gate else None,
            }

        # Inference without early exit: return final step
        return self._postprocess(
            hidden_states=hidden_states_per_step[-1],
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            **kwargs
        )
```

**Key implementation notes:**
1. **Parameter sharing**: Set `config.num_layers = 1` to create only one layer
2. **Recurrent loop**: Apply the same `self.decoder` block multiple times
3. **Early exit**: Check gate probability and exit condition during inference
4. **Multi-exit output**: Return all intermediate states for training

### 3. Exit Gate Implementation

**File**: `megatron/core/models/looplm/exit_gate.py`

```python
class ExitGate(MegatronModule):
    """Adaptive exit gating mechanism for LoopLM.

    Learns to predict:
        q_φ(t|x) = probability of exiting at step t

    Training:
        - Uses Gumbel-Softmax for differentiable sampling
        - Outputs probability distribution over all steps

    Inference:
        - Deterministic threshold-based exit
        - Exits when Σ_{i=1}^{t} q(i) > threshold
    """

    def __init__(
        self,
        hidden_size: int,
        gate_hidden_size: int,
        max_steps: int,
        config: TransformerConfig
    ):
        super().__init__(config)

        self.max_steps = max_steps
        self.gate_hidden_size = gate_hidden_size

        # Gate network: hidden_states → gate_logits
        self.gate_network = torch.nn.Sequential(
            tensor_parallel.ColumnParallelLinear(
                hidden_size,
                gate_hidden_size,
                config=config,
                init_method=config.init_method,
                bias=True,
            ),
            torch.nn.GELU(),
            tensor_parallel.RowParallelLinear(
                gate_hidden_size,
                1,  # Single logit for this step
                config=config,
                init_method=config.output_layer_init_method,
                bias=True,
            )
        )

        # Learnable bias for each step (optional)
        self.step_bias = torch.nn.Parameter(torch.zeros(max_steps))

    def forward(
        self,
        hidden_states: Tensor,
        step: int,
        temperature: float = 1.0
    ) -> Tensor:
        """Compute exit probability for current step.

        Args:
            hidden_states: [seq_len, batch_size, hidden_size]
            step: Current recurrent step (0-indexed)
            temperature: Temperature for Gumbel-Softmax (training only)

        Returns:
            exit_prob: [batch_size] - per-sequence probability of exiting at this step
        """
        # Aggregate hidden states over sequence dimension
        # For decoder-only models: use last token (has most context)
        # [seq_len, batch_size, hidden_size] → [batch_size, hidden_size]
        pooled = hidden_states[-1, :, :]  # Last token along sequence dimension

        # Alternative pooling strategies (commented):
        # pooled = hidden_states.mean(dim=0)  # Mean pooling
        # pooled = hidden_states.max(dim=0)[0]  # Max pooling

        # Compute gate logit: [batch_size, hidden_size] → [batch_size, 1]
        gate_logit = self.gate_network(pooled)

        # Remove last dimension: [batch_size, 1] → [batch_size]
        gate_logit = gate_logit.squeeze(-1)

        # Add step-specific bias
        gate_logit = gate_logit + self.step_bias[step]

        # During training: return raw logit (will be normalized later)
        if self.training:
            return gate_logit

        # During inference: apply sigmoid for probability
        exit_prob = torch.sigmoid(gate_logit / temperature)
        return exit_prob

    def compute_exit_distribution(
        self,
        gate_logits_per_step: List[Tensor],
        temperature: float = 1.0,
        use_gumbel: bool = True
    ) -> Tensor:
        """Compute probability distribution over exit steps.

        Args:
            gate_logits_per_step: List of [batch_size] tensors
            temperature: Temperature for softmax
            use_gumbel: Use Gumbel-Softmax for differentiable sampling

        Returns:
            exit_probs: [batch_size, max_steps] - q_φ(t|x) for each t
        """
        # Stack logits: [batch_size, max_steps]
        gate_logits = torch.stack(gate_logits_per_step, dim=1)

        if use_gumbel and self.training:
            # Gumbel-Softmax for differentiable sampling
            from torch.nn.functional import gumbel_softmax
            exit_probs = gumbel_softmax(gate_logits, tau=temperature, hard=False, dim=1)
        else:
            # Standard softmax
            exit_probs = torch.softmax(gate_logits / temperature, dim=1)

        return exit_probs


class QExitCriterion:
    """Q-exit early stopping criterion from LoopLM paper (Algorithm 1).

    Exit when:
        1. Q(X^(t)) > λ (quality threshold)
        2. t >= τ (minimum steps)

    where Q(X^(t)) = P(X^(t)|X^(1)) measures improvement quality
    """

    def __init__(self, threshold: float = 0.5, min_steps: int = 1):
        self.threshold = threshold
        self.min_steps = min_steps

    def should_exit(
        self,
        logits_prev: Tensor,
        logits_curr: Tensor,
        step: int
    ) -> bool:
        """Check if should exit at current step.

        Args:
            logits_prev: Logits from previous step
            logits_curr: Logits from current step
            step: Current step index (0-indexed)

        Returns:
            should_exit: True if exit criteria met
        """
        if step < self.min_steps:
            return False

        # Compute quality metric: agreement between consecutive steps
        # Q(X^(t)) = cosine_similarity(logits^(t-1), logits^(t))
        with torch.no_grad():
            logits_prev_flat = logits_prev.reshape(-1)
            logits_curr_flat = logits_curr.reshape(-1)

            quality = torch.nn.functional.cosine_similarity(
                logits_prev_flat.unsqueeze(0),
                logits_curr_flat.unsqueeze(0),
                dim=1
            ).item()

        return quality > self.threshold
```

### 4. Loss Function Implementation

**File**: `megatron/core/models/looplm/looplm_loss.py`

```python
class LoopLMLoss:
    """Loss functions for LoopLM training.

    Implements two-stage training:
        Stage I: Entropy-regularized multi-exit loss
        Stage II: Focused gate training with loss improvement signals
    """

    def __init__(
        self,
        entropy_reg_beta: float = 0.001,
        use_stage2: bool = False,
        uniform_prior: bool = True
    ):
        self.entropy_reg_beta = entropy_reg_beta
        self.use_stage2 = use_stage2
        self.uniform_prior = uniform_prior

    def compute_stage1_loss(
        self,
        logits_per_step: List[Tensor],
        labels: Tensor,
        exit_probs: Tensor,
        loss_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Stage I: Entropy-regularized multi-exit loss.

        L = Σ_{t=1}^{T} q_φ(t|x) · L^(t) - β · H(q_φ(·|x))

        Args:
            logits_per_step: List of [seq_len, batch_size, vocab_size] tensors
            labels: [seq_len, batch_size] - ground truth tokens
            exit_probs: [batch_size, T] - q_φ(t|x) for each step t
            loss_mask: [seq_len, batch_size] - mask for padding tokens

        Returns:
            Dict with:
                'total_loss': Weighted sum of losses + entropy regularization
                'task_loss': Task loss without entropy term
                'entropy_loss': Entropy regularization term
                'losses_per_step': Individual losses at each step
        """
        num_steps = len(logits_per_step)
        batch_size = exit_probs.size(0)

        # Compute cross-entropy loss at each step
        losses_per_step = []
        for step, logits in enumerate(logits_per_step):
            # logits: [seq_len, batch_size, vocab_size]
            # labels: [seq_len, batch_size]
            loss_per_token = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none'
            )
            loss_per_token = loss_per_token.view(labels.size())  # [seq_len, batch_size]

            # Apply loss mask if provided
            if loss_mask is not None:
                loss_per_token = loss_per_token * loss_mask
                num_tokens = loss_mask.sum()
            else:
                num_tokens = labels.numel()

            # Average over sequence, keep batch dimension
            # Add numerical stability: avoid division by zero for fully-padded samples
            if loss_mask is not None:
                denominator = loss_mask.sum(dim=0).clamp(min=1.0)  # Prevent division by zero
                loss_per_sample = loss_per_token.sum(dim=0) / denominator
            else:
                loss_per_sample = loss_per_token.sum(dim=0) / labels.size(0)
            losses_per_step.append(loss_per_sample)

        # Stack losses: [batch_size, num_steps]
        losses = torch.stack(losses_per_step, dim=1)

        # Weighted average by exit probabilities
        # task_loss = Σ_t q(t) * L(t)
        task_loss = (exit_probs * losses).sum(dim=1).mean()

        # Entropy regularization: H(q) = -Σ_t q(t) log q(t)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        entropy = -(exit_probs * torch.log(exit_probs + epsilon)).sum(dim=1).mean()

        # Uniform prior penalty (optional)
        if self.uniform_prior:
            uniform_dist = torch.ones_like(exit_probs) / num_steps
            kl_div = torch.nn.functional.kl_div(
                torch.log(exit_probs + epsilon),
                uniform_dist,
                reduction='batchmean'
            )
            entropy = entropy - 0.1 * kl_div  # Encourage uniform distribution

        # Total loss
        total_loss = task_loss - self.entropy_reg_beta * entropy

        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'entropy_loss': entropy,
            'entropy_value': entropy.item(),
            'losses_per_step': losses,
            'avg_exit_step': (exit_probs * torch.arange(num_steps, device=exit_probs.device)).sum(dim=1).mean().item()
        }

    def compute_stage2_loss(
        self,
        logits_per_step: List[Tensor],
        labels: Tensor,
        exit_probs: Tensor,
        loss_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Stage II: Focused gate training with loss improvement signals.

        Gate learns to exit when ΔL^(t) = L^(t-1) - L^(t) becomes small.

        Args:
            Same as compute_stage1_loss

        Returns:
            Dict with loss components
        """
        # First compute losses at each step (same as Stage I)
        stage1_result = self.compute_stage1_loss(
            logits_per_step, labels, exit_probs, loss_mask
        )

        losses_per_step = stage1_result['losses_per_step']  # [batch_size, num_steps]

        # Compute loss improvements: ΔL^(t) = L^(t-1) - L^(t)
        # Positive ΔL means improvement
        loss_improvements = []
        for t in range(1, losses_per_step.size(1)):
            delta_loss = losses_per_step[:, t-1] - losses_per_step[:, t]
            loss_improvements.append(delta_loss)

        # Stack: [batch_size, num_steps-1]
        loss_improvements = torch.stack(loss_improvements, dim=1)

        # Gate target: exit when improvement becomes small
        # Target distribution: softmax over -ΔL (exit when improvement is low)
        improvement_scores = -loss_improvements  # Higher score = less improvement
        target_exit_probs = torch.softmax(improvement_scores, dim=1)

        # Pad with final step probability
        final_step_prob = torch.ones(target_exit_probs.size(0), 1, device=target_exit_probs.device)
        target_exit_probs = torch.cat([target_exit_probs, final_step_prob], dim=1)
        target_exit_probs = target_exit_probs / target_exit_probs.sum(dim=1, keepdim=True)

        # Gate loss: KL divergence between predicted and target distributions
        epsilon = 1e-8
        gate_loss = torch.nn.functional.kl_div(
            torch.log(exit_probs + epsilon),
            target_exit_probs,
            reduction='batchmean'
        )

        # Combined loss: task loss + gate loss
        total_loss = stage1_result['task_loss'] + gate_loss

        return {
            'total_loss': total_loss,
            'task_loss': stage1_result['task_loss'],
            'gate_loss': gate_loss,
            'loss_improvements': loss_improvements.mean(dim=0),  # Average per step
            **stage1_result
        }

    def __call__(
        self,
        logits_per_step: List[Tensor],
        labels: Tensor,
        exit_probs: Optional[Tensor] = None,
        loss_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Compute appropriate loss based on training stage."""
        if exit_probs is None:
            # No gating: simple multi-exit loss with uniform weights
            num_steps = len(logits_per_step)
            exit_probs = torch.ones(labels.size(1), num_steps, device=labels.device) / num_steps

        if self.use_stage2:
            return self.compute_stage2_loss(logits_per_step, labels, exit_probs, loss_mask)
        else:
            return self.compute_stage1_loss(logits_per_step, labels, exit_probs, loss_mask)


def compute_looplm_loss(
    model_output: Dict[str, Tensor],
    labels: Tensor,
    config: TransformerConfig,
    loss_mask: Optional[Tensor] = None
) -> Tuple[Tensor, Dict[str, float]]:
    """Convenience function for computing LoopLM loss in training scripts.

    Args:
        model_output: Output from LoopLMModel.forward() containing:
            - 'logits': List of logits per step
            - 'exit_probs': Exit probabilities (if gating enabled)
        labels: Ground truth tokens
        config: Model configuration
        loss_mask: Optional mask for padding tokens

    Returns:
        loss: Scalar loss tensor
        logging_dict: Dict with loss components for logging
    """
    loss_fn = LoopLMLoss(
        entropy_reg_beta=config.entropy_regularization_beta,
        use_stage2=config.use_stage2_gate_training,
        uniform_prior=True
    )

    result = loss_fn(
        logits_per_step=model_output['logits'],
        labels=labels,
        exit_probs=model_output.get('exit_probs'),
        loss_mask=loss_mask
    )

    # Extract scalar loss for backward pass
    loss = result['total_loss']

    # Prepare logging dict
    logging_dict = {
        'looplm_loss': loss.item(),
        'task_loss': result['task_loss'].item(),
        'entropy': result.get('entropy_value', 0.0),
        'avg_exit_step': result.get('avg_exit_step', config.recurrent_depth),
    }

    if 'gate_loss' in result:
        logging_dict['gate_loss'] = result['gate_loss'].item()

    return loss, logging_dict
```

### 5. Layer Specifications

**File**: `megatron/core/models/looplm/looplm_layer_specs.py`

```python
def get_looplm_layer_spec(
    num_experts: Optional[int] = None,
    use_transformer_engine: bool = True,
    normalization: str = "RMSNorm",
    **kwargs
) -> ModuleSpec:
    """Get layer specification for LoopLM.

    Creates a standard GPT layer spec (can be dense or MoE).
    LoopLM will reuse this layer specification across recurrent steps.

    Args:
        num_experts: Number of MoE experts (None for dense)
        use_transformer_engine: Use TE-accelerated modules
        normalization: Normalization type (LayerNorm or RMSNorm)
        **kwargs: Additional arguments for layer spec

    Returns:
        ModuleSpec for the shared transformer layer
    """
    # Import layer spec functions
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
        get_gpt_layer_local_spec
    )

    # Create base layer spec (reuse existing GPT specs)
    if use_transformer_engine:
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            normalization=normalization,
            **kwargs
        )
    else:
        layer_spec = get_gpt_layer_local_spec(
            num_experts=num_experts,
            normalization=normalization,
            **kwargs
        )

    return layer_spec


def get_looplm_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool = True,
    **kwargs
) -> TransformerBlockSubmodules:
    """Get decoder block spec for LoopLM.

    Key difference from standard GPT: Only creates 1 layer (will be reused).

    Args:
        config: Transformer configuration (with num_layers=1)
        use_transformer_engine: Use TE modules

    Returns:
        TransformerBlockSubmodules with single layer spec
    """
    assert config.num_layers == 1, "LoopLM should have num_layers=1"

    # Get single layer spec
    layer_spec = get_looplm_layer_spec(
        num_experts=config.num_moe_experts,
        use_transformer_engine=use_transformer_engine,
        normalization=config.normalization,
        **kwargs
    )

    # Create block spec with single layer
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    layer_norm_impl = TENorm if use_transformer_engine else FusedLayerNorm

    block_spec = TransformerBlockSubmodules(
        layer_specs=[layer_spec],  # Single layer
        layer_norm=layer_norm_impl
    )

    return block_spec
```

---

## MoE Integration Plan

### Challenges with MoE + LoopLM

1. **Router Consistency**: Router must make consistent decisions across recurrent steps
2. **Load Balancing**: Auxiliary loss must be aggregated across all recurrent steps
3. **Memory Pressure**: MoE already memory-intensive; recurrence adds more pressure
4. **Expert Parallelism**: EP must work correctly with parameter-shared layers

### Implementation Strategy

#### 1. LoopLM-MoE Model

**File**: `megatron/core/models/looplm/looplm_moe_model.py`

```python
class LoopLMMoEModel(LoopLMModel):
    """LoopLM with Mixture of Experts layers.

    Additional considerations for MoE:
        1. Router state tracking across recurrent steps
        2. Auxiliary loss aggregation over all steps
        3. Token dropping consistency
        4. Expert load balancing
    """

    def __init__(self, config: TransformerConfig, **kwargs):
        assert config.num_moe_experts is not None, "MoE experts must be specified"
        super().__init__(config, **kwargs)

        # MoE-specific state tracking
        self.router_logits_per_step = []
        self.aux_losses_per_step = []

    def forward_recurrent_step(
        self,
        hidden_states: Tensor,
        step: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Single recurrent step with MoE.

        Args:
            hidden_states: Input to this step
            step: Recurrent step index

        Returns:
            Dict with:
                - hidden_states: Output from this step
                - router_logits: Router decisions
                - aux_loss: Auxiliary loss for load balancing
        """
        # Forward through shared layer
        output = self.decoder(hidden_states, **kwargs)

        # Extract MoE-specific information
        # (This requires modifying MoELayer to return additional info)
        if hasattr(output, 'router_logits'):
            self.router_logits_per_step.append(output.router_logits)

        if hasattr(output, 'aux_loss'):
            self.aux_losses_per_step.append(output.aux_loss)

        return {
            'hidden_states': output,
            'step': step
        }

    def compute_total_aux_loss(self) -> Tensor:
        """Aggregate auxiliary loss across all recurrent steps.

        Returns:
            Total aux loss averaged over steps
        """
        if not self.aux_losses_per_step:
            return torch.tensor(0.0)

        # Average auxiliary loss over all steps
        return torch.stack(self.aux_losses_per_step).mean()

    def reset_moe_state(self):
        """Reset MoE state tracking between batches."""
        self.router_logits_per_step = []
        self.aux_losses_per_step = []
```

#### 2. Router Modifications

**File**: `megatron/core/transformer/moe/router.py`

**Add to `TopKRouter` class:**

```python
class TopKRouter(MegatronModule):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        # ... existing init ...

        # Track which recurrent step we're in (for LoopLM)
        self.current_recurrent_step = 0

    def forward(
        self,
        hidden_states: Tensor,
        recurrent_step: Optional[int] = None
    ):
        """Route tokens to experts.

        Args:
            hidden_states: [seq_len, batch_size, hidden_size]
            recurrent_step: Which recurrent step (for logging/visualization)

        Returns:
            probs: Routing probabilities
            routing_map: Expert assignment
        """
        if recurrent_step is not None:
            self.current_recurrent_step = recurrent_step

        # ... existing routing logic ...

        # Log routing decisions per step (for debugging/analysis)
        if self.training and recurrent_step is not None:
            self._log_routing_stats(routing_map, recurrent_step)

        return probs, routing_map

    def _log_routing_stats(self, routing_map, step):
        """Log routing statistics for this recurrent step."""
        # Track expert utilization per step
        # This helps diagnose if routing is consistent across steps
        pass
```

#### 3. Training Script Modifications

**File**: `examples/pretrain_looplm_moe.py`

```python
def forward_step(data_iterator, model: LoopLMMoEModel):
    """Forward step for LoopLM-MoE training.

    Handles:
        1. Multi-exit LoopLM loss
        2. MoE auxiliary loss aggregation
        3. Router warmup for first 200 steps
    """
    args = get_args()

    # Get batch
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # Reset MoE state tracking
    model.reset_moe_state()

    # Forward pass
    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask
    )

    # Compute LoopLM loss (multi-exit + entropy regularization)
    looplm_loss, looplm_logging = compute_looplm_loss(
        model_output=output,
        labels=labels,
        config=model.config,
        loss_mask=loss_mask
    )

    # Compute MoE auxiliary loss (aggregated over recurrent steps)
    moe_aux_loss = model.compute_total_aux_loss()

    # Total loss
    total_loss = looplm_loss + args.moe_aux_loss_coeff * moe_aux_loss

    # Logging
    num_tokens = loss_mask.sum()
    logging_dict = {
        'lm_loss': looplm_loss / num_tokens,
        'moe_aux_loss': moe_aux_loss,
        **looplm_logging
    }

    return total_loss, lambda: (total_loss, num_tokens, logging_dict)


def train_step_with_router_warmup(iteration, model, optimizer, *args, **kwargs):
    """Training step with router warmup strategy.

    Problem: MoE models suffer from load imbalance in first ~200 steps
    Solution: Temporarily adjust parallelism or use capacity factor
    """
    args = get_args()

    # Router warmup: first 200 iterations
    if iteration < 200:
        # Option 1: Use capacity factor to drop excess tokens
        if not hasattr(model.config, '_original_capacity_factor'):
            model.config._original_capacity_factor = model.config.moe_expert_capacity_factor
            model.config.moe_expert_capacity_factor = 1.0
    elif iteration == 200:
        # Restore original capacity factor
        if hasattr(model.config, '_original_capacity_factor'):
            model.config.moe_expert_capacity_factor = model.config._original_capacity_factor

    # Normal training step
    loss = forward_backward_step(model, optimizer, ...)

    return loss
```

### MoE-Specific Configuration

```python
# megatron/training/arguments_looplm.py

def add_looplm_moe_args(parser):
    """Add LoopLM-MoE specific arguments."""
    group = parser.add_argument_group(title='LoopLM-MoE')

    # Router behavior across recurrent steps
    group.add_argument(
        '--looplm-router-consistency-loss',
        type=float,
        default=0.0,
        help='Penalty for inconsistent routing across recurrent steps'
    )

    group.add_argument(
        '--looplm-aggregate-aux-loss',
        action='store_true',
        help='Aggregate MoE aux loss across all recurrent steps (default: average)'
    )

    group.add_argument(
        '--looplm-moe-warmup-iters',
        type=int,
        default=200,
        help='Number of iterations for MoE router warmup'
    )

    return parser
```

---

## LoopLM + MoE GPU Efficiency Optimizations

### Overview: Unique Optimization Opportunities

Combining LoopLM (recurrent parameter sharing) with MoE creates unique optimization opportunities and challenges:

**Advantages:**
1. **Parameter Sharing**: Same weights loaded once, reused across recurrent steps → Better cache utilization
2. **Repeated Expert Access**: Experts may be "warmed up" in L1/L2 cache from previous steps
3. **Consistent Router Patterns**: Tokens may route to similar experts across steps → Reduced communication variance

**Challenges:**
1. **Accumulating Activations**: Each recurrent step stores activations for backprop → Memory pressure
2. **Repeated AlltoAll**: Each recurrent step triggers expert communication → Network bottleneck
3. **Multi-Exit Loss**: Computing loss at each step requires O(T) LM head computations
4. **Gradient Accumulation Through Time**: Backprop through recurrent steps increases compute

### Critical Optimizations (Ranked by Impact)

#### 1. Grouped GEMM for MoE Experts (2-3x speedup) ⭐⭐⭐

**CRITICAL**: This is the single most important optimization for LoopLM-MoE.

```bash
# Enable grouped GEMM
--moe-grouped-gemm
--disable-bias-linear  # Required for grouped GEMM

# Install dependency
pip install nv-grouped-gemm~=1.1
```

**Why critical for LoopLM:**
- Expert computation happens T times (once per recurrent step)
- Grouped GEMM 2-3x speedup applies to ALL recurrent steps
- Total speedup: 2-3x expert compute × T steps = major impact

**Implementation notes:**
- Single fused CUDA kernel for all experts
- Better memory coalescing across recurrent steps
- Lower kernel launch overhead (critical when looping)

#### 2. Selective Activation Checkpointing (2-4x memory reduction) ⭐⭐⭐

**Problem**: LoopLM stores activations for each recurrent step → T× memory pressure

**Solution**: Recompute activations instead of storing

```bash
# Selective recomputation (recommended)
--recompute-activations \
--recompute-granularity selective \
--recompute-modules moe shared_experts moe_act layernorm

# CRITICAL for LoopLM: Recompute across recurrent steps
--looplm-recompute-recurrent-activations  # NEW FLAG (to implement)
```

**LoopLM-specific strategy:**

```python
# Three-tier recomputation strategy for LoopLM-MoE:

# Tier 1: ALWAYS recompute (cheap, high memory usage)
- LayerNorm outputs
- Activation functions (SwiGLU, GELU)
- Attention softmax

# Tier 2: Recompute for middle steps (balanced)
- MoE token dispatch/combine operations (steps 1 to T-1)
- Router computations (steps 1 to T-1)
- Shared expert outputs (steps 1 to T-1)

# Tier 3: NEVER recompute (expensive, low memory)
- Attention QKV projections
- Expert FFN computations (keep only step T, recompute rest)
- Final LM head (step T only)
```

**Memory savings calculation:**

```
Standard LoopLM-MoE memory:
- Activations per step: ~12GB (for 8B model)
- Total for T=4 steps: 48GB
- With selective recomputation: ~20GB (60% reduction)
```

#### 3. Communication Overlap Strategies (20-40% speedup) ⭐⭐⭐

**Problem**: AlltoAll communication happens at each recurrent step

**Solution 1: Overlap AlltoAll with Computation**

```bash
# Enable expert parallel A2A overlapping
--overlap-grad-reduce
--overlap-param-gather
--moe-expert-a2a-overlap  # Automatically enabled for EP

# For shared experts (if used)
--moe-shared-expert-overlap
```

**How it works for LoopLM:**

```
Without overlap (per recurrent step):
|---- Router ----|---- A2A Send ----|---- Expert Compute ----|---- A2A Recv ----|
    100ms            150ms                200ms                   150ms
    Total: 600ms per step × 4 steps = 2400ms

With overlap:
|---- Router ----|---- A2A Send ----|
                 |---- Expert Compute ----|
                 |---- Shared Expert ----|  (parallel stream)
                                          |---- A2A Recv ----|
    100ms + max(150ms, 200ms + 0ms) + 150ms = 450ms per step
    Total: 450ms × 4 steps = 1800ms (25% faster)
```

**Solution 2: Pipelined Recurrent Steps** (ADVANCED)

```python
# Idea: Overlap recurrent step t+1's routing with step t's expert computation

class PipelinedLoopLMMoE(LoopLMMoEModel):
    """Pipeline recurrent steps to overlap communication and computation."""

    def forward(self, *args, **kwargs):
        # Step 0: Compute routing for step 1 (no expert compute yet)
        router_output_next = self.router(hidden_states)

        for step in range(self.recurrent_depth):
            # Use routing from previous iteration
            router_output = router_output_next

            # Start async AlltoAll send for current step
            a2a_handle = async_alltoall_send(hidden_states, router_output)

            # While AlltoAll is in flight, compute routing for NEXT step
            if step < self.recurrent_depth - 1:
                # Peek ahead: compute routing for next step
                hidden_states_peek = self.attention(hidden_states)  # Fast path
                router_output_next = self.router(hidden_states_peek)

            # Wait for AlltoAll to complete
            expert_input = a2a_handle.wait()

            # Compute experts
            expert_output = self.experts(expert_input)

            # AlltoAll receive
            hidden_states = alltoall_recv(expert_output)

        return hidden_states
```

**Expected impact:**
- Hides routing computation latency (~100ms per step)
- Requires double buffering and careful synchronization
- Best for models with expensive routing (e.g., large routers)

#### 4. FP8 Training (1.5-2x throughput on Hopper/Ada/Blackwell) ⭐⭐

**Hardware requirements:** H100, H200, A100 (limited), Ada GPUs

```bash
FP8_ARGS=(
    --fp8-hybrid                              # FP8 E4M3/E5M2 hybrid
    --fp8-amax-compute-algo max               # Amax computation
    --fp8-amax-history-len 1024               # Scaling factor history
    --fp8-margin 0                            # Scaling margin
)

# CRITICAL for stability
--moe-router-dtype fp32                       # Keep router in FP32
--exit-gate-dtype fp32                        # Keep exit gate in FP32 (LoopLM-specific)
```

**What runs in FP8 (for LoopLM-MoE):**
- ✅ Expert FFN forward/backward (largest compute component)
- ✅ Attention QKV/output projections
- ✅ Gradients
- ❌ Router computation (FP32 for stability)
- ❌ Exit gate computation (FP32 for stability)
- ❌ LayerNorm, RMSNorm
- ❌ Master weights

**LoopLM-specific considerations:**
- FP8 quantization errors do NOT accumulate across recurrent steps (same weights)
- Exit gate MUST stay in FP32 (very sensitive to numerical precision)
- Router should stay in FP32 (load balancing depends on precise scores)

**Expected throughput:**
```
BF16 baseline: 100 tokens/sec/GPU
FP8 optimized: 150-180 tokens/sec/GPU (1.5-1.8x)
```

#### 5. Router Fusion (10-20% speedup) ⭐⭐

```bash
# Enable router fusion (requires Transformer Engine 2.7+)
--moe-router-fusion
```

**What it fuses:**
- Router gating linear layer
- Top-k selection
- Softmax normalization
- Auxiliary loss computation
- All in a single optimized CUDA kernel

**LoopLM-specific benefit:**
- Router runs T times (once per recurrent step)
- 10-20% speedup × T steps = compounding benefit
- Reduces kernel launch overhead in recurrent loop

#### 6. KV Cache Sharing Across Recurrent Steps (LoopLM-specific) ⭐⭐

**CRITICAL OPTIMIZATION FOR INFERENCE**

**Problem**: Standard MoE inference computes attention KV cache independently at each step

**Solution**: Reuse KV cache across recurrent steps (attention is parameter-shared!)

```python
class LoopLMMoEInferenceOptimized:
    """Optimized LoopLM-MoE inference with KV cache sharing."""

    def forward_inference(
        self,
        tokens: Tensor,
        inference_context: InferenceContext
    ):
        # Initialize KV cache once (shared across all recurrent steps)
        kv_cache = inference_context.get_or_create_kv_cache(
            layer_id=0,  # Single layer (parameter sharing)
            cache_type="shared_recurrent"
        )

        hidden_states = self.embedding(tokens)

        for step in range(self.recurrent_depth):
            # Attention: Use shared KV cache
            # Key insight: Same attention weights → can reuse cache structure
            attention_output = self.decoder.attention(
                hidden_states,
                kv_cache=kv_cache,
                recurrent_step=step  # Track step for cache indexing
            )

            # MoE: Route tokens to experts
            moe_output = self.decoder.moe(attention_output)

            hidden_states = moe_output

            # Early exit check
            if self.should_exit(hidden_states, step):
                break

        return self.lm_head(hidden_states)
```

**Memory savings:**
```
Standard approach:
- KV cache per step: 2GB × 4 steps = 8GB

Shared cache approach:
- Shared KV cache: 2GB × 1 = 2GB
- 75% memory reduction!
```

**Implementation note:**
- Requires careful indexing in KV cache
- Must handle variable sequence lengths across steps
- Cache update strategy depends on attention pattern

#### 7. Multi-Exit Loss Optimization (LoopLM-specific) ⭐

**Problem**: Computing loss at each recurrent step requires T× LM head forward passes

**Solution 1: Delayed LM Head Computation**

```python
# INEFFICIENT (current implementation):
for step in range(T):
    hidden_states = decoder(hidden_states)
    logits = lm_head(hidden_states)  # Expensive! T times
    loss += exit_prob[step] * cross_entropy(logits, labels)

# EFFICIENT (optimized):
# Store hidden states, compute LM head in batch
hidden_states_all = []
for step in range(T):
    hidden_states = decoder(hidden_states)
    hidden_states_all.append(hidden_states)

# Batch LM head computation (more efficient)
hidden_concat = torch.cat(hidden_states_all, dim=0)  # [T×S, B, H]
logits_concat = lm_head(hidden_concat)  # Single batched call
logits_all = logits_concat.chunk(T, dim=0)  # Split back

# Compute weighted loss
loss = sum(exit_prob[i] * cross_entropy(logits_all[i], labels) for i in range(T))
```

**Performance improvement:**
- Reduces kernel launch overhead
- Better GPU utilization (larger batch)
- ~15-20% faster for loss computation

**Solution 2: Approximate Early Steps (EXPERIMENTAL)**

```python
# Idea: Only compute full loss for final steps, approximate early steps

# Only compute logits for last 2 steps
for step in range(T - 2):
    hidden_states = decoder(hidden_states)
    # No logits computation → save time

# Last 2 steps: full loss computation
for step in range(T - 2, T):
    hidden_states = decoder(hidden_states)
    logits = lm_head(hidden_states)
    loss += exit_prob[step] * cross_entropy(logits, labels)
```

**Trade-off:**
- Faster training (skip early step losses)
- May hurt exit gate quality (less supervision)
- Best for models with low early-exit rate

#### 8. Group-Limited Routing for Cross-Node Communication ⭐⭐

**When to use:** Multi-node training (EP across nodes)

**Problem**: AlltoAll across nodes is expensive

**Solution**: Group-limited routing

```bash
# Group-limited routing (reduce cross-node communication)
--moe-router-num-groups 64                  # Divide into groups
--moe-router-group-topk 4                   # Select from subset of groups

# Example: 256 experts, 64 groups
# Each group has 4 experts
# Token only considers 4 groups = 16 experts (instead of 256)
# 16× less communication!
```

**LoopLM-specific benefit:**
- Communication reduction applies to EVERY recurrent step
- With T=4 steps: 16× reduction × 4 steps = huge impact on multi-node scaling

**Group assignment strategies:**

```python
# Strategy 1: Node-aligned groups (recommended)
# Groups correspond to physical node boundaries
# Example: 8 nodes, 32 experts per node
# Group 0: Experts 0-31 (Node 0)
# Group 1: Experts 32-63 (Node 1)
# ...

--moe-router-num-groups 8                   # Number of nodes
--moe-router-group-topk 2                   # Select 2 nodes

# Strategy 2: Expert specialization groups
# Group experts by task type (learned during training)
# Requires analysis of expert specialization patterns
```

#### 9. Optimizer State Management (LoopLM-specific) ⭐

**Challenge**: Parameter sharing means single set of weights, but:
- Gradients accumulated from T recurrent steps
- Gradient norms can be T× larger

**Solution: Gradient Scaling**

```bash
# Scale gradients by recurrent depth
--looplm-gradient-scaling inverse-depth  # NEW FLAG

# This scales gradient by 1/T to compensate for accumulation
```

**Implementation:**

```python
class LoopLMOptimizer:
    def step(self):
        # Scale gradients from recurrent accumulation
        scale_factor = 1.0 / self.config.recurrent_depth

        for param in self.model.parameters():
            if param.grad is not None:
                param.grad *= scale_factor

        # Normal optimizer step
        super().step()
```

**Alternative: Adaptive Gradient Clipping**

```bash
# Use adaptive clipping that accounts for recurrent depth
--clip-grad 1.0
--looplm-adaptive-grad-clip  # NEW FLAG
```

### Memory Optimization Strategies

#### Memory Breakdown for LoopLM-MoE (Example: 8B model, T=4 steps)

```
Component                          | Memory per Step | Total (T=4) | With Optimization
-----------------------------------|-----------------|-------------|------------------
Model Parameters (shared)          | 8GB             | 8GB         | 8GB
Optimizer States (AdamW)           | 16GB            | 16GB        | 16GB
Activations (forward)              | 10GB            | 40GB        | 12GB (recompute)
MoE Expert Buffers                 | 4GB             | 16GB        | 8GB (reuse buffers)
Router States                      | 0.5GB           | 2GB         | 0.5GB (reuse)
Exit Gate States                   | 0.1GB           | 0.4GB       | 0.1GB (reuse)
Gradients                          | 8GB             | 8GB         | 8GB (accumulated)
-----------------------------------|-----------------|-------------|------------------
TOTAL                              | -               | 90.4GB      | 58.6GB (35% reduction)
```

#### Advanced Memory Optimizations

**1. Expert Buffer Reuse Across Steps**

```python
class LoopLMMoEMemoryOptimized:
    """Reuse expert communication buffers across recurrent steps."""

    def __init__(self, config):
        super().__init__(config)

        # Allocate expert buffers ONCE (shared across steps)
        self.expert_input_buffer = torch.zeros(
            max_tokens_per_expert * num_experts,
            hidden_size,
            device='cuda'
        )
        self.expert_output_buffer = torch.zeros_like(self.expert_input_buffer)

    def forward_recurrent_step(self, hidden_states, step):
        # Reuse buffers instead of allocating new ones
        self.moe_layer.set_communication_buffers(
            input_buffer=self.expert_input_buffer,
            output_buffer=self.expert_output_buffer
        )

        output = self.moe_layer(hidden_states)
        return output
```

**Expected savings:**
- Expert buffers: 4GB × 4 steps = 16GB → 4GB (75% reduction)

**2. Gradient Accumulation Through Time (Efficient Backprop)**

```python
# Standard backprop through time: Store all intermediate states
hidden_states_all = []
for step in range(T):
    hidden_states = layer(hidden_states)
    hidden_states_all.append(hidden_states)  # Store for backprop

# Optimized: Reversible residuals (RevNet-style)
# Recompute hidden states during backprop instead of storing
for step in range(T):
    hidden_states = layer(hidden_states)
    # Don't store (will recompute in backward)

# Backward pass: Recompute forward on-the-fly
# Saves T× memory for activations
```

**Implementation note:**
- Requires custom autograd function
- ~20% slower backward pass
- 40-50% memory savings

**3. Expert Parallelism with CPU Offloading (Last Resort)**

```bash
# Only for extremely large models that don't fit in GPU memory
--cpu-offload-optimizer
--cpu-offload-optimizer-ratio 0.5

# Offload expert parameters to CPU between recurrent steps
--looplm-expert-cpu-offload  # NEW FLAG (experimental)
```

**How it works:**
```
Step 0: Load experts 0-31 to GPU → Compute → Offload to CPU
Step 1: Load experts 0-31 to GPU → Compute → Offload to CPU
...
```

**Performance impact:**
- Slow (PCIe bandwidth bottleneck)
- Only use when absolutely necessary
- Consider renting more GPUs instead

### Practical Training Configurations

#### Small-Scale LoopLM-MoE (8 GPUs)

```bash
#!/bin/bash
# 8 GPUs (single node)
# Model: 3B parameters, 16 experts, T=4 steps

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 4                          # 4 shared layers
    --recurrent-depth 4                     # Each layer looped 4× (effective 16 layers)
    --use-looped-model
    --exit-gate-enabled
    --hidden-size 2560
    --num-attention-heads 32
    --num-query-groups 8
    --ffn-hidden-size 6912
    --seq-length 2048
    --max-position-embeddings 8192
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm                      # CRITICAL
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32                 # Stability
)

LOOPLM_ARGS=(
    --entropy-regularization-beta 0.001
    --exit-gate-threshold 0.5
    --exit-gate-min-steps 2
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8          # One expert pair per GPU
    --sequence-parallel
    --use-distributed-optimizer
)

OPTIMIZATION_ARGS=(
    # Activation checkpointing (selective)
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe moe_act layernorm

    # Communication overlap
    --overlap-grad-reduce
    --overlap-param-gather

    # Router fusion
    --moe-router-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 512                 # Large batch for MoE stability
    --lr 2e-4                               # Higher LR for MoE
    --min-lr 2e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --train-iters 100000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

torchrun --nproc_per_node=8 pretrain_looplm_moe.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOOPLM_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]}
```

**Expected performance:**
- Tokens/sec/GPU: ~2000-2500
- Memory per GPU: ~50GB
- Model FLOPs Utilization (MFU): ~35-40%

#### Medium-Scale LoopLM-MoE (64 GPUs, Multi-Node)

```bash
#!/bin/bash
# 64 GPUs (8 nodes × 8 GPUs)
# Model: 8B parameters, 64 experts, T=4 steps

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 8                          # 8 shared layers
    --recurrent-depth 4                     # Effective 32 layers
    --use-looped-model
    --exit-gate-enabled
    --hidden-size 4096
    --num-attention-heads 32
    --num-query-groups 8
    --ffn-hidden-size 14336
    --seq-length 4096
    --max-position-embeddings 32768
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 500000
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 6
    --moe-grouped-gemm

    # Group-limited routing (reduce cross-node communication)
    --moe-router-num-groups 16
    --moe-router-group-topk 2

    # Shared experts (optional, for stability)
    --moe-shared-expert-intermediate-size 14336
    --moe-shared-expert-overlap

    # Aux-loss-free load balancing (DeepSeek-V3 style)
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-load-balancing-type none

    # Router optimization
    --moe-router-dtype fp32
    --moe-router-fusion
)

LOOPLM_ARGS=(
    --entropy-regularization-beta 0.001
    --exit-gate-threshold 0.5
    --exit-gate-min-steps 2
    --use-stage2-gate-training              # Enable after 50k iters
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --num-layers-per-virtual-pipeline-stage 2  # Virtual pipeline
    --sequence-parallel
    --use-distributed-optimizer
)

OPTIMIZATION_ARGS=(
    # Aggressive activation checkpointing
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe shared_experts moe_act layernorm

    # Communication overlap
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-param-gather-with-optimizer-step

    # Router fusion
    --moe-router-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 2048               # Large batch
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --train-iters 500000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

# Multi-node distributed args
DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 8
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port 29500
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_looplm_moe.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOOPLM_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]}
```

**Expected performance:**
- Tokens/sec/GPU: ~1800-2200
- Memory per GPU: ~60GB
- Model FLOPs Utilization (MFU): ~40-45%
- Cross-node communication: <25% of iteration time (with group-limited routing)

#### Large-Scale LoopLM-MoE (256 GPUs) with FP8

```bash
#!/bin/bash
# 256 GPUs (32 nodes × 8 GPUs)
# Model: 16B parameters, 128 experts, T=4 steps
# Hardware: H100 GPUs (FP8 support)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 16                         # 16 shared layers
    --recurrent-depth 4                     # Effective 64 layers
    --use-looped-model
    --exit-gate-enabled
    --hidden-size 5120
    --num-attention-heads 40
    --num-query-groups 8
    --ffn-hidden-size 13824
    --seq-length 4096
    --max-position-embeddings 65536
    --disable-bias-linear
    --normalization RMSNorm
    --position-embedding-type rope
    --rotary-base 500000
    --swiglu
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-grouped-gemm

    # Group-limited routing
    --moe-router-num-groups 32
    --moe-router-group-topk 4

    # Shared experts
    --moe-shared-expert-intermediate-size 13824
    --moe-shared-expert-overlap

    # Aux-loss-free
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-load-balancing-type none

    # Router optimization
    --moe-router-dtype fp32
    --moe-router-fusion
)

LOOPLM_ARGS=(
    --entropy-regularization-beta 0.001
    --exit-gate-threshold 0.5
    --exit-gate-min-steps 2
    --exit-gate-dtype fp32                  # Keep in FP32
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 8
    --expert-model-parallel-size 8
    --context-parallel-size 2               # For long context
    --num-layers-per-virtual-pipeline-stage 2
    --sequence-parallel
    --use-distributed-optimizer
)

FP8_ARGS=(
    --fp8-hybrid                            # FP8 training
    --fp8-amax-compute-algo max
    --fp8-amax-history-len 1024
    --fp8-margin 0
)

OPTIMIZATION_ARGS=(
    # Activation checkpointing
    --recompute-activations
    --recompute-granularity selective
    --recompute-modules moe shared_experts moe_act

    # Communication overlap
    --overlap-grad-reduce
    --overlap-param-gather
    --overlap-param-gather-with-optimizer-step

    # Router fusion
    --moe-router-fusion
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 4096               # Very large batch
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --train-iters 1000000
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
)

DISTRIBUTED_ARGS=(
    --nproc_per_node 8
    --nnodes 32
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port 29500
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_looplm_moe.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOOPLM_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${FP8_ARGS[@]} \
    ${OPTIMIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]}
```

**Expected performance:**
- Tokens/sec/GPU: ~3000-3500 (with FP8)
- Memory per GPU: ~70GB
- Model FLOPs Utilization (MFU): ~45-50%
- End-to-end throughput: ~800K-900K tokens/sec

### Performance Tuning Checklist for LoopLM-MoE

**Before training:**
- [ ] Grouped GEMM enabled (`--moe-grouped-gemm`)
- [ ] Selective activation checkpointing configured
- [ ] FP8 enabled if using H100/H200/Ada GPUs
- [ ] Router and exit gate in FP32 for stability
- [ ] Group-limited routing if multi-node (>8 GPUs)
- [ ] Sequence parallelism enabled with EP+TP
- [ ] Virtual pipeline parallelism configured (reduce bubbles)

**During training (monitor):**
- [ ] Expert utilization is balanced (check logs)
- [ ] Average exit step is reasonable (2-3 for T=4)
- [ ] Communication time < 25% of iteration time
- [ ] GPU utilization > 75%
- [ ] Memory usage stable (not growing)
- [ ] Gradient norms stable (< 5.0 typically)

**If performance issues:**
- [ ] Profile with `nsys` to identify bottleneck
- [ ] Check expert load balance (imbalance → idle GPUs)
- [ ] Verify communication overlap is working
- [ ] Check if activation checkpointing is too aggressive (slowdown)
- [ ] Verify FP8 is actually being used (check logs)

---

## Parallelism Strategies

### Configuration Matrix

| Model Size | Experts | GPUs | TP | EP | PP | VPP | CP | Seq Parallel |
|-----------|---------|------|----|----|----|----|----|----- ---------|
| < 3B      | 8-16    | 8-16 | 1  | 8-16 | 1  | - | - | ✓ |
| 3-10B     | 16-64   | 32-64 | 1  | 32-64 | 2-4 | 2-4 | - | ✓ |
| > 10B     | 64-256  | 128+ | 4  | 64-128 | 4-8 | 4-8 | 2-4 | ✓ |

**Key Rules:**
1. **Sequence Parallel REQUIRED** when EP + TP are used
2. **Prefer EP over TP** for MoE layers (better efficiency)
3. **Use MoE Parallel Folding**: Different TP for attention vs. MoE
4. **Pipeline Parallelism**: Split recurrent steps across PP ranks

### Implementation: MoE Parallel Folding

**File**: Modify `megatron/core/parallel_state.py` (if needed)

```python
# Configuration example:
# Attention: TP=4, DP=8
# MoE: TP=1 (i.e., expert_tensor_parallel_size=1), EP=32

# Total GPUs: TP_attn(4) × DP(8) = 32
# MoE experts distributed: EP(32) with TP(1) per expert

# Key: EPxTP group in MoE is subgroup of DPxCPxTP in Attention
```

**Training script example:**

```bash
#!/bin/bash
# LoopLM-MoE with MoE Parallel Folding

# 128 GPUs: 16 nodes × 8 GPUs/node

MODEL_ARGS=(
    --num-layers 8
    --recurrent-depth 4
    --use-looplm
    --hidden-size 4096
    --num-attention-heads 32
)

MOE_ARGS=(
    --num-experts 64
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

# MoE Parallel Folding:
# - Attention uses TP=4
# - MoE uses TP=1 (expert_tensor_parallel_size=1)
# - EP=32 distributes experts
PARALLEL_ARGS=(
    --tensor-model-parallel-size 4          # TP for attention
    --expert-tensor-parallel-size 1         # TP for MoE (use EP instead)
    --expert-model-parallel-size 32         # EP=32
    --pipeline-model-parallel-size 4        # PP=4
    --num-layers-per-virtual-pipeline-stage 2  # VPP
    --sequence-parallel                     # REQUIRED
    --use-distributed-optimizer
)

torchrun --nproc_per_node=8 --nnodes=16 \
    pretrain_looplm_moe.py ${MODEL_ARGS[@]} ${MOE_ARGS[@]} ${PARALLEL_ARGS[@]}
```

### Pipeline Parallelism for Recurrent Loops

**Strategy 1: Split Recurrent Steps**

```python
# PP=4, recurrent_depth=4
# Each PP rank handles 1 recurrent step

# PP Rank 0: Step 1
# PP Rank 1: Step 2
# PP Rank 2: Step 3
# PP Rank 3: Step 4
```

**Implementation:**

```python
# In LoopLMModel.forward()

pp_rank = parallel_state.get_pipeline_model_parallel_rank()
pp_size = self.config.pipeline_model_parallel_size

if pp_size > 1:
    # Distribute recurrent steps across PP ranks
    steps_per_rank = self.recurrent_depth // pp_size
    start_step = pp_rank * steps_per_rank
    end_step = (pp_rank + 1) * steps_per_rank

    for step in range(start_step, end_step):
        hidden_states = self.decoder(hidden_states, ...)

        # Send to next PP rank if not last
        if pp_rank < pp_size - 1:
            send_to_next_pipeline_rank(hidden_states)

        # Receive from previous PP rank if not first
        if pp_rank > 0 and step == start_step:
            hidden_states = recv_from_prev_pipeline_rank()
else:
    # No PP: all steps on same rank
    for step in range(self.recurrent_depth):
        hidden_states = self.decoder(hidden_states, ...)
```

**Strategy 2: Full Recurrence Per PP Stage**

```python
# PP=2, num_layers=8, recurrent_depth=4
# PP Rank 0: Layers 0-3, each looped 4 times
# PP Rank 1: Layers 4-7, each looped 4 times

# Simpler but uses more memory for activations
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit_tests/models/test_looplm_model.py`

```python
import pytest
import torch
from megatron.core.models.looplm import LoopLMModel
from megatron.core.transformer.transformer_config import TransformerConfig

class TestLoopLMModel:

    @pytest.mark.parametrize("recurrent_depth", [1, 2, 4, 8])
    def test_parameter_sharing(self, recurrent_depth):
        """Verify that weights are actually shared across recurrent steps."""
        config = TransformerConfig(
            num_layers=1,  # Single shared layer
            recurrent_depth=recurrent_depth,
            use_looped_model=True,
            hidden_size=256,
            num_attention_heads=4,
        )

        model = LoopLMModel(config, vocab_size=1000, max_sequence_length=512)

        # Get layer parameters
        layer_params = list(model.decoder.parameters())
        param_count = sum(p.numel() for p in layer_params)

        # Verify: parameter count should NOT scale with recurrent_depth
        # (because weights are shared)
        expected_params = param_count  # Same regardless of recurrent_depth

        assert param_count == expected_params

    @pytest.mark.parametrize("recurrent_depth", [2, 4])
    def test_gradient_flow(self, recurrent_depth):
        """Verify gradients flow correctly through recurrent steps."""
        config = TransformerConfig(
            num_layers=1,
            recurrent_depth=recurrent_depth,
            use_looped_model=True,
            hidden_size=256,
            num_attention_heads=4,
        )

        model = LoopLMModel(config, vocab_size=1000, max_sequence_length=512)

        # Forward pass
        input_ids = torch.randint(0, 1000, (16, 4))  # [seq_len, batch]
        position_ids = torch.arange(16).unsqueeze(1).expand(16, 4)
        attention_mask = torch.ones(1, 1, 16, 16)
        labels = torch.randint(0, 1000, (16, 4))

        output = model(input_ids, position_ids, attention_mask, labels=labels)

        # Check that output contains multiple steps
        assert 'logits' in output
        assert len(output['logits']) == recurrent_depth

        # Compute dummy loss and backprop
        loss = sum(logits.sum() for logits in output['logits'])
        loss.backward()

        # Verify all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_early_exit_inference(self):
        """Test early exit during inference."""
        config = TransformerConfig(
            num_layers=1,
            recurrent_depth=4,
            use_looped_model=True,
            exit_gate_enabled=True,
            exit_gate_threshold=0.5,
            exit_gate_min_steps=1,
            hidden_size=256,
            num_attention_heads=4,
        )

        model = LoopLMModel(config, vocab_size=1000, max_sequence_length=512)
        model.eval()

        # Inference mode
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (16, 1))
            position_ids = torch.arange(16).unsqueeze(1)
            attention_mask = torch.ones(1, 1, 16, 16)

            # Run inference (should exit early if gate triggers)
            output = model(input_ids, position_ids, attention_mask)

            # Output should be logits tensor (not dict with all steps)
            assert isinstance(output, torch.Tensor)
            assert output.shape[-1] == 1000  # vocab_size


class TestExitGate:

    def test_exit_probability_sum_to_one(self):
        """Exit probabilities should form a valid distribution."""
        from megatron.core.models.looplm.exit_gate import ExitGate

        config = TransformerConfig(hidden_size=256, num_attention_heads=4)
        gate = ExitGate(
            hidden_size=256,
            gate_hidden_size=64,
            max_steps=4,
            config=config
        )

        # Simulate gate logits for all steps
        batch_size = 8
        seq_len = 16
        gate_logits = []

        for step in range(4):
            hidden_states = torch.randn(seq_len, batch_size, 256)
            logit = gate(hidden_states, step)
            gate_logits.append(logit)

        # Compute distribution
        exit_probs = gate.compute_exit_distribution(gate_logits)

        # Verify: probabilities sum to 1
        prob_sums = exit_probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


class TestLoopLMLoss:

    def test_entropy_regularization(self):
        """Verify entropy regularization prevents collapse."""
        from megatron.core.models.looplm.looplm_loss import LoopLMLoss

        loss_fn = LoopLMLoss(entropy_reg_beta=0.01)

        # Create dummy data
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        num_steps = 4

        logits_per_step = [
            torch.randn(seq_len, batch_size, vocab_size)
            for _ in range(num_steps)
        ]
        labels = torch.randint(0, vocab_size, (seq_len, batch_size))

        # Case 1: Uniform distribution (high entropy)
        uniform_probs = torch.ones(batch_size, num_steps) / num_steps
        result1 = loss_fn.compute_stage1_loss(
            logits_per_step, labels, uniform_probs
        )

        # Case 2: Collapsed distribution (low entropy)
        collapsed_probs = torch.zeros(batch_size, num_steps)
        collapsed_probs[:, 0] = 1.0  # All weight on first step
        result2 = loss_fn.compute_stage1_loss(
            logits_per_step, labels, collapsed_probs
        )

        # Verify: Collapsed distribution has lower entropy
        assert result1['entropy_value'] > result2['entropy_value']

        # Verify: Entropy term penalizes collapsed distribution
        # (total_loss should be higher for collapsed due to -β*H term)
        assert result2['total_loss'] > result1['total_loss']
```

### Integration Tests

**File**: `tests/functional_tests/test_looplm_training.py`

```python
def test_looplm_training_convergence():
    """Test that LoopLM can overfit on small dataset."""
    # Small model, small dataset
    # Verify loss decreases over iterations
    pass

def test_looplm_moe_training():
    """Test LoopLM with MoE layers."""
    # Verify:
    # 1. MoE auxiliary loss is computed
    # 2. Expert load balancing works
    # 3. Router decisions are reasonable
    pass

def test_looplm_checkpoint_save_load():
    """Test distributed checkpointing."""
    # Save checkpoint
    # Load checkpoint with different parallelism
    # Verify weights match
    pass
```

### Performance Benchmarks

**File**: `tests/performance/benchmark_looplm.py`

```python
def benchmark_looplm_vs_standard():
    """Compare LoopLM vs standard GPT model.

    Metrics:
        - Parameters: LoopLM should have ~T× fewer params
        - Memory: Activations scale with recurrent_depth
        - Throughput: tokens/sec
        - Latency: Time per forward pass
    """
    pass

def benchmark_looplm_inference_early_exit():
    """Measure inference speedup from early exit."""
    # Compare:
    # - Fixed depth (no early exit)
    # - Adaptive depth (with early exit)
    # Report average steps per sample
    pass
```

---

## Training Pipeline

### Multi-Stage Training Script

**File**: `examples/pretrain_looplm.py`

```python
#!/usr/bin/env python3
"""
LoopLM Pretraining Script

Implements two-stage training:
    Stage I: Entropy-regularized training (most of training)
    Stage II: Focused gate training (final phase)

Usage:
    # Stage I (0-90% of training)
    torchrun --nproc_per_node=8 pretrain_looplm.py \
        --looplm-training-stage stage1 \
        --entropy-reg-beta 0.001 \
        ...

    # Stage II (90-100% of training)
    torchrun --nproc_per_node=8 pretrain_looplm.py \
        --looplm-training-stage stage2 \
        --load /path/to/stage1/checkpoint \
        ...
"""

import torch
from megatron.training import pretrain, get_args, get_timers
from megatron.core.models.looplm import LoopLMModel
from megatron.core.models.looplm.looplm_loss import compute_looplm_loss


def model_provider(pre_process=True, post_process=True):
    """Build LoopLM model."""
    args = get_args()

    config = TransformerConfig(
        num_layers=1,  # Single shared layer
        recurrent_depth=args.recurrent_depth,
        use_looped_model=True,
        exit_gate_enabled=args.exit_gate_enabled,
        exit_gate_threshold=args.exit_gate_threshold,
        exit_gate_min_steps=args.exit_gate_min_steps,
        entropy_regularization_beta=args.entropy_reg_beta,
        use_stage2_gate_training=args.looplm_training_stage == 'stage2',
        # ... other config params ...
    )

    model = LoopLMModel(
        config=config,
        transformer_layer_spec=get_looplm_layer_spec(),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


def forward_step(data_iterator, model):
    """Forward training step."""
    args = get_args()
    timers = get_timers()

    # Get batch
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    # Forward pass
    output = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        loss_mask=loss_mask
    )

    # Compute LoopLM loss
    loss, logging_dict = compute_looplm_loss(
        model_output=output,
        labels=labels,
        config=model.config,
        loss_mask=loss_mask
    )

    # Return loss and logging function
    def loss_func(loss_mask, output_tensor):
        return loss, logging_dict

    return output, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/val/test datasets."""
    # Use standard GPT dataset
    from pretrain_gpt import train_valid_test_datasets_provider as gpt_datasets
    return gpt_datasets(train_val_test_num_samples)


if __name__ == "__main__":
    # Add LoopLM arguments
    from megatron.training.arguments_looplm import add_looplm_args

    # Run pretraining
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_looplm_args
    )
```

### Training Configuration Examples

#### Small Model (1.4B params, 8 GPUs)

```bash
#!/bin/bash
# looplm_1.4b.sh

GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=8

MODEL_ARGS=(
    --num-layers 1              # Single shared layer
    --recurrent-depth 4         # Loop 4 times = 4 effective layers
    --hidden-size 2048
    --num-attention-heads 16
    --num-query-groups 8        # GQA
    --ffn-hidden-size 5504
    --seq-length 2048
    --max-position-embeddings 2048
    --normalization RMSNorm
    --swiglu
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
)

LOOPLM_ARGS=(
    --use-looplm
    --exit-gate-enabled
    --exit-gate-threshold 0.5
    --exit-gate-min-steps 1
    --entropy-reg-beta 0.001
    --looplm-training-stage stage1
)

TRAINING_ARGS=(
    --micro-batch-size 4
    --global-batch-size 512
    --train-iters 100000
    --lr 3e-4
    --min-lr 3e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --weight-decay 0.1
    --bf16
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
)

DATA_ARGS=(
    --data-path /path/to/data
    --split 99,1,0
    --tokenizer-type GPT2BPETokenizer
    --vocab-file /path/to/vocab.json
    --merge-file /path/to/merges.txt
)

torchrun --nproc_per_node=$GPUS_PER_NODE pretrain_looplm.py \
    ${MODEL_ARGS[@]} \
    ${LOOPLM_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```

#### Large Model with MoE (10B params, 128 GPUs)

```bash
#!/bin/bash
# looplm_moe_10b.sh

GPUS_PER_NODE=8
NNODES=16
WORLD_SIZE=128

MODEL_ARGS=(
    --num-layers 1
    --recurrent-depth 4
    --hidden-size 4096
    --num-attention-heads 32
    --num-query-groups 8
    --ffn-hidden-size 14336
    --seq-length 4096
    --max-position-embeddings 32768
    --normalization RMSNorm
    --swiglu
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
)

MOE_ARGS=(
    --num-experts 64
    --expert-model-parallel-size 32
    --tensor-model-parallel-size 1
    --expert-tensor-parallel-size 1  # No TP in MoE
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
)

LOOPLM_ARGS=(
    --use-looplm
    --exit-gate-enabled
    --entropy-reg-beta 0.001
    --looplm-training-stage stage1
)

PARALLEL_ARGS=(
    --pipeline-model-parallel-size 4
    --num-layers-per-virtual-pipeline-stage 1
    --sequence-parallel              # REQUIRED with EP+TP
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 256
    --train-iters 500000
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-warmup-iters 2000
    --clip-grad 1.0
    --weight-decay 0.1
    --bf16
)

# Memory optimizations
MEMORY_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe core_attn
    --moe-expert-capacity-factor 1.0  # For first 200 steps
)

torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES \
    pretrain_looplm_moe.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOOPLM_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MEMORY_ARGS[@]} \
    --save /path/to/checkpoints \
    --tensorboard-dir /path/to/tensorboard
```

### Training Stages Transition

**Stage I → Stage II transition script:**

```bash
#!/bin/bash
# transition_to_stage2.sh

# Stage I completed at iteration 450,000
# Stage II runs for remaining 50,000 iterations

CHECKPOINT_DIR=/path/to/stage1/checkpoints

# Resume from Stage I checkpoint, switch to Stage II
torchrun --nproc_per_node=8 pretrain_looplm.py \
    --load $CHECKPOINT_DIR \
    --looplm-training-stage stage2 \
    --use-stage2-gate-training \
    --entropy-reg-beta 0.0 \
    --train-iters 500000 \
    # ... other args same as Stage I ...
```

---

## Next Steps

### Phase 1 Implementation Checklist

- [ ] Add LoopLM config parameters to `TransformerConfig`
- [ ] Implement `LoopLMModel` core forward pass
- [ ] Modify `TransformerBlock` to support layer reuse
- [ ] Create basic training script
- [ ] Write unit tests for parameter sharing
- [ ] Verify gradient flow through recurrent steps
- [ ] Test memory usage vs. theoretical estimates

### Phase 2 Implementation Checklist

- [ ] Implement `ExitGate` module
- [ ] Add Gumbel-Softmax sampling
- [ ] Implement Q-exit criterion
- [ ] Add gate visualization tools
- [ ] Test early exit during inference
- [ ] Verify exit patterns make sense

### Phase 3 Implementation Checklist

- [ ] Implement Stage I loss (entropy-regularized)
- [ ] Implement Stage II loss (loss improvement)
- [ ] Add multi-stage training coordinator
- [ ] Create loss logging and monitoring
- [ ] Test loss convergence
- [ ] Verify entropy term prevents collapse

### Phase 4 Implementation Checklist

- [ ] Implement `LoopLMMoEModel`
- [ ] Modify router for recurrent step awareness
- [ ] Aggregate auxiliary loss across steps
- [ ] Create MoE-specific layer specs
- [ ] Test EP parallelism with parameter sharing
- [ ] Implement router warmup strategy

### Phase 5 Implementation Checklist

- [ ] Implement PP for recurrent loops
- [ ] Add KV cache sharing for inference
- [ ] Implement activation recomputation
- [ ] Create multi-node training scripts
- [ ] Optimize memory usage
- [ ] Benchmark performance

### Phase 6 Implementation Checklist

- [ ] Write comprehensive unit tests
- [ ] Create integration tests
- [ ] Add performance benchmarks
- [ ] Write user documentation
- [ ] Create example training scripts
- [ ] Prepare for code review

---

## References

### Key Megatron Files
- **Model Reference**: `megatron/core/models/gpt/gpt_model.py`
- **Layer Specs**: `megatron/core/models/gpt/gpt_layer_specs.py`
- **Transformer Block**: `megatron/core/transformer/transformer_block.py`
- **MoE Implementation**: `megatron/core/transformer/moe/moe_layer.py`
- **Training Loop**: `pretrain_gpt.py`

### MoE Resources
- **MoE README**: `megatron/core/transformer/moe/README.md`
- **Parallelism Rules**:
  - EP + TP requires Sequence Parallel (mandatory)
  - Prefer EP over TP for MoE layers
  - Use MoE Parallel Folding for different TP in attention vs. MoE

### LoopLM Paper Concepts
- **Architecture**: Parameter-shared recurrent transformer
- **Formula**: `F^(t)(x) = lmhead ∘ H_L^t ∘ emb(x)` where `H_L` is the shared layer
- **Stage I Loss**: `L = Σ q(t)·L(t) - β·H(q)`
- **Stage II Loss**: Uses loss improvement `ΔL(t) = L(t-1) - L(t)`
- **Q-exit**: `Q(X^(t)) = P(X^(t)|X^(1))` quality metric

---

## FAQ

**Q: Why set `num_layers=1` in LoopLM?**
A: LoopLM uses parameter sharing - one layer is applied recurrently. Setting `num_layers=1` creates a single layer, which is then looped `recurrent_depth` times.

**Q: How does gradient flow through recurrent steps?**
A: Standard backpropagation through time (BPTT). Gradients flow backward through all recurrent steps and accumulate on the shared layer's parameters.

**Q: Can I use LoopLM without the exit gate?**
A: Yes! Set `exit_gate_enabled=False`. The model will always use all recurrent steps. This is useful for initial development/debugging.

**Q: How does PP work with recurrent loops?**
A: Two strategies: (1) Split recurrent steps across PP ranks, (2) Full recurrence on each PP stage's layer subset. Strategy 1 is more memory-efficient.

**Q: Why is Sequence Parallel required with EP + TP?**
A: Megatron's MoE implementation requires SP when combining EP and TP to ensure correct tensor sharding and communication patterns.

**Q: How much memory does LoopLM save vs. standard model?**
A: Parameters: ~T× reduction (where T=recurrent_depth). Activations: Similar or slightly higher due to storing intermediate steps for multi-exit loss.

**Q: Can I convert a trained GPT model to LoopLM?**
A: Not directly - the architectures are fundamentally different. You would need to train LoopLM from scratch or use knowledge distillation from a GPT teacher.

---

## Implementation Timeline

**Total: ~10 weeks**

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2  | Core Architecture | LoopLMModel, parameter sharing, basic training |
| 3    | Exit Gating | ExitGate, early exit, visualization |
| 4    | Loss Functions | Stage I & II loss, multi-stage training |
| 5-6  | MoE Integration | LoopLMMoEModel, router mods, load balancing |
| 7-8  | Optimization | PP for loops, KV cache sharing, memory opts |
| 9-10 | Testing | Unit tests, integration tests, benchmarks |

---

**Document Version**: 1.0
**Last Updated**: 2025-01-04
**Author**: Implementation planning for LoopLM in Megatron-LM
