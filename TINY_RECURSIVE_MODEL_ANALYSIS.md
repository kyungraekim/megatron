# TinyRecursiveModel Implementation Analysis for Megatron-LM

## Executive Summary

The **TinyRecursiveModel** is a research architecture that performs iterative refinement through recursive depth-wise processing with adaptive halting. This analysis examines its feasibility and implementation path within Megatron-LM's framework.

**Can it be implemented?** ✅ **YES** - All core components can be adapted to Megatron's architecture

**Implementation Complexity:** ⚠️ **MEDIUM-HIGH** - Requires custom training loop and novel architectural patterns

---

## 1. Model Architecture Overview

### Core Concept
TinyRecursiveModel implements **iterative refinement** where:
- A single network is applied recursively to refine outputs
- Latent states are maintained and refined multiple times before each output update
- Adaptive halting allows the model to "stop thinking" when confident

### Key Components

```python
class TinyRecursiveModel:
    # Parameters
    - dim: Hidden dimension
    - num_tokens: Vocabulary size
    - network: Module (can be MLPMixer1D, Transformer, etc.)
    - num_refinement_blocks: T (depth of recursion)
    - num_latent_refinements: n (latents refined n times per output refinement)
    - halt_loss_weight: Weight for early stopping loss
    - num_register_tokens: Optional register tokens
```

### Refinement Process

```
For each refinement block t ∈ [1, T]:
    For i ∈ [1, n]:
        latents = network(outputs + latents + inputs)  # Latent refinement
    outputs = network(outputs + latents)               # Output refinement
```

### Key Innovation: Recursive Depth
- **NOT** like standard Transformers with fixed depth
- **Single network** applied T times (parameter sharing across "depth")
- Gradient flows only through **last** refinement block (controlled by `torch.no_grad`)

---

## 2. Mapping to Megatron Components

### 2.1 Core Network → TransformerBlock

**TinyRecursiveModel.network** can be:

```python
# Option 1: Use Megatron's TransformerBlock directly
network = TransformerBlock(
    config=config,
    spec=get_gpt_layer_local_spec(),
    post_layer_norm=True,
    pre_process=False,  # No embedding
    post_process=False   # No output projection
)

# Option 2: MLPMixer1D (custom implementation)
network = MLPMixer1D(...)
```

**Key Consideration:** The network must be **reusable** (stateless except for parameters).

### 2.2 Embedding Layer → LanguageModelEmbedding

```python
# TinyRecursiveModel has:
self.input_embed = nn.Embedding(num_tokens, dim)

# Maps to:
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding
)

self.embedding = LanguageModelEmbedding(
    config=config,
    vocab_size=num_tokens,
    max_sequence_length=max_seq_len,
    position_embedding_type='none',  # Or 'rope' if needed
)
```

### 2.3 Output Projection → ColumnParallelLinear

```python
# TinyRecursiveModel has:
self.to_pred = nn.Linear(dim, num_tokens, bias=False)

# Maps to:
from megatron.core import tensor_parallel

self.output_layer = tensor_parallel.ColumnParallelLinear(
    config.hidden_size,
    vocab_size,
    config=config,
    init_method=config.init_method,
    bias=False,
    gather_output=not parallel_output,
    tp_group=pg_collection.tp,
)
```

### 2.4 Halt Prediction Head

```python
# Current implementation:
self.to_halt_pred = nn.Sequential(
    Reduce('b n d -> b d', 'mean'),
    nn.Linear(dim, 1, bias=False),
    nn.Sigmoid(),
    Rearrange('... 1 -> ...')
)

# Megatron-compatible:
self.halt_predictor = tensor_parallel.ColumnParallelLinear(
    config.hidden_size,
    1,
    config=config,
    bias=False,
    gather_output=True,  # Need full prediction for halting logic
    tp_group=pg_collection.tp,
)
# + Reduce/Sigmoid/Rearrange in forward()
```

---

## 3. Implementation Challenges

### 3.1 ⚠️ **CRITICAL: Non-Standard Training Loop**

**Problem:** Megatron's standard training assumes:
```python
# Standard: One forward pass per sample
hidden_states = model(input_ids, ...)
loss = compute_loss(hidden_states, labels)
```

**TinyRecursiveModel requires:**
```python
# Initialize state
outputs, latents = model.get_initial()

# RECURSIVE refinement over T blocks
for t in range(1, T+1):
    context = torch.no_grad() if t < T else nullcontext()
    with context():
        outputs, latents = model.refine_latent_then_output_once(
            inputs, outputs, latents
        )

# Compute loss ONLY on final refinement
loss = compute_loss(outputs, labels)
```

**Solution:** Custom training loop or modify `forward()` to encapsulate recursion.

### 3.2 Gradient Flow Management

**Key Design:**
```python
# Gradients ONLY flow through last block
for step in range_from_one(num_refinement_blocks):
    is_last = step == num_refinement_blocks
    context = torch.no_grad if not is_last else nullcontext

    with context():
        outputs, latents = self.refine_latent_then_output_once(...)
```

**Implication:**
- Memory efficient (only last block's activations saved)
- But: Can't use standard activation checkpointing
- Megatron's `recompute_granularity` won't work out-of-the-box

### 3.3 Persistent State Management

**Outputs and Latents must persist across refinements:**

```python
# Option A: Keep in forward() scope (simpler)
def forward(self, input_ids, ...):
    outputs, latents = self.get_initial()
    # Expand to batch
    outputs = repeat(outputs, 'd -> b n d', b=batch, n=seq_len)
    latents = repeat(latents, 'd -> b n d', b=batch, n=seq_len)

    # Recursive refinement
    for t in range(T):
        outputs, latents = self.refine(...)

    return self.output_layer(outputs)

# Option B: External state management (for inference)
# Store outputs/latents as model attributes or pass explicitly
```

**Megatron Consideration:**
- With Pipeline Parallelism, state must be passed between stages
- Use `set_input_tensor()` pattern for PP

### 3.4 Register Tokens (Optional Feature)

```python
# Original:
registers = repeat(self.register_tokens, 'n d -> b n d', b=batch)
inputs, packed_shape = pack([registers, inputs], 'b * d')

# After processing:
registers, outputs = unpack(outputs, packed_shape, 'b * d')
```

**Megatron Integration:**
- Similar to prefix tokens in multimodal models
- Can concatenate to sequence dimension before TransformerBlock
- Need to handle in attention masks

### 3.5 Adaptive Halting During Inference

**Current Implementation:**
```python
@torch.no_grad()
def predict(self, seq, halt_prob_thres=0.5, max_steps=12):
    for step in range(1, max_steps+1):
        outputs, latents = self.deep_refinement(...)
        halt_prob = self.to_halt_pred(outputs)
        should_halt = (halt_prob >= halt_prob_thres) | is_last

        # Early exit for samples that halt
        if should_halt.any():
            # Remove halted samples from batch
            inputs = inputs[~should_halt]
            outputs = outputs[~should_halt]
            latents = latents[~should_halt]
```

**Challenge for Distributed Inference:**
- Different samples halt at different steps
- With Data Parallelism: Each rank has different batch sizes dynamically
- With Tensor/Pipeline Parallelism: Need synchronized halting decisions

**Solution:**
- Simplified: Run all samples for max_steps (no early exit)
- Advanced: Implement dynamic batching with padding/masking

---

## 4. Proposed Implementation Strategy

### Phase 1: Single-GPU Prototype

**Goal:** Validate the architecture works in Megatron

```python
class TinyRecursiveGPTModel(LanguageModule):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        num_refinement_blocks: int = 3,
        num_latent_refinements: int = 6,
        halt_loss_weight: float = 1.0,
    ):
        super().__init__(config=config)

        # Embedding
        self.embedding = LanguageModelEmbedding(...)

        # Single reusable TransformerBlock
        self.network = TransformerBlock(
            config=config,
            spec=transformer_layer_spec,
            post_layer_norm=True,
            pre_process=False,
            post_process=False,
        )

        # Initial state embeddings
        self.output_init_embed = nn.Parameter(torch.randn(config.hidden_size) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(config.hidden_size) * 1e-2)

        # Output projection
        self.output_layer = tensor_parallel.ColumnParallelLinear(...)

        # Halt predictor
        self.halt_predictor = tensor_parallel.ColumnParallelLinear(...)

        self.num_refinement_blocks = num_refinement_blocks
        self.num_latent_refinements = num_latent_refinements
        self.halt_loss_weight = halt_loss_weight

    def get_initial_state(self, batch_size, seq_len):
        outputs = repeat(self.output_init_embed, 'd -> b n d',
                        b=batch_size, n=seq_len)
        latents = repeat(self.latent_init_embed, 'd -> b n d',
                        b=batch_size, n=seq_len)
        return outputs, latents

    def refine_latent_then_output_once(
        self,
        inputs,
        outputs,
        latents,
        attention_mask,
    ):
        # Latent refinement (n times)
        for _ in range(self.num_latent_refinements):
            latents = self.network(
                hidden_states=outputs + latents + inputs,
                attention_mask=attention_mask,
            )

        # Output refinement (once)
        outputs = self.network(
            hidden_states=outputs + latents,
            attention_mask=attention_mask,
        )

        return outputs, latents

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels=None,
    ):
        batch_size, seq_len = input_ids.shape

        # Embed inputs
        inputs = self.embedding(input_ids, position_ids)

        # Initialize states
        outputs, latents = self.get_initial_state(batch_size, seq_len)

        # Deep refinement with gradient control
        for step in range(1, self.num_refinement_blocks + 1):
            is_last = step == self.num_refinement_blocks
            context = nullcontext() if is_last else torch.no_grad()

            with context():
                outputs, latents = self.refine_latent_then_output_once(
                    inputs, outputs, latents, attention_mask
                )

        # Predictions
        logits = self.output_layer(outputs)

        # Halt probability
        halt_prob = self.compute_halt_prob(outputs)

        if labels is None:
            return logits, halt_prob

        # Compute losses
        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n v -> b v n'),
            labels,
            reduction='none'
        )
        ce_loss = reduce(ce_loss, 'b ... -> b', 'mean')

        is_all_correct = (logits.argmax(dim=-1) == labels).all(dim=-1)
        halt_loss = F.binary_cross_entropy(
            halt_prob,
            is_all_correct.float(),
            reduction='none'
        )

        total_loss = ce_loss + halt_loss * self.halt_loss_weight

        return total_loss.sum(), (ce_loss, halt_loss), logits, halt_prob
```

### Phase 2: Distributed Training Support

**Add Tensor Parallelism:**

```python
# Key changes:
# 1. All linear layers use ColumnParallelLinear/RowParallelLinear
# 2. Handle sequence parallel for inputs/outputs/latents
# 3. Gather halt_prob across TP for halting decision

def compute_halt_prob(self, hidden_states):
    # Average across sequence (reduce along seq dim)
    pooled = hidden_states.mean(dim=1)  # [b, d]

    # TP-parallel linear layer
    halt_logit = self.halt_predictor(pooled)  # [b, 1]

    # Gather across TP if needed
    if self.config.tensor_model_parallel_size > 1:
        halt_logit = tensor_parallel.gather_from_tensor_model_parallel_region(
            halt_logit
        )

    return torch.sigmoid(halt_logit).squeeze(-1)
```

**Add Pipeline Parallelism:**

```python
# Challenge: State (outputs, latents) must pass between PP stages
#
# Option 1: Keep entire model on single PP stage
#   - Refinement loop doesn't span PP boundaries
#   - State stays local
#
# Option 2: Custom PP schedule
#   - Pass (outputs, latents) as auxiliary tensors
#   - Modify pipeline_parallel schedules to handle state
```

**Recommendation:** Start with PP=1 (no pipeline parallelism) for simplicity.

### Phase 3: Inference Optimization

```python
@torch.no_grad()
def generate(
    self,
    input_ids,
    position_ids,
    attention_mask,
    halt_prob_thres=0.5,
    max_refinement_steps=12,
):
    batch_size, seq_len = input_ids.shape

    inputs = self.embedding(input_ids, position_ids)
    outputs, latents = self.get_initial_state(batch_size, seq_len)

    # Track active samples
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=inputs.device)
    final_logits = torch.zeros(
        batch_size, seq_len, self.vocab_size,
        device=inputs.device
    )
    exit_steps = torch.zeros(batch_size, dtype=torch.long, device=inputs.device)

    for step in range(1, max_refinement_steps + 1):
        # Refine only active samples
        active_inputs = inputs[active_mask]
        active_outputs = outputs[active_mask]
        active_latents = latents[active_mask]
        active_attention_mask = attention_mask[active_mask]

        # Deep refinement
        for _ in range(self.num_refinement_blocks):
            active_outputs, active_latents = self.refine_latent_then_output_once(
                active_inputs, active_outputs, active_latents,
                active_attention_mask
            )

        # Compute halt probability
        halt_prob = self.compute_halt_prob(active_outputs)
        should_halt = (halt_prob >= halt_prob_thres) | (step == max_refinement_steps)

        # Store results for halted samples
        halted_indices = torch.where(active_mask)[0][should_halt]
        final_logits[halted_indices] = self.output_layer(
            active_outputs[should_halt]
        )
        exit_steps[halted_indices] = step

        # Update active mask
        active_indices = torch.where(active_mask)[0]
        active_mask[active_indices[should_halt]] = False

        if not active_mask.any():
            break

        # Update state for next iteration
        outputs[active_mask] = active_outputs[~should_halt]
        latents[active_mask] = active_latents[~should_halt]

    return final_logits.argmax(dim=-1), exit_steps
```

---

## 5. Key Architectural Decisions

### Decision 1: Shared vs Separate Networks

**Option A: Single Shared Network (Original)**
```python
self.network = TransformerBlock(...)  # One network

# Used recursively:
for t in range(T):
    latents = self.network(...)
    outputs = self.network(...)
```

✅ **Pros:**
- Fewer parameters
- True recursive depth
- Memory efficient

❌ **Cons:**
- Less expressive
- All refinements use same weights

**Option B: Separate Networks per Refinement**
```python
self.latent_networks = nn.ModuleList([
    TransformerBlock(...) for _ in range(T)
])
self.output_networks = nn.ModuleList([
    TransformerBlock(...) for _ in range(T)
])
```

✅ **Pros:**
- More expressive
- Each refinement can specialize
- Standard training (no gradient blocking)

❌ **Cons:**
- T times more parameters
- Loses recursive property

**Recommendation:** Start with Option A (shared) to stay true to the paper's intent.

### Decision 2: Gradient Flow Strategy

**Current (Paper):**
- Only last refinement block gets gradients
- Earlier blocks use `torch.no_grad()`

**Alternative:**
- Gradient through all blocks
- Use smaller learning rates for earlier blocks
- Standard activation checkpointing

**Recommendation:** Follow paper's approach initially, can experiment with alternatives.

### Decision 3: Refinement Schedule

**Fixed Schedule:**
```python
num_refinement_blocks = 3       # T
num_latent_refinements = 6      # n
```

**Adaptive Schedule:**
```python
# Learn when to stop refining
for t in range(max_blocks):
    should_continue = learned_gate(outputs)
    if not should_continue:
        break
    outputs, latents = refine(...)
```

**Recommendation:** Start with fixed, add adaptive later.

---

## 6. Integration Points with Megatron

### 6.1 Model Registration

```python
# In megatron/core/models/tiny_recursive/
├── __init__.py
├── tiny_recursive_gpt_model.py
├── recursive_spec_provider.py
└── README.md
```

### 6.2 Config Extension

```python
@dataclass
class RecursiveTransformerConfig(TransformerConfig):
    """Config for recursive refinement models."""

    num_refinement_blocks: int = 3
    """Number of deep refinement iterations (T in paper)."""

    num_latent_refinements: int = 6
    """Number of latent refinements per output refinement (n in paper)."""

    halt_loss_weight: float = 1.0
    """Weight for adaptive halting loss."""

    use_register_tokens: bool = False
    """Whether to use register tokens."""

    num_register_tokens: int = 0
    """Number of register tokens to prepend."""

    max_inference_refinements: int = 12
    """Maximum refinement steps during inference."""

    halt_threshold: float = 0.5
    """Halting probability threshold for early exit."""
```

### 6.3 Training Script

```python
# pretrain_recursive_gpt.py

def model_provider(pre_process=True, post_process=True):
    config = RecursiveTransformerConfig(
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        num_refinement_blocks=3,
        num_latent_refinements=6,
        halt_loss_weight=1.0,
    )

    model = TinyRecursiveGPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=50257,
        max_sequence_length=1024,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model

def forward_step(data_iterator, model):
    data = next(data_iterator)
    tokens = data['tokens']
    labels = data['labels']
    attention_mask = data['attention_mask']
    position_ids = data['position_ids']

    # Custom forward for recursive model
    total_loss, losses, logits, halt_prob = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    ce_loss, halt_loss = losses

    # Return loss and detailed metrics
    return total_loss, {
        'lm_loss': ce_loss.mean(),
        'halt_loss': halt_loss.mean(),
        'total_loss': total_loss,
    }
```

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/unit_tests/models/test_tiny_recursive.py

def test_recursive_refinement():
    """Test that recursive refinement works correctly."""
    config = RecursiveTransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_refinement_blocks=3,
        num_latent_refinements=2,
    )

    model = TinyRecursiveGPTModel(config, ...)

    input_ids = torch.randint(0, 1000, (2, 10))
    position_ids = torch.arange(10).unsqueeze(0).expand(2, 10)
    attention_mask = torch.ones(2, 1, 10, 10)
    labels = torch.randint(0, 1000, (2, 10))

    loss, losses, logits, halt_prob = model(
        input_ids, position_ids, attention_mask, labels
    )

    assert loss.shape == ()
    assert logits.shape == (2, 10, 1000)
    assert halt_prob.shape == (2,)

def test_gradient_flow():
    """Test that gradients only flow through last block."""
    # Implementation to verify gradient blocking works
    pass

def test_adaptive_halting():
    """Test inference with early stopping."""
    pass
```

### Integration Tests

```bash
# Run single-GPU training
pytest tests/unit_tests/models/test_tiny_recursive.py -v

# Run distributed training (TP=2)
torchrun --nproc_per_node=2 tests/functional_tests/test_tiny_recursive_tp.py
```

---

## 8. Performance Considerations

### Memory Usage

**Standard Transformer (depth D):**
- Parameters: `D * params_per_layer`
- Activations: `D * hidden_size * batch * seq_len` (with gradient checkpointing)

**TinyRecursiveModel (depth T with parameter sharing):**
- Parameters: `1 * params_per_layer` (shared network)
- Activations: `1 * hidden_size * batch * seq_len` (only last block)

**Memory Savings:** ~T times fewer activations stored!

### Compute Cost

**Per Training Step:**
```
Standard: D forward passes through unique layers
Recursive: T * (n + 1) forward passes through same layer
```

If T=3, n=6:
- 3 * 7 = 21 passes through the shared network
- vs 12 passes through unique layers (if D=12)

**Trade-off:** More compute, fewer parameters, less memory.

### Throughput Optimization

1. **Fuse Refinement Loops:** Reduce Python overhead
2. **Use Compiled Mode:** `torch.compile()` on refinement function
3. **Batch Halting:** Don't halt samples individually during training

---

## 9. Recommended Implementation Path

### ✅ **Step 1: Standalone Prototype** (1-2 weeks)
- Implement `TinyRecursiveGPTModel` without distributed features
- Single GPU training on small dataset
- Verify loss convergence and halting behavior

### ✅ **Step 2: Megatron Integration** (2-3 weeks)
- Add to `megatron/core/models/tiny_recursive/`
- Integrate with Megatron's embedding and output layers
- Create `RecursiveTransformerConfig`
- Add training script `pretrain_recursive_gpt.py`

### ✅ **Step 3: Tensor Parallelism** (1-2 weeks)
- Make all linear layers TP-compatible
- Handle sequence parallel for state tensors
- Test with TP=2, TP=4

### ⚠️ **Step 4: Pipeline Parallelism** (2-3 weeks) - **OPTIONAL**
- Design state passing mechanism
- Modify schedules or keep model on single PP stage
- Test with PP=2

### ✅ **Step 5: Inference Optimization** (1 week)
- Implement adaptive halting for inference
- Add KV caching if using attention
- Optimize dynamic batching

### ✅ **Step 6: Testing & Documentation** (1 week)
- Comprehensive unit tests
- Functional tests with different parallelism configs
- Performance benchmarking
- User documentation

**Total Estimated Time:** 8-12 weeks for full implementation

---

## 10. Open Questions & Research Directions

### Question 1: Optimal Refinement Schedule
- How many refinement blocks (T)?
- How many latent refinements per output refinement (n)?
- Should this be learned or fixed?

### Question 2: Scaling Laws
- How does recursive depth trade off with model width?
- Can we use smaller hidden sizes with more refinements?

### Question 3: Alternative Network Architectures
- Does this work better with MLPMixer vs Transformers?
- Can we use MoE for different refinement "experts"?

### Question 4: Curriculum Learning
- Start with fewer refinements, increase during training?
- Gradual unfreezing of earlier refinement blocks?

### Question 5: Multi-Task Learning
- Can same recursive network handle different tasks?
- Task-specific initial states?

---

## 11. Conclusion

### ✅ **Feasibility: HIGH**
All components can be implemented in Megatron-LM:
- Transformer as recursive network ✅
- State management ✅
- Gradient control ✅
- Distributed training (TP) ✅
- Adaptive halting ✅

### ⚠️ **Challenges:**
1. Non-standard training loop
2. Gradient flow management
3. State persistence across refinements
4. Dynamic batching for inference
5. Pipeline parallelism integration

### 🎯 **Recommended Approach:**
Start with **single-GPU prototype**, then add **TP support**, skip **PP** initially.

### 📊 **Expected Benefits:**
- Parameter efficiency (shared network)
- Memory efficiency (gradient blocking)
- Adaptive computation (halting)
- Novel architecture for research

### 🔬 **Research Value:**
This is a **novel architecture** that explores:
- Recursive depth vs fixed depth
- Iterative refinement
- Adaptive computation
- Parameter sharing at scale

**Implementation in Megatron would enable:**
- Large-scale experiments (billions of parameters)
- Distributed training across many GPUs
- Comparison with standard Transformers
- New insights into recursive processing

---

## 12. References & Resources

### Papers
- Original TinyRecursiveModel paper (if available)
- Universal Transformers (similar recursive concept)
- Adaptive Computation Time (ACT) for halting

### Megatron-LM Resources
- Architecture: `CLAUDE.md` in repo
- TransformerBlock: `megatron/core/transformer/transformer_block.py`
- GPTModel: `megatron/core/models/gpt/gpt_model.py`
- Parallelism: `megatron/core/parallel_state.py`

### Related Work
- PonderNet: Learning to think with adaptive computation
- Universal Transformers: Recursive self-attention
- Confident Adaptive Language Modeling (CALM): Adaptive depth

---

**Document Version:** 1.0
**Last Updated:** October 14, 2025
**Status:** ✅ Analysis Complete