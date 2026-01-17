# CLAUDE.md - rl

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Reinforcement learning: GRPO, PPO, rollout generation, reward modeling
> **Parent**: [../CLAUDE.md](../../CLAUDE.md)

---

## Overview

`megatron/rl` provides a native reinforcement learning post-training framework for Megatron-LM.
It implements policy optimization algorithms (GRPO, PPO) and supports custom RL environments
with flexible agent/inference interfaces.

**Key Design**: Agents (environments) are decoupled from RL training logic. Agents take an
`InferenceInterface` handle and return rollouts/rewards. The framework handles batching,
inference orchestration, and training.

---

## File Map

| File | Size | Purpose |
|------|------|---------|
| `rl_utils.py` | 73KB | GRPO/PPO training core, batch rollout generation, loss computation |
| `sequence_packing_utils.py` | 45KB | Efficient sequence packing, bin allocation, micro-batching |
| `parallel_utils.py` | 4KB | Distributed training utilities, all-reduce, group sync |
| `logging.py` | <1KB | Structured logging setup |
| `__init__.py` | 2.5KB | Module exports, Request/Response types |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `agent/` | Agent framework (API, reward-only agent, multi-task agent, HF dataset agent) |
| `inference/` | Inference interface definitions and implementations |
| `server/` | FastAPI server infrastructure for distributed agent/inference services |

---

## Architecture & Design

### Agent/Environment Pattern

Agents implement a generate-evaluate loop:
```
Agent(InferenceInterface) -> Rollout[] + Rewards[]
```

- **Agent** takes prompts, calls `inference_interface.generate()`, collects responses, computes
  rewards
- **InferenceInterface** wraps model inference (Megatron, OpenAI, HuggingFace, etc.)
- **Trainer** orchestrates batching, distributed rollout collection, and training steps

### Rollout Collection

1. `rl_utils.py` batches prompts across workers
2. Distributed inference via `InferenceInterface`
3. Agent evaluates responses (reward model, pass-at-k, etc.)
4. Gradients computed by GRPO/PPO algorithm

### Sequence Packing

RL training packs sequences for efficiency:
- `PackingInfo`: Metadata for bin allocation (seq_starts, seq_lengths, seq_to_bin_idx)
- Algorithms: FIFO, round-robin distribution
- `get_microbatch_dataloader()`: Creates DataLoader with packed sequences
- `pack_inference_logprobs()`: Unpacks logprobs to original sequence boundaries

---

## Common Tasks

### Adding a New Agent Type

1. Inherit from `TypeLookupable` in `agent/api.py`
2. Implement `generate_rollouts()` (or `generate_grouped_rollouts()`)
3. Register with `@BaseAgent.register_subclass()`
4. Examples: `RewardOnlyAgent`, `WeightedMultiTaskAgent`

### Implementing Custom Inference Interface

1. Subclass `InferenceInterface` in `inference/api.py`
2. Implement `generate()` method
3. Return `InferenceResponse` with text, logprobs, token_ids
4. Examples: `megatron.py` (Megatron backend), `huggingface_dataset_agent.py`

### Training with GRPO/PPO

1. Set up agent with inference interface
2. Call `rl_utils.train_rl_epoch()` or similar training function
3. Framework handles:
   - Batching prompts for parallel inference
   - Sequence packing for efficiency
   - Loss computation (advantage, policy, entropy)
   - Distributed synchronization via `parallel_utils`

### Debugging Rollout Generation

- Check `GenerationRequest`/`RolloutRequest` shape matching agent expectations
- Verify `InferenceInterface.generate()` returns correct response format
- Use `logging.setup_distributed_logging()` for distributed debugging
- Inspect packed sequence boundaries via `PackingInfo.seq_starts`, `seq_lengths`

---

## Dependencies

### Internal

- `megatron.core` - Inference, distributed utilities, model architectures
- `megatron.training` - Tokenizer, global args, training loop integration
- `megatron.core.full_cuda_graph`, `parallel_state` - Distributed training

### External

- `torch` - Distributed communication, tensor ops
- `pydantic` - Request/Response validation
- `numpy` - Array operations
- `yaml` - Config loading
- `fastapi` - Server infrastructure (optional, in `server/`)

---

## Gotchas & Considerations

### Sequence Packing Complexity

- `PackingInfo` tracks both bin-level and sequence-level metadata
- `seq_to_bin_idx` maps global sequence index → bin; use for unpacking logprobs
- FIFO vs round-robin packing can affect communication patterns in distributed training

### Distributed Synchronization

- `parallel_utils` provides barriers for multi-rank training
- All ranks must complete rollout generation before training step
- Use `get_num_microbatches()` to align with Megatron training loop

### Type Registration (TypeLookupable)

- Subclasses must define `type_name` field with unique default value
- Register via `@Agent.register_subclass()` decorator before deserialization
- Unwrap via `.unwrap()` to resolve from base type to concrete type

### Inference Interface Async

- Some implementations (e.g., remote agents) are async
- Use `trace_async_exceptions()` from `megatron.core.utils` for error handling
- Ensure event loop availability in distributed context

### Development Status

- Megatron-RL is actively developed; APIs may change
- External support limited; designed for research teams
- See `examples/rl/` for usage patterns
- Check GitHub issue #1776 for roadmap

---

## See Also

- `megatron/training/CLAUDE.md` - Training loop integration points
- `megatron/core/inference/CLAUDE.md` - Inference interface details
- `megatron/core/distributed/CLAUDE.md` - Distributed training utilities
- `examples/rl/` - Example agents and environments
- `README.md` - Design philosophy and high-level overview
