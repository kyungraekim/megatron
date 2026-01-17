# CLAUDE.md - inference

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Inference engine: dynamic batching, text generation, and model wrappers
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

The `inference` directory provides NVIDIA Megatron-LM's production inference engine for
efficient large-scale model serving. It implements dynamic batching, request scheduling,
and multi-GPU text generation orchestration.

**Problem Solved**: Training libraries are optimized for throughput with fixed batch sizes.
Inference requires variable batch sizes, priority scheduling, and low latency. This module
bridges that gap with two execution modes:
- **Dynamic Mode**: Variable batching, throughput-optimized
- **Static Mode**: Fixed shapes, latency-optimized

**Key Concepts**:
- **DynamicEngine**: Main inference engine with dynamic batching and request scheduler
- **DynamicContext**: GPU execution state (KV cache, token positions, request tracking)
- **InferenceRequest**: Unit of work (prompt, generation config, metadata)
- **Model Wrappers**: Abstractions for GPT, T5, and multimodal models
- **Scheduler**: Priority-based request batching and sequencing
- **KV Cache Management**: Contiguous memory allocation with fused kernels

---

## File Map

| File | Lines | Purpose | Key Exports |
|------|-------|---------|-------------|
| `contexts/dynamic_context.py` | 2143 | GPU execution state for dynamic mode | `DynamicContext`, state management |
| `engines/dynamic_engine.py` | 1636 | Main inference engine, batching logic | `DynamicEngine`, request loop |
| `text_generation_controllers/text_generation_controller.py` | 1570 | Text generation orchestration | `TextGenerationController`, generation loop |
| `inference_request.py` | 537 | Request object, generation config | `InferenceRequest`, `GenerationConfig` |
| `batch_dimensions_utils.py` | 486 | Batch size calculations | `get_batch_sizes()` utilities |
| `unified_memory.py` | 478 | Unified memory management | `UnifiedMemory` allocator |
| `model_inference_wrappers/abstract_*.py` | 401 | Base class for model wrappers | `AbstractModelInferenceWrapper` |
| `engines/static_engine.py` | 393 | Fixed-batch inference | `StaticEngine` |
| `data_parallel_inference_coordinator.py` | 313 | DP coordination | `DataParallelInferenceCoordinator` |
| `inference_client.py` | 251 | Client interface, ZMQ transport | `InferenceClient` |
| `utils.py` | 218 | Tensor utilities, helpers | `get_last_token_logits()` etc |
| `text_generation_server/text_generation_server.py` | 211 | Server implementation | `TextGenerationServer` |
| `communication_utils.py` | 211 | Communication utilities | `broadcast_from_last_stage()` |
| `scheduler.py` | 193 | Request scheduler | `Scheduler` |
| `contexts/fused_kv_append_kernel.py` | 174 | CUDA kernel for KV append | Kernel wrappers |
| `contexts/dynamic_block_allocator.py` | 131 | KV cache block allocation | `DynamicBlockAllocator` |
| `contexts/static_context.py` | 130 | GPU state for static mode | `StaticContext` |
| `engines/async_zmq_communicator.py` | 116 | Async ZMQ communication | `AsyncZmqCommunicator` |
| `text_generation_server/run_mcore_engine.py` | 113 | Server entry point | Launch script |
| `text_generation_server/tokenization.py` | 110 | Tokenizer integration | Token encoding/decoding |

---

## Architecture

### Execution Modes

```
DynamicEngine (variable batching)        StaticEngine (fixed batching)
       ↓                                        ↓
  DynamicContext                          StaticContext
  (GPU memory, batch state)               (Fixed shapes, prealloc)
       ↓                                        ↓
   Scheduler                              Direct inference
   (priority queue)                       (no scheduling)
       ↓
  Batched forward pass
```

### Request Lifecycle (DynamicEngine)

1. **Submission**: Client sends `InferenceRequest` (prompt + config)
2. **Scheduling**: `Scheduler` batches compatible requests
3. **Context Prep**: `DynamicContext.prepare_batch()` arranges GPU tensors
4. **Forward Pass**: Model forward with dynamic batch size
5. **Token Generation**: Sample next token, update KV cache
6. **Completion**: Return tokens to client via ZMQ

### Key Components

- **DynamicContext**: Manages GPU tensors, KV cache allocation, request-to-batch mappings
- **Scheduler**: Priority queue, batching heuristics, request ordering
- **Model Wrappers**: Transform requests into forward calls (GPT, T5, multimodal)
- **TextGenerationController**: High-level generation loop (sampling, stopping)
- **TextGenerationServer**: ZMQ-based client/server (for distributed serving)

### Data Flow

```
InferenceRequest → Scheduler → DynamicContext → Model Wrapper → forward()
                                     ↓
                        KV Cache + Attention Mask
                        (fused append kernel)
                                     ↓
                        Token logits → TextGenerationController
                                     → Sample/Greedy
                                     → Return to client
```

---

## Common Tasks

### Adding a New Model Wrapper

```python
# megatron/core/inference/model_inference_wrappers/my_model_wrapper.py
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper \
    import AbstractModelInferenceWrapper

class MyModelWrapper(AbstractModelInferenceWrapper):
    def prepare_batch(self, requests, batch_size):
        # Convert requests to input tensors
        return prompt_ids, attention_mask

    def forward(self, input_ids, attention_mask):
        return logits  # Shape: [batch_size, vocab_size]

    def get_updated_kv_cache(self):
        return self.model.get_new_cache()
```

### Using DynamicEngine Programmatically

```python
from megatron.core.inference.engines.dynamic_engine import DynamicEngine
from megatron.core.inference.inference_request import InferenceRequest

engine = DynamicEngine(model, ...)
request = InferenceRequest(prompt_tokens=[1, 2, 3], max_seq_len=512)
response = engine.add_request(request)
tokens = engine.generate(num_tokens=10)
```

### Configuring the Scheduler

```python
# In DynamicEngine init
self.scheduler.max_batch_size = 32
self.scheduler.batching_strategy = "continuous_batching"
# Matches requests with similar sequence lengths for efficiency
```

---

## Dependencies

**Depends On**:
- `megatron.core.parallel_state` - Process groups, rank info
- `megatron.core.transformer` - Model layers, configs
- `megatron.core.extensions.transformer_engine` - Optional FP8 kernels
- `megatron.core.models` - Model implementations (GPT, T5)
- Standard PyTorch, CUDA

**Used By**:
- `megatron/training/` - Inference utilities for eval
- `examples/` - Inference scripts
- External services - ZMQ server for remote inference

---

## Gotchas & Tips

1. **KV Cache Memory**: Use `DynamicBlockAllocator` for fragmentation-free allocation.
   Contiguous blocks are required for the fused append kernel.

2. **Batch Incompatibility**: `Scheduler` should not mix requests with vastly different
   sequence lengths. Add a `max_seq_len_diff` parameter if performance matters.

3. **TP/PP Coordination**: Model wrappers must broadcast logits from last pipeline stage
   (see `communication_utils.broadcast_from_last_stage()`). Easy to forget with custom models.

4. **Static Engine Limitations**: Pre-allocates all KV cache upfront. Use only when you
   know batch size in advance. DynamicEngine is preferred for variable workloads.

5. **Client Disconnects**: ZMQ server doesn't clean up orphaned requests automatically.
   Implement timeout logic in production deployments.

6. **Tokenizer Serialization**: `tokenization.py` expects HuggingFace tokenizers. Custom
   tokenizers need adapter code.
