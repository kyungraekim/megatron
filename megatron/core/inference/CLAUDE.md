# Inference Guide

> Covers: inference engines, text generation, KV cache, model wrappers
> Parent: [../CLAUDE.md](../CLAUDE.md)

## Key Concepts

- **Two modes**: `DynamicEngine` (variable batching, throughput-optimized) vs `StaticEngine` (fixed shapes, latency-optimized). Prefer Dynamic for production.
- **Request lifecycle**: submit `InferenceRequest` → `Scheduler` batches → `DynamicContext` prepares GPU tensors → model forward → sample token → return via ZMQ.
- **AbstractModelInferenceWrapper**: required interface for custom models — implement `prepare_batch()`, `forward()`, `get_updated_kv_cache()`.

## New in v0.16

- **Dynamic text generation server** (`text_generation_server/dynamic_text_gen_server/`): Flask-based server with OpenAI-compatible `/chat/completions` and `/completions` endpoints.
- **Automatic port selection**: ZMQ now auto-chooses available ports.

## Gotchas

- **DynamicBlockAllocator** required for fragmentation-free KV cache. The fused append kernel requires contiguous blocks.
- Model wrappers **must broadcast logits from last pipeline stage** via `communication_utils.broadcast_from_last_stage()`. Easy to forget in custom models.
- **Static engine** pre-allocates all KV cache upfront. Use only with known batch size.
- ZMQ server **doesn't clean up orphaned requests**. Implement timeout logic in production.
- `tokenization.py` expects HuggingFace tokenizers. Custom tokenizers need adapter code.
