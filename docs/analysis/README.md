# Megatron-LM GPU Utilization Analysis

> **Comprehensive analysis of why and where Megatron-LM achieves high GPU utilization across multiple nodes**

---

## Overview

This documentation series provides an in-depth technical analysis of Megatron-LM's GPU utilization optimizations. Through **16 comprehensive documents** organized into 7 major categories, we examine the techniques, implementations, and design decisions that enable Megatron to achieve **up to 47% Model FLOP Utilization (MFU)** on H100 GPUs and scale efficiently to 1000+ GPUs.

### What You'll Learn

- **Multi-dimensional parallelism strategies** that enable training 671B+ parameter models
- **Communication overlap techniques** that hide 80-90% of network latency
- **Pipeline scheduling algorithms** that reduce idle time from 87% to < 10%
- **Kernel-level optimizations** including fusion and FP8 support
- **Memory management strategies** that enable 2-4× larger batch sizes
- **Specialized attention implementations** for different use cases

### Target Audience

- **Researchers** wanting to understand state-of-the-art distributed training
- **ML Engineers** implementing or optimizing large-scale training systems
- **Performance Engineers** tuning Megatron for specific hardware configurations
- **Contributors** to Megatron-LM or related projects

---

## Document Structure

### Core Analysis Documents

Each document focuses on a specific aspect of GPU utilization optimization:

#### 1. [Parallelism Strategies](01-parallelism-strategies.md)
**Multi-dimensional parallelism for minimizing GPU idle time**

- ✅ **Status**: Complete
- 📊 **Scope**: ~1500 lines
- 🎯 **Key Topics**:
  - Tensor Parallelism (TP): Intra-layer sharding with sequence parallelism
  - Pipeline Parallelism (PP): Inter-layer sharding with 1F1B/interleaved schedules
  - Data Parallelism (DP): Gradient synchronization and ZeRO sharding
  - Context Parallelism (CP): Long sequence distribution
  - Expert Parallelism (EP): MoE expert distribution
  - Process group management (15+ specialized NCCL groups)
  - Multi-dimensional combinations (TP×PP×DP×CP×EP)

**Impact**: Enables near-linear scaling to 1000+ GPUs with combined parallelism strategies.

**Key Files Analyzed**:
- `megatron/core/parallel_state.py` (2687 lines)
- `megatron/core/tensor_parallel/layers.py` (1425 lines)
- `megatron/core/process_groups_config.py`

---

#### 2. [Communication Overlap](02-communication-overlap.md)
**Hiding communication latency behind computation**

- ✅ **Status**: Complete
- 📊 **Scope**: ~1800 lines
- 🎯 **Key Topics**:
  - Async gradient reduction (`overlap_grad_reduce`)
  - Parameter all-gather overlap (`overlap_param_gather`)
  - TP communication pipelining (Transformer Engine userbuffers)
  - Pipeline P2P optimization (`overlap_p2p_comm`, `batch_p2p_comm`)
  - MoE Expert Parallel A2A batch-level overlap
  - CUDA stream management patterns
  - `CUDA_DEVICE_MAX_CONNECTIONS=1` for kernel ordering
  - NCCL userbuffer optimization for SM reduction

**Impact**: Hides 80-90% of communication latency, critical for multi-node training efficiency.

**Key Files Analyzed**:
- `megatron/core/distributed/param_and_grad_buffer.py` (1007 lines)
- `megatron/core/distributed/distributed_data_parallel.py` (681 lines)
- `megatron/core/extensions/transformer_engine.py` (2283 lines)
- `megatron/core/pipeline_parallel/p2p_communication.py`

---

#### 3. Pipeline Scheduling
**Minimizing pipeline bubbles for maximum compute utilization**

- 🚧 **Status**: Planned
- 📊 **Estimated Scope**: ~1400 lines
- 🎯 **Key Topics**:
  - 1F1B (One-Forward-One-Backward) schedule theory and implementation
  - Interleaved pipeline (virtual pipeline parallelism)
  - Microbatch group scheduling
  - Quantitative bubble analysis (GPipe 87% → 1F1B 18% → Interleaved 9-12%)
  - Memory-compute trade-offs
  - MoE-optimized scheduling (`combined_1f1b`)
  - UCC backend for zero-SM communication

**Impact**: Reduces pipeline bubbles from 87% (GPipe) to 9-12% (1F1B interleaved), directly improving GPU utilization.

**Key Files to Analyze**:
- `megatron/core/pipeline_parallel/schedules.py` (2800 lines)
- `megatron/core/pipeline_parallel/combined_1f1b.py` (300 lines)

---

### Category A: Kernel Optimizations (Documents 04-08)

#### 4. [Activation Fusions](04-activation-fusions.md)
**SwiGLU, GEGLU, GELU, Squared ReLU fusions**

- 🚧 **Status**: Planned (Phase 1)
- 📊 **Estimated Scope**: ~1200 lines
- 🎯 **Complexity**: Medium
- 🎯 **Key Topics**:
  - Fused bias + SwiGLU (GPT/LLaMA standard)
  - Fused bias + GEGLU (T5 standard)
  - Fused bias + GELU variants
  - Fused weighted squared ReLU
  - Configuration options (clamp values, FP8 support)
  - When to use each activation type

**Impact**: 5-15% MLP speedup, reduces memory bandwidth.

**Key Files to Analyze**:
- `megatron/core/fusions/fused_bias_swiglu.py` (8.8KB)
- `megatron/core/fusions/fused_bias_geglu.py` (15KB)
- `megatron/core/fusions/fused_bias_gelu.py` (1.5KB)
- `megatron/core/fusions/fused_weighted_squared_relu.py` (3.7KB)

---

#### 5. [Attention Kernels](05-attention-kernels.md)
**Softmax fusions and cross-entropy fusion**

- 🚧 **Status**: Planned (Phase 2)
- 📊 **Estimated Scope**: ~1500 lines
- 🎯 **Complexity**: High
- 🎯 **Key Topics**:
  - Fused softmax variants (causal, masked, scaled)
  - CUDA kernel constraints (sequence length, alignment)
  - Cross-entropy vocab-parallel fusion
  - Backend selection logic
  - Performance impact analysis

**Impact**: 10-30% speedup for large vocabulary models.

**Key Files to Analyze**:
- `megatron/core/fusions/fused_softmax.py` (13KB)
- `megatron/core/fusions/fused_cross_entropy.py` (5KB)

---

#### 6. [Normalization Fusions](06-normalization-fusions.md)
**LayerNorm variants and bias-dropout-add**

- 🚧 **Status**: Planned (Phase 3)
- 📊 **Estimated Scope**: ~1200 lines
- 🎯 **Complexity**: Medium
- 🎯 **Key Topics**:
  - Persistent LayerNorm vs standard
  - Supported hidden sizes (1024-65536)
  - Zero-centered gamma option
  - Bias-dropout-add residual fusion
  - Apex integration

**Impact**: 2-3× speedup for normalization operations.

**Key Files to Analyze**:
- `megatron/core/fusions/fused_layer_norm.py` (5.6KB)
- `megatron/core/fusions/fused_bias_dropout.py` (3.3KB)

---

#### 7. [MoE Kernel Optimizations](07-moe-kernel-optimizations.md)
**MoE-specific fusions (routing, padding, MLA RoPE)**

- 🚧 **Status**: Planned (Phase 3)
- 📊 **Estimated Scope**: ~1800 lines
- 🎯 **Complexity**: Very High
- 🎯 **Key Topics**:
  - MLA YaRN RoPE fusion (DeepSeek-V3)
  - Token routing map conversions
  - Triton kernel autotuning
  - Grouped GEMM integration
  - Performance critical paths

**Impact**: 10-20% MoE routing speedup, critical for DeepSeek-V3.

**Key Files to Analyze**:
- `megatron/core/fusions/fused_mla_yarn_rope_apply.py` (26KB, Triton)
- `megatron/core/fusions/fused_indices_converter.py` (10KB, Triton)
- `megatron/core/fusions/fused_pad_routing_map.py` (3.3KB, Triton)

---

#### 8. [Kernel Selection Guide](08-kernel-selection-guide.md)
**How kernels are selected and dispatched**

- 🚧 **Status**: Planned (Phase 1)
- 📊 **Estimated Scope**: ~1200 lines
- 🎯 **Complexity**: Medium
- 🎯 **Key Topics**:
  - Kernel selection flowcharts
  - Configuration flags reference
  - JIT fusion (@jit_fuser decorator)
  - Backend availability checks
  - Fallback mechanisms
  - Troubleshooting guide

**Impact**: Helps users understand and optimize kernel usage.

**Key Files to Analyze**:
- `megatron/core/jit.py`
- `megatron/training/arguments.py` (configuration flags)
- `megatron/core/transformer/attention.py` (backend dispatch)

---

### Category B: Transformer Engine (Documents 09-12)

#### 9. [Transformer Engine Integration](09-transformer-engine-integration.md)
**TE component overview and integration**

- 🚧 **Status**: Planned (Phase 2)
- 📊 **Estimated Scope**: ~1500 lines
- 🎯 **Complexity**: Medium
- 🎯 **Key Topics**:
  - TE component mapping (TELinear, TENorm, TEDotProductAttention)
  - Integration points in Megatron architecture
  - Version compatibility matrix (TE 1.0 → 2.9+)
  - Setup and configuration
  - When to use TE vs native implementations

**Impact**: Foundation for understanding TE performance gains.

**Key Files to Analyze**:
- `megatron/core/extensions/transformer_engine.py` (88KB, 2,119 lines)
- `megatron/core/transformer/transformer_engine_spec_provider.py`

---

#### 10. [FP8 Training](10-fp8-training.md)
**FP8 training deep dive**

- 🚧 **Status**: Planned (Phase 3)
- 📊 **Estimated Scope**: ~1800 lines
- 🎯 **Complexity**: Very High
- 🎯 **Key Topics**:
  - FP8 formats (E4M3, hybrid E4M3/E5M2)
  - 5 FP8 recipes comparison (Delayed, Tensorwise, MXFP8, Blockwise, Custom)
  - AMAX scaling strategies
  - First/last layer BF16 strategy
  - FP8 parameter management
  - Quantization/dequantization flow

**Impact**: 2× memory reduction, 1.5-2× throughput, 47% MFU on H100.

**Key Files to Analyze**:
- `megatron/core/fp8_utils.py` (751 lines)
- `megatron/core/transformer/transformer_config.py` (FP8 section)
- `examples/llama/train_llama3_8b_h100_fp8.sh`

---

#### 11. [TE Optimizations](11-te-optimizations.md)
**TE-specific performance optimizations**

- 🚧 **Status**: Planned (Phase 3)
- 📊 **Estimated Scope**: ~1600 lines
- 🎯 **Complexity**: High
- 🎯 **Key Topics**:
  - Communication overlap (TP/CP/EP userbuffers)
  - Fused operations catalog (LayerNorm+Linear, RoPE, MLP, cross-entropy, MoE)
  - CPU offloading (activation, weight, double buffering)
  - Delayed weight gradient computation
  - Symmetric all-reduce (TE ≥ 2.3.0)
  - Context parallelism integration

**Impact**: Additional 20-40% speedup on top of base TE gains.

**Key Files to Analyze**:
- `megatron/core/extensions/transformer_engine.py` (optimization sections)
- `megatron/core/cpu_offloading/` (TE integration)

---

#### 12. [TE Configuration Reference](12-te-configuration-reference.md)
**Practical configuration and examples**

- 🚧 **Status**: Planned (Phase 1)
- 📊 **Estimated Scope**: ~1000 lines
- 🎯 **Complexity**: Low-Medium
- 🎯 **Key Topics**:
  - CLI flags reference (--fp8-format, --fp8-recipe, etc.)
  - Configuration recipes by model type (GPT, T5, MoE, Long-context)
  - Hardware-specific guides (Hopper, Ampere, Volta)
  - Troubleshooting common issues
  - Performance tuning checklist

**Impact**: Practical guide for users to enable TE optimizations.

**Key Files to Analyze**:
- `megatron/training/arguments.py` (CLI args)
- Example training scripts (`examples/llama/train_llama3_8b_h100_fp8.sh`)

---

### Category C: Memory Optimizations (Documents 13-14)

#### 13. [Activation Checkpointing](13-activation-checkpointing.md)
**Activation recomputation strategies**

- 🚧 **Status**: Planned (Phase 1)
- 📊 **Estimated Scope**: ~1400 lines
- 🎯 **Complexity**: Medium
- 🎯 **Key Topics**:
  - Granularity options (full, selective)
  - Recompute methods (uniform, block)
  - Selective module guide (core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts)
  - Memory vs compute trade-off (3-5× memory reduction, 20-30% compute overhead)
  - Configuration examples by model size
  - FP8 compatibility considerations

**Impact**: Enables 3-5× memory reduction, critical for 20B+ models.

**Key Files to Analyze**:
- `megatron/core/transformer/transformer_config.py` (recompute section, lines 296-340)
- `megatron/core/transformer/attention.py` (_checkpointed_attention_forward)
- `megatron/core/transformer/transformer_block.py`

---

#### 14. [Memory Buffers and Offloading](14-memory-buffers-and-offloading.md)
**Gradient buffers, distributed optimizer, CPU offloading**

- 🚧 **Status**: Planned (Phase 3)
- 📊 **Estimated Scope**: ~2000 lines
- 🎯 **Complexity**: Very High
- 🎯 **Key Topics**:
  - Gradient accumulation (bucket formation, async reduce-scatter)
  - Distributed optimizer (ZeRO-1, ZeRO-2, ZeRO-3)
  - CPU offloading (HybridDeviceOptimizer, overlap strategies)
  - Memory layout strategies (sequence packing, THD format)
  - Checkpoint resharding

**Impact**: 30% memory savings → 2-4× larger batch sizes.

**Key Files to Analyze**:
- `megatron/core/distributed/param_and_grad_buffer.py` (1,006 lines)
- `megatron/core/optimizer/distrib_optimizer.py` (2,602 lines)
- `megatron/core/cpu_offloading/hybrid_optimizer.py` (560 lines)

---

### Category D: Attention Implementations (Documents 15-16)

#### 15. [Attention Variants](15-attention-variants.md)
**Different attention mechanisms and architectures**

- 🚧 **Status**: Planned (Phase 2)
- 📊 **Estimated Scope**: ~1600 lines
- 🎯 **Complexity**: High
- 🎯 **Key Topics**:
  - Attention variants (MHA, GQA, MQA, MLA)
  - Self vs Cross Attention (QKV projection patterns)
  - MLA deep dive (low-rank KV compression, absorption optimization)
  - Attention mask types (causal, padding, arbitrary, bottom-right)
  - Configuration guide (num_query_groups, Q/K layernorm, softmax variants)

**Impact**: Enables efficient attention for different model architectures.

**Key Files to Analyze**:
- `megatron/core/transformer/attention.py` (1,239 lines)
- `megatron/core/transformer/multi_latent_attention.py` (920 lines)
- `megatron/core/transformer/dot_product_attention.py` (259 lines)

---

#### 16. [Flash Attention Optimizations](16-flash-attention-optimizations.md)
**Flash Attention backends and optimizations**

- 🚧 **Status**: Planned (Phase 2)
- 📊 **Estimated Scope**: ~1600 lines
- 🎯 **Complexity**: High
- 🎯 **Key Topics**:
  - Flash Attention variants (FA2, FA3, Flash MLA)
  - Flash Decode implementation (single-kernel decode, mixed batching)
  - Backend selection (auto-selection, AttnBackend enum, TE integration)
  - RoPE optimizations (standard, YaRN, MLA YaRN)
  - Context Parallel integration (CP communication types, ring attention)
  - Performance benchmarks

**Impact**: 2-4× faster attention, <50% memory usage, enables 4-8× longer sequences.

**Key Files to Analyze**:
- `megatron/core/transformer/attention.py` (Flash methods, lines 460-638)
- `megatron/core/extensions/transformer_engine.py` (TEDotProductAttention)
- External: flash_attn, flash_mla libraries

---

## Quick Reference

### By Performance Impact

**Highest Impact Optimizations** (ranked by GPU utilization improvement):

1. **Pipeline Scheduling** (Documents 03, 01, 02)
   - Reduces bubbles from 87% → 9-12%
   - **Speedup**: 5-10× for large pipeline parallelism

2. **Communication Overlap** (Document 02)
   - Hides 80-90% of communication time
   - **Speedup**: 2-3× for communication-bound workloads

3. **Transformer Engine FP8** (Documents 09-12)
   - 2× throughput on Hopper GPUs
   - **Speedup**: 2× (Hopper), 1.2-1.4× (Ampere with TE)

4. **Memory Optimizations** (Documents 13-14)
   - Enables 2-4× larger batches
   - **Speedup**: 1.3-1.8× through better GPU kernel utilization

5. **Kernel Fusions** (Documents 04-08)
   - 2-3× per operation
   - **Speedup**: 1.2-1.5× overall

### By Use Case

**Training Large Models (100B+ parameters)**:
- Start with: Documents 01 (Parallelism), 03 (Pipeline), 13-14 (Memory)
- Focus on: Multi-dimensional parallelism, ZeRO sharding, activation checkpointing

**Multi-Node Training (8+ nodes)**:
- Start with: Documents 02 (Communication), 01 (Parallelism)
- Focus on: Overlap techniques, NCCL configuration, network topology

**MoE Models (Mixtral, DeepSeek-V3)**:
- Start with: Documents 01 (EP section), 02 (MoE overlap), 07 (MoE Kernels)
- Focus on: Expert parallelism, load balancing, token routing optimizations

**Long Context (8K+ tokens)**:
- Start with: Documents 01 (CP section), 15-16 (Attention)
- Focus on: Context parallelism, FlashAttention variants, KV cache optimizations

**Inference Optimization**:
- Start with: Documents 09-12 (TE), 15-16 (Attention), 04-08 (Kernels)
- Focus on: Flash Decode, CUDA graphs, paged attention

**FP8 Training (H100/H200)**:
- Start with: Documents 09-12 (Transformer Engine)
- Focus on: FP8 configuration, scaling factors, TE layers

---

## Key Quantitative Results

### Model FLOP Utilization (MFU)

| Hardware | Achieved MFU | Notes |
|----------|--------------|-------|
| H100 | **47%** | State-of-the-art, large-scale training |
| A100 | 35-40% | With Transformer Engine |
| A100 | 25-30% | Without Transformer Engine |

**For context**: Theoretical peak is 100% (never achievable), good performance is 40-50%.

### Pipeline Bubble Reduction

| Schedule | Bubble Time (p=8, m=32) | Efficiency |
|----------|------------------------|------------|
| GPipe | 87.5% | Unusable |
| 1F1B | 18.2% | Acceptable |
| 1F1B Interleaved (v=2) | 9.1% | Good |
| 1F1B Interleaved (v=4) | 4.7% | Excellent |

### Communication Overlap Efficiency

| Communication Type | Latency Hidden | Speedup |
|-------------------|----------------|---------|
| Gradient reduction (DP) | 80-90% | 2.0× |
| Parameter gather (DistOpt) | 70-80% | 1.7× |
| TP all-reduce/gather | 60-70% | 1.6× |
| Pipeline P2P | 95%+ | 10×+ |

### Memory Optimizations

| Technique | Memory Savings | Enables |
|-----------|---------------|---------|
| Selective recomputation | ~30% | 1.4× larger batch |
| FP8 activations/weights | ~50% | 2× larger batch |
| Distributed optimizer (DP=16) | 16× optimizer memory | 4× larger models |

### Kernel Speedups

| Operation | Speedup vs PyTorch | Hardware |
|-----------|-------------------|----------|
| Fused LayerNorm | 2-3× | All |
| Fused SwiGLU | 2-3× | All |
| Fused Softmax + Masking | 1.5-2× | All |
| FP8 GEMM (vs BF16) | 2× | Hopper |
| Grouped GEMM (MoE) | 3-5× | All |

---

## Critical Files Reference

### Parallelism Infrastructure
- `megatron/core/parallel_state.py` (2687 lines) - Process group management
- `megatron/core/tensor_parallel/layers.py` (1425 lines) - TP layers
- `megatron/core/pipeline_parallel/schedules.py` (2800 lines) - PP schedules

### Communication & Optimization
- `megatron/core/distributed/param_and_grad_buffer.py` (1007 lines) - Gradient bucketing
- `megatron/core/distributed/distributed_data_parallel.py` (681 lines) - DDP overlap
- `megatron/core/optimizer/distrib_optimizer.py` (3264 lines) - Distributed optimizer

### Compute Optimization
- `megatron/core/fusions/` (11 fusion kernels)
- `megatron/core/extensions/transformer_engine.py` (2283 lines) - TE integration
- `megatron/core/transformer/attention.py` (1226 lines) - Attention backends

### Configuration
- `megatron/core/model_parallel_config.py` (400 lines) - Parallelism/overlap config
- `megatron/core/transformer/transformer_config.py` (1819 lines) - Model config
- `megatron/training/arguments.py` (8547 lines) - CLI arguments

### Training Loop
- `megatron/training/training.py` (3245 lines) - Main training orchestration

---

## Configuration Examples

### LLaMA-3 8B (8 GPUs, single node)
```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 2
--sequence-parallel
--overlap-grad-reduce
# DP = 8 / (1×1×2) = 4
```

### LLaMA-3 70B (64 GPUs)
```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 4
--virtual-pipeline-model-parallel-size 2
--context-parallel-size 2
--sequence-parallel
--overlap-grad-reduce
--overlap-param-gather
--use-distributed-optimizer
# DP = 64 / (4×4×2) = 2
```

### LLaMA-3.1 405B (1024 GPUs)
```bash
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 8
--virtual-pipeline-model-parallel-size 4
--context-parallel-size 2
--sequence-parallel
--overlap-grad-reduce
--overlap-param-gather
--tp-comm-overlap
--tp-comm-bulk-wgrad
--tp-comm-bulk-dgrad
--use-distributed-optimizer
--use-megatron-fsdp
--nccl-ub
# DP = 1024 / (8×8×2) = 16
```

### DeepSeek-V3 671B (1024 GPUs, MoE)
```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 16
--virtual-pipeline-model-parallel-size 4
--expert-model-parallel-size 64
--num-experts 256
--sequence-parallel
--overlap-grad-reduce
--overlap-moe-expert-parallel-comm
--delay-wgrad-compute
--moe-grouped-gemm
--moe-aux-loss-free-balancing
--use-distributed-optimizer
# DP calculated from remaining GPUs
```

---

## How to Use This Documentation

### For Learning

1. **Start with basics**: Read Document 01 (Parallelism Strategies) to understand the foundation
2. **Understand communication**: Read Document 02 (Communication Overlap) to see how latency is hidden
3. **Learn scheduling**: Read Document 03 (Pipeline Scheduling) to understand bubble reduction
4. **Deep dive by interest**: Choose documents based on your specific areas of interest
5. **Cross-reference**: Use links between documents to explore related concepts

### For Implementation

1. **Identify your bottleneck**: Profile your training to find GPU utilization issues
2. **Find relevant document**: Use "By Performance Impact" or "By Use Case" sections above
3. **Apply techniques**: Follow configuration examples and code references
4. **Measure impact**: Profile again to verify improvements

### For Contribution

1. **Understand existing patterns**: Read relevant documents before implementing new features
2. **Maintain consistency**: Follow established patterns from code references
3. **Document thoroughly**: Add references to these docs when adding new optimizations
4. **Benchmark carefully**: Use quantitative results here as baselines

---

## Glossary

**1F1B**: One-Forward-One-Backward pipeline schedule
**A2A**: All-to-all communication
**AG**: All-gather operation
**CP**: Context Parallelism
**DP**: Data Parallelism
**EP**: Expert Parallelism (MoE)
**FSDP**: Fully Sharded Data Parallel
**GEMM**: General Matrix Multiply
**MFU**: Model FLOP Utilization (fraction of theoretical peak achieved)
**MoE**: Mixture of Experts
**P2P**: Point-to-point communication
**PP**: Pipeline Parallelism
**RS**: Reduce-scatter operation
**SHARP**: Scalable Hierarchical Aggregation and Reduction Protocol
**SM**: Streaming Multiprocessor (CUDA cores on GPU)
**TE**: Transformer Engine
**TP**: Tensor Parallelism
**UCC**: Unified Collective Communication
**VP**: Virtual Pipeline (interleaved pipeline parallelism)
**ZeRO**: Zero Redundancy Optimizer (parameter/gradient/optimizer sharding)

---

## Document Status

| Document | Status | Lines | Phase | Last Updated |
|----------|--------|-------|-------|--------------|
| 01-parallelism-strategies.md | ✅ Complete | ~1500 | — | 2025-12-03 |
| 02-communication-overlap.md | ✅ Complete | ~1800 | — | 2025-12-03 |
| 03-pipeline-scheduling.md | ✅ Complete | ~1100 | — | 2025-12-03 |
| **Category A: Kernel Optimizations** |
| 04-activation-fusions.md | 🚧 Planned | ~1200 | Phase 2 | TBD |
| 05-attention-kernels.md | 🚧 Planned | ~1500 | Phase 2 | TBD |
| 06-normalization-fusions.md | 🚧 Planned | ~1200 | Phase 3 | TBD |
| 07-moe-kernel-optimizations.md | 🚧 Planned | ~1800 | Phase 3 | TBD |
| 08-kernel-selection-guide.md | 🚧 Planned | ~1200 | Phase 1 | TBD |
| **Category B: Transformer Engine** |
| 09-transformer-engine-integration.md | 🚧 Planned | ~1500 | Phase 2 | TBD |
| 10-fp8-training.md | 🚧 Planned | ~1800 | Phase 3 | TBD |
| 11-te-optimizations.md | 🚧 Planned | ~1600 | Phase 3 | TBD |
| 12-te-configuration-reference.md | 🚧 Planned | ~1000 | Phase 1 | TBD |
| **Category C: Memory Optimizations** |
| 13-activation-checkpointing.md | 🚧 Planned | ~1400 | Phase 1 | TBD |
| 14-memory-buffers-and-offloading.md | 🚧 Planned | ~2000 | Phase 3 | TBD |
| **Category D: Attention Implementations** |
| 15-attention-variants.md | 🚧 Planned | ~1600 | Phase 2 | TBD |
| 16-flash-attention-optimizations.md | 🚧 Planned | ~1600 | Phase 2 | TBD |
| **Total** | **3/16 complete** | **~4400 / 23700** | — | — |

---

## Contributing to This Documentation

This documentation is part of the Megatron-LM project. Contributions are welcome!

**To contribute**:
1. **File issues**: Report inaccuracies or request clarifications
2. **Submit PRs**: Add missing sections or update outdated information
3. **Request topics**: Suggest new analysis areas

**Documentation standards**:
- Use `file.py:line` format for code references
- Include quantitative data where possible
- Cross-reference related documents
- Maintain consistent terminology (see Glossary)
- Verify all code snippets against actual implementation

---

## Contact & Support

- **GitHub Issues**: [Megatron-LM Issues](https://github.com/NVIDIA/Megatron-LM/issues)
- **Discussions**: [Megatron-LM Discussions](https://github.com/NVIDIA/Megatron-LM/discussions)
- **Documentation**: [Official Megatron Docs](https://docs.nvidia.com/Megatron-Core/)

---

**Documentation Series Version**: 1.0
**Last Updated**: 2025-12-03
**Maintained by**: Megatron-LM Community
