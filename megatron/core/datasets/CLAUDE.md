# CLAUDE.md - datasets

> Part of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
> **Purpose**: Data pipeline: indexed binary datasets, blending, and preprocessing
> **Parent**: [../CLAUDE.md](../CLAUDE.md)

---

## Overview

The datasets module provides efficient data loading for large-scale training:

- **IndexedDataset**: Binary (.bin + .idx) format for fast, memory-mapped access
- **Dataset Blending**: Mix multiple datasets with configurable weights
- **Task-Specific Loaders**: GPT, BERT, T5 implementations with proper masking
- **Object Storage**: Native S3/cloud object storage support
- **RETRO Support**: Retrieval-augmented training utilities

This module is the primary data pipeline for Megatron-LM training scripts.

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `indexed_dataset.py` | 1028 | Binary indexed dataset format, memory-mapped I/O |
| `gpt_dataset.py` | 882 | GPT/decoder training (causal LM) |
| `blended_megatron_dataset_builder.py` | 581 | Multi-dataset blending orchestration |
| `masked_dataset.py` | 441 | Token masking for BERT/masked LM |
| `t5_dataset.py` | 340 | T5/encoder-decoder training |
| `object_storage_utils.py` | 281 | S3/cloud storage backends |
| `blended_megatron_dataset_config.py` | 218 | Blending configuration and validation |
| `blended_dataset.py` | 239 | Base dataset blending logic |
| `bert_dataset.py` | 190 | BERT/masked LM training |
| `megatron_dataset.py` | 185 | Base MegatronDataset class |
| `megatron_tokenizer.py` | 162 | Tokenizer wrapper utilities |
| `retro/utils.py` | 386 | RETRO retrieval utilities |

---

## Architecture

### Design Patterns

1. **IndexedDataset** (indexed_dataset.py)
   - Binary format: `.bin` (data) + `.idx` (offsets)
   - Memory-mapped for constant memory footprint
   - Support for document-level and sample-level indexing
   - C++ helpers for performance-critical operations

2. **Dataset Hierarchy**
   ```
   MegatronDataset (base class)
   ├── GPTDataset (causal LM)
   ├── BertDataset (masked LM)
   ├── T5Dataset (encoder-decoder)
   └── MaskedDataset (token masking)
   ```

3. **Blended Datasets**
   - BlendedDataset: Static mix with fixed weights
   - BlendedMegatronDatasetBuilder: Dynamic blending with per-epoch sampling
   - Supports weighted round-robin or random sampling

4. **Object Storage**
   - S3Backend, MSCBackend for cloud datasets
   - Falls back to local filesystem if not configured
   - Lazy downloading on first access

### Key Concepts

- **Document**: Unit of text (e.g., article, document chunk)
- **Sequence**: Training sample (typically 1 document + padding or multiple documents)
- **Blending**: Mixing datasets by controlling sampling weights
- **Preprocessing**: Tokenization, masking, position IDs handled per task type
- **Sample Index**: Maps global sample ID to (dataset_id, local_sample_id)

---

## Common Tasks

### Load a Dataset for GPT Training
```python
from megatron.core.datasets import GPTDataset, BlendedMegatronDatasetBuilder

builder = BlendedMegatronDatasetBuilder(
    config=BlendedMegatronDatasetConfig(
        datasets=[
            {"name": "wikipedia", "weight": 0.7},
            {"name": "books", "weight": 0.3}
        ]
    )
)
dataset = builder.build()
dataloader = DataLoader(dataset, batch_size=32)
```

### Add a New Task-Specific Dataset
1. Subclass `MegatronDataset` in a new file
2. Implement `__getitem__()` to return (tokens, labels, loss_mask)
3. Register in `BlendedMegatronDatasetBuilder`
4. Add tests in `tests/unit_tests/datasets/`

### Create a Binary Indexed Dataset
```bash
# Use preprocessing scripts in tools/
python tools/preprocess_data.py \
  --input data.txt \
  --output-prefix dataset \
  --tokenizer-type GPT2BPE
```

---

## Dependencies

**Internal**:
- `megatron.core.tokenizers`: Tokenization abstractions
- `megatron.core.parallel_state`: GPU rank/world size info
- `megatron.core.utils`: Common utilities

**External**:
- `torch.utils.data`: DataLoader, IterableDataset
- `numpy`: Array operations
- `boto3` (optional): S3 access
- `retro` (optional): Retrieval-augmented training

---

## Gotchas & Best Practices

### Indexing
- Always verify .idx files match corresponding .bin files
- Use `IndexedDataset.exists()` before loading
- Re-index after modifying preprocessed data

### Blending
- Weights are normalized internally; use decimals (0.7, 0.3)
- Document boundaries are respected (won't split docs across samples)
- Per-epoch sampling in BlendedMegatronDatasetBuilder changes distribution

### Object Storage
- Test S3 credentials before large training runs
- Downloads are cached locally to avoid repeated transfers
- Set `retries=3` in config for unreliable networks

### Parallelism
- IndexedDataset is distributed-aware (handles rank-specific splits)
- Blending respects data parallelism; each rank gets same distribution
- For large datasets, use index caching (see `megatron_dataset.py`)

### Performance
- Use IndexedDataset (binary) over raw text for 10-50x speedup
- Memory-mapped I/O scales to TB-sized datasets
- Profile with `indexed_dataset.statistics()` to tune sample size

---

## Related Modules

- `megatron/training/arguments.py`: CLI args for data (vocab_size, seq_length, etc.)
- `megatron/core/tokenizers/`: Tokenizer implementations
- `megatron/core/datasets/retro/`: RETRO retrieval pipeline
- `tools/preprocess_data.py`: Dataset creation pipeline
