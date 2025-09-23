# GPT Training with Per-Layer Embedding

This directory contains training scripts and examples for GPT models with per-layer embedding support, based on the Gemma3n architecture.

## Overview

Per-layer embedding provides each transformer layer with its own specialized embeddings, enabling more sophisticated and layer-aware processing compared to standard transformer architectures.

## Quick Start

### 1. Prepare Your Data

First, preprocess your data for both standard GPT training and per-layer embedding:

```bash
# Standard GPT data preprocessing
python tools/preprocess_data.py \
    --input your_data.json \
    --output-prefix ./data/processed/my_data \
    --vocab-file ./data/vocab/gpt2-vocab.json \
    --merge-file ./data/vocab/gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer

# Per-layer data preprocessing
python preprocess_per_layer_data.py \
    --input your_data.json \
    --output-prefix ./data/processed/my_data \
    --per-layer-output-prefix ./data/processed/my_data_per_layer \
    --vocab-file ./data/vocab/gpt2-vocab.json \
    --per-layer-vocab-file ./data/vocab/per_layer_vocab.json \
    --per-layer-vocab-size 5000 \
    --per-layer-strategy syntax_focus
```

### 2. Small-Scale Training (Single GPU)

For development and testing:

```bash
cd examples/gpt_per_layer
bash train_gpt_per_layer_small.sh
```

### 3. Large-Scale Training (Multi-GPU)

For production training:

```bash
# Edit paths in the script first
bash ../../train_gpt_per_layer_embedding.sh \
    /path/to/checkpoints \
    /path/to/logs \
    /path/to/gpt2-vocab.json \
    /path/to/gpt2-merges.txt \
    /path/to/data \
    /path/to/per_layer_vocab.json \
    /path/to/per_layer_data
```

## Configuration Options

### Per-Layer Embedding Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-per-layer-embedding` | False | Enable per-layer embedding |
| `--per-layer-vocab-size` | 10000 | Size of per-layer vocabulary |
| `--per-layer-hidden-size` | 1024 | Hidden size for per-layer embeddings |
| `--per-layer-combination-method` | scaled_add | How to combine embeddings (scaled_add/concat/gated) |
| `--per-layer-gate-activation` | silu | Activation for per-layer gates |
| `--per-layer-projection-dropout` | 0.0 | Dropout for per-layer projections |

### Data Processing Strategies

| Strategy | Description |
|----------|-------------|
| `syntax_focus` | Focus on syntactic/structural elements |
| `semantic_focus` | Focus on semantic/content elements |
| `attention_guidance` | Provide attention guidance tokens |
| `random_subset` | Random subset baseline |

## Model Architecture

The per-layer embedding integration adds three main components:

1. **PerLayerEmbedding**: Creates separate embeddings for each layer
2. **PerLayerProjection**: Combines main and per-layer embeddings
3. **PerLayerGate**: Processes per-layer input within each transformer layer

```python
# Per-layer processing flow
per_layer_embeds = per_layer_embedding(per_layer_tokens)  # [B, S, L, H_per]
combined = per_layer_projection(main_embeds, per_layer_embeds)  # [B, S, L, H_per]

for layer_idx, transformer_layer in enumerate(layers):
    layer_input = combined[:, :, layer_idx, :]  # [B, S, H_per]
    hidden_states = per_layer_gate(hidden_states, layer_input)  # [B, S, H]
    hidden_states = transformer_layer(hidden_states)
```

## Performance Considerations

### Memory Usage
- Per-layer components add ~40-50% to total parameters
- Memory overhead scales with number of layers and per-layer hidden size
- Use smaller `per-layer-hidden-size` relative to main `hidden-size` for efficiency

### Computational Overhead
- ~10-20% computational overhead in production models (much higher in small test models)
- Overhead decreases as model size increases
- Can be optimized with tensor parallelism and gradient checkpointing

### Recommended Settings

#### Development/Testing
```bash
--per-layer-vocab-size 5000
--per-layer-hidden-size 256        # Small for testing
--micro-batch-size 4
--global-batch-size 16
```

#### Production (175B-scale)
```bash
--per-layer-vocab-size 10000
--per-layer-hidden-size 2048       # ~1/6 of main hidden size
--micro-batch-size 1
--global-batch-size 1536
```

## Data Format

### Input Data
Your input data should be in JSON Lines format:
```json
{"text": "This is a sample text for training."}
{"text": "Another example sentence."}
```

### Per-Layer Vocabulary
The per-layer vocabulary focuses on:
- Function words and syntax markers
- Common content words
- Special tokens for layer-specific processing

Example per-layer vocab:
```json
{
  "<|startoftext|>": 0,
  "<|endoftext|>": 1,
  "<|layer_start|>": 2,
  "<|attention|>": 3,
  "the": 4,
  "and": 5,
  ...
}
```

## Monitoring and Logging

### Per-Layer Statistics
Enable detailed logging:
```bash
--log-per-layer-stats
--log-per-layer-gradients
--per-layer-stats-interval 100
```

### TensorBoard Metrics
- Per-layer embedding norms
- Per-layer gradient norms
- Combination weights
- Layer-specific activations

## Troubleshooting

### Common Issues

**Data Mismatch Error**
```
ERROR: Per-layer data file not found
```
Solution: Run per-layer data preprocessing first

**Memory Issues**
```
CUDA out of memory
```
Solution: Reduce `per-layer-hidden-size` or `per-layer-vocab-size`

**Slow Training**
```
Training much slower than expected
```
Solution: Check tensor parallelism settings, reduce logging frequency

### Validation

Verify your setup:
```bash
# Test configuration
python -c "
from per_layer_arguments import validate_per_layer_args
from test_per_layer_config import get_tiny_config
config = get_tiny_config()
print('Configuration valid!')
"

# Test components
python per_layer_embedding_demo.py
```

## Example Results

With per-layer embedding, you should see:
- Faster convergence (lower loss in fewer steps)
- Better handling of syntactic structures
- Improved performance on tasks requiring layer-specific processing

## Integration with Existing Megatron Training

To integrate with existing training pipelines:

1. Add per-layer arguments to your argument parser
2. Modify your model provider to include per-layer components
3. Update data loading to include per-layer tokens
4. Extend logging to monitor per-layer statistics

See `pretrain_gpt_per_layer.py` for a complete integration example.

## Citations

Based on the per-layer embedding architecture from:
- Gemma 3 family models (Google DeepMind, 2024)
- Alternating Updates (Darcet et al., 2023)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Validate your configuration with the test scripts
3. Review the demo for proper usage patterns