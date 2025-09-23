"""
Test configuration for per-layer embedding development
Creates minimal configurations for fast CPU testing
"""

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

def get_tiny_config():
    """Create a minimal configuration for CPU testing"""
    return TransformerConfig(
        num_layers=4,                    # Very small model
        hidden_size=64,                  # Tiny hidden size
        num_attention_heads=4,           # Small attention heads

        # Per-layer embedding config (new parameters - we'll add these later)
        # vocab_size_per_layer_input=500,  # Separate per-layer vocab
        # hidden_size_per_layer_input=32,  # Small per-layer dimension
        # use_per_layer_embedding=True,    # Flag to enable feature

        # Disable expensive features for CPU testing
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,

        # Standard config
        ffn_hidden_size=256,             # 4 * hidden_size
        kv_channels=16,                  # hidden_size / num_attention_heads
        add_bias_linear=False,
        gated_linear_unit=True,
        # activation_func will use default
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        add_qkv_bias=False,
    )

def get_model_params():
    """Get model parameters that are passed separately to GPTModel"""
    return {
        'vocab_size': 1000,              # Small vocab
        'max_sequence_length': 128,      # Short sequences

        # Per-layer embedding params (new)
        'vocab_size_per_layer_input': 500,  # Separate per-layer vocab
        'hidden_size_per_layer_input': 32,  # Small per-layer dimension
        'use_per_layer_embedding': True,    # Flag to enable feature
    }

def get_layer_spec():
    """Get the layer specification for testing"""
    return get_gpt_layer_local_spec()

if __name__ == "__main__":
    config = get_tiny_config()
    model_params = get_model_params()

    print("Tiny config created successfully!")
    print(f"Layers: {config.num_layers}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"FFN hidden size: {config.ffn_hidden_size}")

    print("\nModel parameters:")
    print(f"Vocab size: {model_params['vocab_size']}")
    print(f"Max sequence length: {model_params['max_sequence_length']}")
    print(f"Per-layer vocab size: {model_params['vocab_size_per_layer_input']}")
    print(f"Per-layer hidden size: {model_params['hidden_size_per_layer_input']}")

    print("\nLayer spec available:", get_layer_spec() is not None)