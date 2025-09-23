"""
Per-Layer Embedding Implementation for Megatron GPT
Based on Gemma3n per-layer embedding architecture

This implements the key components:
1. PerLayerEmbedding: Separate embedding for each layer
2. PerLayerProjection: Projects main embeddings to per-layer space
3. PerLayerGate: Processes per-layer input in each transformer layer
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule


class PerLayerEmbedding(MegatronModule):
    """
    Per-layer embedding that creates embeddings for all layers simultaneously.

    This embeds per-layer tokens and reshapes output to provide separate
    embeddings for each transformer layer.

    Args:
        config: TransformerConfig
        vocab_size_per_layer: Size of per-layer vocabulary
        hidden_size_per_layer: Hidden size for per-layer embeddings
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size_per_layer: int,
        hidden_size_per_layer: int,
        num_layers: int,
    ):
        super().__init__(config)

        self.vocab_size_per_layer = vocab_size_per_layer
        self.hidden_size_per_layer = hidden_size_per_layer
        self.num_layers = num_layers

        # Single embedding that outputs for all layers at once
        self.embedding = nn.Embedding(
            vocab_size_per_layer,
            num_layers * hidden_size_per_layer
        )

        # Scale factor similar to Gemma3n
        self.embed_scale = hidden_size_per_layer ** 0.5

    def forward(self, per_layer_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for per-layer embedding.

        Args:
            per_layer_input_ids: [batch_size, seq_len] tensor of per-layer token IDs

        Returns:
            per_layer_embeds: [batch_size, seq_len, num_layers, hidden_size_per_layer]
        """
        batch_size, seq_len = per_layer_input_ids.shape

        # Embed tokens for all layers at once
        embedded = self.embedding(per_layer_input_ids)  # [B, S, num_layers * hidden_size_per_layer]
        embedded = embedded * self.embed_scale

        # Reshape to separate per-layer embeddings
        per_layer_embeds = embedded.view(
            batch_size, seq_len, self.num_layers, self.hidden_size_per_layer
        )

        return per_layer_embeds


class PerLayerProjection(MegatronModule):
    """
    Projects main token embeddings to per-layer space and combines with per-layer embeddings.

    This is analogous to Gemma3n's per_layer_model_projection functionality.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size_per_layer: int,
        num_layers: int,
    ):
        super().__init__(config)

        self.hidden_size_per_layer = hidden_size_per_layer
        self.num_layers = num_layers

        # Project main embeddings to per-layer space for all layers
        self.projection = nn.Linear(
            config.hidden_size,
            num_layers * hidden_size_per_layer,
            bias=False
        )

        # RMS normalization for per-layer inputs
        self.norm = nn.RMSNorm(hidden_size_per_layer, eps=config.layernorm_epsilon)

        # Scale factor for combining projections (1/sqrt(2) like Gemma3n)
        self.combination_scale = 1.0 / math.sqrt(2.0)
        self.projection_scale = config.hidden_size ** -0.5

    def forward(
        self,
        main_embeddings: torch.Tensor,
        per_layer_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project main embeddings and combine with per-layer embeddings.

        Args:
            main_embeddings: [batch_size, seq_len, hidden_size] main token embeddings
            per_layer_embeddings: [batch_size, seq_len, num_layers, hidden_size_per_layer]
                                 per-layer embeddings (optional)

        Returns:
            combined: [batch_size, seq_len, num_layers, hidden_size_per_layer]
        """
        batch_size, seq_len = main_embeddings.shape[:2]

        # Project main embeddings to per-layer space
        projected = self.projection(main_embeddings) * self.projection_scale
        projected = projected.view(
            batch_size, seq_len, self.num_layers, self.hidden_size_per_layer
        )

        # Apply normalization
        projected = self.norm(projected)

        if per_layer_embeddings is None:
            return projected

        # Combine projected main embeddings with per-layer embeddings
        combined = (projected + per_layer_embeddings) * self.combination_scale

        return combined


class PerLayerGate(MegatronModule):
    """
    Per-layer gate that processes per-layer input within a transformer layer.

    This implements the per-layer processing inside each transformer layer,
    similar to Gemma3n's per_layer_input_gate and per_layer_projection.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size_per_layer: int,
        layer_idx: int,
    ):
        super().__init__(config)

        self.hidden_size_per_layer = hidden_size_per_layer
        self.layer_idx = layer_idx

        # Gate that controls how per-layer input affects this layer
        self.gate = nn.Linear(config.hidden_size, hidden_size_per_layer, bias=False)

        # Projection back to main hidden space
        self.projection = nn.Linear(hidden_size_per_layer, config.hidden_size, bias=False)

        # Post-projection normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # Activation function (using SiLU like Gemma3n)
        self.activation = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply per-layer processing to hidden states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] main hidden states
            per_layer_input: [batch_size, seq_len, hidden_size_per_layer] this layer's input

        Returns:
            output: [batch_size, seq_len, hidden_size] processed hidden states
        """
        # Gate the hidden states to per-layer space
        gated = self.gate(hidden_states)
        gated = self.activation(gated)

        # Element-wise multiply with per-layer input (key interaction)
        per_layer_processed = gated * per_layer_input

        # Project back to main hidden space
        projected = self.projection(per_layer_processed)
        projected = self.norm(projected)

        # Add to original hidden states (residual connection)
        output = hidden_states + projected

        return output


# Test functions
def test_per_layer_embedding():
    """Test the per-layer embedding component"""
    print("Testing PerLayerEmbedding...")

    from test_per_layer_config import get_tiny_config, get_model_params

    config = get_tiny_config()
    model_params = get_model_params()

    embedding = PerLayerEmbedding(
        config=config,
        vocab_size_per_layer=model_params['vocab_size_per_layer_input'],
        hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
        num_layers=config.num_layers,
    )

    # Test input
    batch_size, seq_len = 2, 16
    per_layer_input_ids = torch.randint(0, model_params['vocab_size_per_layer_input'], (batch_size, seq_len))

    # Forward pass
    per_layer_embeds = embedding(per_layer_input_ids)

    expected_shape = (batch_size, seq_len, config.num_layers, model_params['hidden_size_per_layer_input'])
    assert per_layer_embeds.shape == expected_shape, f"Expected {expected_shape}, got {per_layer_embeds.shape}"

    print(f"✅ PerLayerEmbedding output shape: {per_layer_embeds.shape}")
    return per_layer_embeds


def test_per_layer_projection():
    """Test the per-layer projection component"""
    print("Testing PerLayerProjection...")

    from test_per_layer_config import get_tiny_config, get_model_params

    config = get_tiny_config()
    model_params = get_model_params()

    projection = PerLayerProjection(
        config=config,
        hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
        num_layers=config.num_layers,
    )

    # Test inputs
    batch_size, seq_len = 2, 16
    main_embeddings = torch.randn(batch_size, seq_len, config.hidden_size)
    per_layer_embeddings = torch.randn(
        batch_size, seq_len, config.num_layers, model_params['hidden_size_per_layer_input']
    )

    # Test projection only
    projected_only = projection(main_embeddings)
    expected_shape = (batch_size, seq_len, config.num_layers, model_params['hidden_size_per_layer_input'])
    assert projected_only.shape == expected_shape

    # Test projection + combination
    combined = projection(main_embeddings, per_layer_embeddings)
    assert combined.shape == expected_shape

    print(f"✅ PerLayerProjection output shape: {combined.shape}")
    return combined


def test_per_layer_gate():
    """Test the per-layer gate component"""
    print("Testing PerLayerGate...")

    from test_per_layer_config import get_tiny_config, get_model_params

    config = get_tiny_config()
    model_params = get_model_params()

    gate = PerLayerGate(
        config=config,
        hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
        layer_idx=0,
    )

    # Test inputs
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    per_layer_input = torch.randn(batch_size, seq_len, model_params['hidden_size_per_layer_input'])

    # Forward pass
    output = gate(hidden_states, per_layer_input)

    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    print(f"✅ PerLayerGate output shape: {output.shape}")
    return output


if __name__ == "__main__":
    print("🧪 Testing Per-Layer Embedding Components...")

    try:
        # Test individual components
        per_layer_embeds = test_per_layer_embedding()
        combined_embeds = test_per_layer_projection()
        gated_output = test_per_layer_gate()

        print("\n🎉 All per-layer embedding components working correctly!")

        # Test gradient flow
        print("\n🔄 Testing gradient flow...")
        loss = (per_layer_embeds.sum() + combined_embeds.sum() + gated_output.sum())
        loss.backward()
        print("✅ Gradients flow correctly through all components")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()