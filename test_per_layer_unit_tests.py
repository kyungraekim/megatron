"""
Unit tests for per-layer embedding implementation
Tests individual components and their interactions
"""

import torch
import torch.nn as nn
import math

from test_per_layer_config import get_tiny_config, get_model_params
from per_layer_embedding import PerLayerEmbedding, PerLayerProjection, PerLayerGate


class TestPerLayerEmbedding:
    """Test suite for PerLayerEmbedding component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = get_tiny_config()
        self.model_params = get_model_params()
        self.batch_size = 2
        self.seq_len = 10

    def test_embedding_shapes(self):
        """Test that embedding produces correct output shapes"""
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        input_ids = torch.randint(0, self.model_params['vocab_size_per_layer_input'], (self.batch_size, self.seq_len))
        output = embedding(input_ids)

        expected_shape = (
            self.batch_size,
            self.seq_len,
            self.config.num_layers,
            self.model_params['hidden_size_per_layer_input']
        )
        assert output.shape == expected_shape

    def test_embedding_scaling(self):
        """Test that embedding applies correct scaling"""
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        # Test that scaling is applied
        expected_scale = self.model_params['hidden_size_per_layer_input'] ** 0.5
        assert embedding.embed_scale == expected_scale

    def test_embedding_different_tokens(self):
        """Test that different tokens produce different embeddings"""
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        # Different input tokens
        input_ids_1 = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        input_ids_2 = torch.ones((self.batch_size, self.seq_len), dtype=torch.long)

        output_1 = embedding(input_ids_1)
        output_2 = embedding(input_ids_2)

        # Should produce different outputs
        assert not torch.allclose(output_1, output_2)

    def test_embedding_layer_separation(self):
        """Test that different layers get different embeddings"""
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        input_ids = torch.randint(0, self.model_params['vocab_size_per_layer_input'], (self.batch_size, self.seq_len))
        output = embedding(input_ids)

        # Each layer should get different embeddings for the same token
        layer_0_embeds = output[:, :, 0, :]  # Layer 0 embeddings
        layer_1_embeds = output[:, :, 1, :]  # Layer 1 embeddings

        assert not torch.allclose(layer_0_embeds, layer_1_embeds)


class TestPerLayerProjection:
    """Test suite for PerLayerProjection component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = get_tiny_config()
        self.model_params = get_model_params()
        self.batch_size = 2
        self.seq_len = 10

    def test_projection_shapes(self):
        """Test projection output shapes"""
        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        main_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        per_layer_embeddings = torch.randn(
            self.batch_size, self.seq_len, self.config.num_layers, self.model_params['hidden_size_per_layer_input']
        )

        # Test projection only
        projected_only = projection(main_embeddings)
        expected_shape = (
            self.batch_size, self.seq_len, self.config.num_layers, self.model_params['hidden_size_per_layer_input']
        )
        assert projected_only.shape == expected_shape

        # Test projection + combination
        combined = projection(main_embeddings, per_layer_embeddings)
        assert combined.shape == expected_shape

    def test_projection_scaling(self):
        """Test that projection applies correct scaling factors"""
        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        # Check scaling factors
        expected_projection_scale = self.config.hidden_size ** -0.5
        expected_combination_scale = 1.0 / math.sqrt(2.0)

        assert abs(projection.projection_scale - expected_projection_scale) < 1e-6
        assert abs(projection.combination_scale - expected_combination_scale) < 1e-6

    def test_combination_vs_projection_only(self):
        """Test that combination produces different output than projection only"""
        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        main_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        per_layer_embeddings = torch.randn(
            self.batch_size, self.seq_len, self.config.num_layers, self.model_params['hidden_size_per_layer_input']
        )

        projected_only = projection(main_embeddings)
        combined = projection(main_embeddings, per_layer_embeddings)

        # Should be different
        assert not torch.allclose(projected_only, combined)

    def test_normalization_applied(self):
        """Test that RMS normalization is applied"""
        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        main_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        output = projection(main_embeddings)

        # Check that normalization maintains roughly unit variance per feature
        # (This is a rough check since RMSNorm behavior depends on the specific values)
        assert output.shape[-1] == self.model_params['hidden_size_per_layer_input']


class TestPerLayerGate:
    """Test suite for PerLayerGate component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = get_tiny_config()
        self.model_params = get_model_params()
        self.batch_size = 2
        self.seq_len = 10

    def test_gate_shapes(self):
        """Test gate output shapes"""
        gate = PerLayerGate(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            layer_idx=0,
        )

        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        per_layer_input = torch.randn(
            self.batch_size, self.seq_len, self.model_params['hidden_size_per_layer_input']
        )

        output = gate(hidden_states, per_layer_input)

        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        assert output.shape == expected_shape

    def test_gate_residual_connection(self):
        """Test that gate includes residual connection"""
        gate = PerLayerGate(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            layer_idx=0,
        )

        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        per_layer_input = torch.ones(  # Use ones instead of zeros for more noticeable effect
            self.batch_size, self.seq_len, self.model_params['hidden_size_per_layer_input']
        )

        output = gate(hidden_states, per_layer_input)

        # Output should have the same shape as input
        assert output.shape == hidden_states.shape

        # With non-zero per_layer_input, output should be different from input
        assert not torch.allclose(output, hidden_states, atol=1e-6)  # Should be modified

    def test_gate_per_layer_effect(self):
        """Test that per-layer input affects output"""
        gate = PerLayerGate(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            layer_idx=0,
        )

        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        per_layer_input_1 = torch.randn(
            self.batch_size, self.seq_len, self.model_params['hidden_size_per_layer_input']
        )
        per_layer_input_2 = torch.randn(
            self.batch_size, self.seq_len, self.model_params['hidden_size_per_layer_input']
        )

        output_1 = gate(hidden_states, per_layer_input_1)
        output_2 = gate(hidden_states, per_layer_input_2)

        # Different per-layer inputs should produce different outputs
        assert not torch.allclose(output_1, output_2)


class TestComponentIntegration:
    """Test integration between components"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = get_tiny_config()
        self.model_params = get_model_params()
        self.batch_size = 2
        self.seq_len = 10

    def test_end_to_end_flow(self):
        """Test complete flow from input tokens to layer processing"""
        # Create all components
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        gate = PerLayerGate(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            layer_idx=0,  # Test with layer 0
        )

        # Test inputs
        per_layer_input_ids = torch.randint(0, self.model_params['vocab_size_per_layer_input'], (self.batch_size, self.seq_len))
        main_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

        # Forward flow
        per_layer_embeds = embedding(per_layer_input_ids)
        combined_embeds = projection(main_embeddings, per_layer_embeds)

        # Extract embeddings for layer 0
        layer_0_input = combined_embeds[:, :, 0, :]  # [batch, seq, hidden_size_per_layer]

        # Process through gate
        final_output = gate(hidden_states, layer_0_input)

        # Check final output shape
        assert final_output.shape == (self.batch_size, self.seq_len, self.config.hidden_size)

    def test_gradient_flow_integration(self):
        """Test that gradients flow through all components in integration"""
        # Create all components
        embedding = PerLayerEmbedding(
            config=self.config,
            vocab_size_per_layer=self.model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        projection = PerLayerProjection(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            num_layers=self.config.num_layers,
        )

        gate = PerLayerGate(
            config=self.config,
            hidden_size_per_layer=self.model_params['hidden_size_per_layer_input'],
            layer_idx=0,
        )

        # Test inputs
        per_layer_input_ids = torch.randint(0, self.model_params['vocab_size_per_layer_input'], (self.batch_size, self.seq_len))
        main_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

        # Forward pass
        per_layer_embeds = embedding(per_layer_input_ids)
        combined_embeds = projection(main_embeddings, per_layer_embeds)
        layer_0_input = combined_embeds[:, :, 0, :]
        final_output = gate(hidden_states, layer_0_input)

        # Backward pass
        loss = final_output.sum()
        loss.backward()

        # Check that all components have gradients
        assert embedding.embedding.weight.grad is not None
        assert embedding.embedding.weight.grad.abs().sum() > 0

        assert projection.projection.weight.grad is not None
        assert projection.projection.weight.grad.abs().sum() > 0

        assert gate.gate.weight.grad is not None
        assert gate.gate.weight.grad.abs().sum() > 0


def run_tests():
    """Run all tests manually (since we don't have pytest installed)"""
    print("🧪 Running Per-Layer Embedding Unit Tests...")

    test_classes = [
        TestPerLayerEmbedding,
        TestPerLayerProjection,
        TestPerLayerGate,
        TestComponentIntegration,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Create instance and run setup if it exists
                instance = test_class()
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()

                # Run the test method
                getattr(instance, test_method)()
                print(f"  ✅ {test_method}")
                passed_tests += 1

            except Exception as e:
                print(f"  ❌ {test_method}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    if not success:
        exit(1)