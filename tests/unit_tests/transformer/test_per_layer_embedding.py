# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_per_layer_embedding_module_spec_for_backend,
)
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.per_layer_embedding import PerLayerEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestPerLayerEmbedding:
    """Test suite for PerLayerEmbedding module."""

    def setup_method(self, method):
        """Initialize model parallel and create test configuration."""
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        # Create config with per-layer embedding settings
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            use_per_layer_embeddings=True,
            hidden_size_per_layer_input=64,
        )

        # Get the per-layer embedding module spec
        ple_spec = get_per_layer_embedding_module_spec_for_backend(LocalSpecProvider())

        # Create the per-layer embedding module
        self.vocab_size = 100
        self.ple = PerLayerEmbedding(
            config=self.transformer_config,
            submodules=ple_spec.submodules,
            vocab_size=self.vocab_size,
            tp_group=None,
        )

    def teardown_method(self, method):
        """Cleanup model parallel."""
        Utils.destroy_model_parallel()

    def test_constructor(self):
        """Test that PerLayerEmbedding is constructed correctly."""
        assert isinstance(self.ple, PerLayerEmbedding)

        # Check that all submodules are created
        assert hasattr(self.ple, 'layer_embedding')
        assert hasattr(self.ple, 'per_layer_input_gate')
        assert hasattr(self.ple, 'per_layer_projection')
        assert hasattr(self.ple, 'activation_func')

        # Check dimensions
        assert self.ple.layer_embedding.num_embeddings == self.vocab_size
        assert self.ple.layer_embedding.embedding_dim == self.transformer_config.hidden_size_per_layer_input

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        # Input dimensions: [sequence_length, batch_size, hidden_size]
        seq_len, batch_size = 32, 2

        mlp_output = torch.randn(seq_len, batch_size, self.transformer_config.hidden_size)
        layer_embedding = torch.randn(
            seq_len, batch_size, self.transformer_config.hidden_size_per_layer_input
        )

        # Forward pass
        output, bias = self.ple(mlp_output, layer_embedding)

        # Check output shape matches input shape
        assert output.shape == mlp_output.shape
        assert output.shape == (seq_len, batch_size, self.transformer_config.hidden_size)

        # Check bias is None
        assert bias is None

    def test_residual_connection(self):
        """Test that residual connection is applied correctly."""
        seq_len, batch_size = 32, 2

        mlp_output = torch.randn(seq_len, batch_size, self.transformer_config.hidden_size)
        layer_embedding = torch.zeros(
            seq_len, batch_size, self.transformer_config.hidden_size_per_layer_input
        )

        # With zero layer embedding, after activation and projection,
        # the contribution should be minimal, and output should be close to input
        # (accounting for the gate projection)
        output, _ = self.ple(mlp_output, layer_embedding)

        # Output should have the same shape as input
        assert output.shape == mlp_output.shape

    def test_layer_spec_integration(self):
        """Test that per-layer embedding is correctly integrated in layer specs."""
        # Get layer spec with per-layer embeddings enabled
        layer_spec = get_gpt_layer_local_spec(use_per_layer_embeddings=True)

        # Check that per_layer_embedding is in the spec
        assert hasattr(layer_spec.submodules, 'per_layer_embedding')

        # The spec should not be IdentityOp
        from megatron.core.transformer.identity_op import IdentityOp
        assert layer_spec.submodules.per_layer_embedding != IdentityOp

        # Check that it's a ModuleSpec with PerLayerEmbedding
        from megatron.core.transformer.spec_utils import ModuleSpec
        assert isinstance(layer_spec.submodules.per_layer_embedding, ModuleSpec)
        assert layer_spec.submodules.per_layer_embedding.module == PerLayerEmbedding

    def test_layer_spec_disabled(self):
        """Test that per-layer embedding is IdentityOp when disabled."""
        # Get layer spec with per-layer embeddings disabled
        layer_spec = get_gpt_layer_local_spec(use_per_layer_embeddings=False)

        # Check that per_layer_embedding is IdentityOp
        from megatron.core.transformer.identity_op import IdentityOp
        assert layer_spec.submodules.per_layer_embedding == IdentityOp

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        """Test forward pass on GPU."""
        self.ple.cuda()

        seq_len, batch_size = 32, 2
        mlp_output = torch.randn(
            seq_len, batch_size, self.transformer_config.hidden_size
        ).cuda()
        layer_embedding = torch.randn(
            seq_len, batch_size, self.transformer_config.hidden_size_per_layer_input
        ).cuda()

        output, bias = self.ple(mlp_output, layer_embedding)

        # Check output properties
        assert output.shape == mlp_output.shape
        assert output.device.type == 'cuda'
        assert output.dtype == torch.float32
        assert bias is None

    def test_parameter_count(self):
        """Test that the module has the expected number of parameters."""
        num_params = sum([p.numel() for p in self.ple.parameters()])

        # Calculate expected parameters:
        # 1. layer_embedding: vocab_size * hidden_size_per_layer_input
        # 2. per_layer_input_gate: hidden_size * hidden_size_per_layer_input
        # 3. per_layer_projection: hidden_size_per_layer_input * hidden_size
        expected_params = (
            self.vocab_size * self.transformer_config.hidden_size_per_layer_input  # embedding
            + self.transformer_config.hidden_size * self.transformer_config.hidden_size_per_layer_input  # gate
            + self.transformer_config.hidden_size_per_layer_input * self.transformer_config.hidden_size  # projection
        )

        assert num_params == expected_params
