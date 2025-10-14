# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for RecursiveGPTModel with iterative refinement."""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.recursive_gpt import RecursiveGPTModel, RecursiveTransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


class TestRecursiveGPTModel:
    """Test suite for RecursiveGPTModel."""

    def setup_method(self, method):
        """Set up test environment before each test."""
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        # Create RecursiveTransformerConfig
        self.config = RecursiveTransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            num_refinement_blocks=3,
            num_latent_refinements=2,
            halt_loss_weight=1.0,
            enable_recursive_refinement=True,
            detach_between_refinements=True,
        )

        # Create base GPTModel
        self.gpt_model = GPTModel(
            config=self.config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

        # Create RecursiveGPTModel
        self.recursive_model = RecursiveGPTModel(
            gpt_model=self.gpt_model, config=self.config
        )

    def teardown_method(self, method):
        """Clean up test environment after each test."""
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        """Test that RecursiveGPTModel is constructed correctly."""
        assert isinstance(self.recursive_model, RecursiveGPTModel)
        assert self.recursive_model.gpt == self.gpt_model
        assert self.recursive_model.config == self.config
        assert self.recursive_model.num_refinement_blocks == 3
        assert self.recursive_model.num_latent_refinements == 2

    @pytest.mark.internal
    def test_initial_state_shape(self):
        """Test that initial state has correct shape."""
        batch_size = 2
        seq_length = 4
        hidden_size = self.config.hidden_size

        self.recursive_model.cuda()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()

        # Get initial state
        outputs, latents = self.recursive_model.get_initial_state(
            input_ids=input_ids, position_ids=position_ids, attention_mask=None
        )

        # Check shapes: [S, B, H]
        assert outputs.shape == (seq_length, batch_size, hidden_size)
        assert latents.shape == (seq_length, batch_size, hidden_size)

        # Check that states are initialized to zeros
        assert torch.allclose(outputs, torch.zeros_like(outputs))
        assert torch.allclose(latents, torch.zeros_like(latents))

    @pytest.mark.internal
    def test_forward_training(self):
        """Test forward pass during training with labels."""
        batch_size = 2
        seq_length = 4

        self.recursive_model.cuda()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()

        # Create dummy labels
        labels = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.int64).cuda()

        # Forward pass
        result = self.recursive_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'logits' in result

        # Check loss shape and type
        loss = result['loss']
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar loss

        # Check logits shape: [B, S, V]
        logits = result['logits']
        assert logits.shape == (batch_size, seq_length, self.gpt_model.vocab_size)

    @pytest.mark.internal
    def test_forward_inference(self):
        """Test forward pass during inference without labels."""
        batch_size = 2
        seq_length = 4

        self.recursive_model.cuda()
        self.recursive_model.eval()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()

        # Forward pass without labels
        with torch.no_grad():
            logits = self.recursive_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,
            )

        # Check logits shape: [B, S, V]
        assert logits.shape == (batch_size, seq_length, self.gpt_model.vocab_size)

    @pytest.mark.internal
    def test_disable_recursive_refinement(self):
        """Test that model behaves like standard GPT when refinement is disabled."""
        # Create config with recursive refinement disabled
        config_no_recursive = RecursiveTransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_recursive_refinement=False,
        )

        gpt_model = GPTModel(
            config=config_no_recursive,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

        recursive_model = RecursiveGPTModel(gpt_model=gpt_model, config=config_no_recursive)

        batch_size = 2
        seq_length = 4

        recursive_model.cuda()
        recursive_model.eval()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()

        # Forward pass should work like standard GPT
        with torch.no_grad():
            logits = recursive_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,
            )

        assert logits.shape == (batch_size, seq_length, gpt_model.vocab_size)

    @pytest.mark.internal
    def test_generate_with_adaptive_halting(self):
        """Test generation with adaptive halting."""
        batch_size = 2
        seq_length = 4

        self.recursive_model.cuda()
        self.recursive_model.eval()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()

        # Test generation
        with torch.no_grad():
            result = self.recursive_model.generate(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        # Check result dictionary
        assert isinstance(result, dict)
        assert 'logits' in result
        assert 'refinement_steps' in result

        # Check logits shape
        logits = result['logits']
        assert logits.shape == (batch_size, seq_length, self.gpt_model.vocab_size)

        # Check refinement steps (should be a tensor with step count for each sample)
        refinement_steps = result['refinement_steps']
        assert isinstance(refinement_steps, torch.Tensor)
        assert refinement_steps.shape == (batch_size,)
        assert torch.all(refinement_steps >= 0)
        assert torch.all(
            refinement_steps <= self.config.max_inference_refinement_steps
        )

    @pytest.mark.internal
    def test_gradient_flow_with_detach(self):
        """Test that gradients flow correctly with detach_between_refinements=True."""
        batch_size = 2
        seq_length = 4

        self.recursive_model.cuda()
        self.recursive_model.train()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.int64).cuda()

        # Forward pass
        result = self.recursive_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = result['loss']

        # Backward pass
        loss.backward()

        # Check that gradients exist for model parameters
        for name, param in self.recursive_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.internal
    def test_gradient_flow_without_detach(self):
        """Test that gradients flow through all refinements when detach=False."""
        # Create model with detach_between_refinements=False
        config_no_detach = RecursiveTransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            num_refinement_blocks=3,
            num_latent_refinements=2,
            detach_between_refinements=False,
        )

        gpt_model = GPTModel(
            config=config_no_detach,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

        recursive_model = RecursiveGPTModel(gpt_model=gpt_model, config=config_no_detach)

        batch_size = 2
        seq_length = 4

        recursive_model.cuda()
        recursive_model.train()

        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        labels = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.int64).cuda()

        # Forward pass
        result = recursive_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = result['loss']

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in recursive_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.internal
    def test_config_validation(self):
        """Test that config validation catches invalid parameters."""
        # Test num_refinement_blocks < 1
        with pytest.raises(ValueError, match="num_refinement_blocks must be >= 1"):
            RecursiveTransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                num_refinement_blocks=0,
            )

        # Test num_latent_refinements < 1
        with pytest.raises(ValueError, match="num_latent_refinements must be >= 1"):
            RecursiveTransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                num_latent_refinements=0,
            )

        # Test halt_loss_weight < 0
        with pytest.raises(ValueError, match="halt_loss_weight must be >= 0"):
            RecursiveTransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                halt_loss_weight=-1.0,
            )

        # Test invalid halt_threshold
        with pytest.raises(ValueError, match="halt_threshold must be in"):
            RecursiveTransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                halt_threshold=1.5,
            )

        # Test invalid max_inference_refinement_steps
        with pytest.raises(ValueError, match="max_inference_refinement_steps must be >= 1"):
            RecursiveTransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                max_inference_refinement_steps=0,
            )


@pytest.mark.parametrize("num_refinement_blocks", [1, 3, 5])
@pytest.mark.parametrize("num_latent_refinements", [1, 2, 4])
def test_recursive_gpt_with_different_refinement_params(
    num_refinement_blocks, num_latent_refinements
):
    """Test RecursiveGPTModel with different refinement parameters."""
    # Setup
    os.environ.pop('NVTE_FUSED_ATTN', None)
    os.environ.pop('NVTE_FLASH_ATTN', None)
    os.environ.pop('NVTE_UNFUSED_ATTN', None)
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)

    config = RecursiveTransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        num_refinement_blocks=num_refinement_blocks,
        num_latent_refinements=num_latent_refinements,
    )

    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=100,
        max_sequence_length=4,
    )

    recursive_model = RecursiveGPTModel(gpt_model=gpt_model, config=config)

    # Test forward pass
    batch_size = 2
    seq_length = 4

    recursive_model.cuda()

    data = list(range(seq_length))
    input_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
    position_ids = torch.tensor(data, dtype=torch.int64).repeat((batch_size, 1)).cuda()
    attention_mask = torch.ones(
        (batch_size, 1, seq_length, seq_length), dtype=bool
    ).cuda()
    labels = torch.randint(0, 100, (batch_size, seq_length), dtype=torch.int64).cuda()

    result = recursive_model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    # Check outputs
    assert 'loss' in result
    assert 'logits' in result
    assert result['logits'].shape == (batch_size, seq_length, gpt_model.vocab_size)

    # Teardown
    Utils.destroy_model_parallel()
