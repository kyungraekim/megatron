# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    _get_embedding_mixer_spec,
    _get_layer_freq_pattern,
    get_gpt_layer_local_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.embedding_mixer import EmbeddingMixer, EmbeddingMixerSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class TestEmbeddingMixerSpec:
    """Tests for spec-level helpers (no GPU required beyond init)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_get_embedding_mixer_spec_disabled(self):
        """When latent_size is None, spec should return IdentityOp."""
        from megatron.core.models.backends import LocalSpecProvider

        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            embedding_mixer_latent_size=None,
        )
        backend = LocalSpecProvider()
        spec = _get_embedding_mixer_spec(config, backend)
        assert spec is IdentityOp

    def test_get_embedding_mixer_spec_enabled(self):
        """When latent_size is set, spec should return a ModuleSpec for EmbeddingMixer."""
        from megatron.core.models.backends import LocalSpecProvider

        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            embedding_mixer_latent_size=8,
        )
        backend = LocalSpecProvider()
        spec = _get_embedding_mixer_spec(config, backend)
        assert isinstance(spec, ModuleSpec)
        assert spec.module is EmbeddingMixer

    def test_layer_spec_with_config_wires_mixer(self):
        """get_gpt_layer_local_spec with config should produce a real embedding mixer spec."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            embedding_mixer_latent_size=8,
            embedding_mixer_num_embeddings=4,
            embedding_mixer_topk=2,
        )
        spec = get_gpt_layer_local_spec(config=config)
        assert isinstance(spec.submodules.embedding_mixer, ModuleSpec)
        assert spec.submodules.embedding_mixer.module is EmbeddingMixer

    def test_layer_spec_without_config_uses_identity(self):
        """get_gpt_layer_local_spec without config should leave IdentityOp."""
        spec = get_gpt_layer_local_spec(config=None)
        assert spec.submodules.embedding_mixer is IdentityOp


class TestLayerFreqPattern:
    """Tests for _get_layer_freq_pattern helper."""

    def test_int_every_layer(self):
        assert _get_layer_freq_pattern(1, 4, "test") == [1, 1, 1, 1]

    def test_int_every_other_layer(self):
        assert _get_layer_freq_pattern(2, 4, "test") == [1, 0, 1, 0]

    def test_int_every_third_layer(self):
        assert _get_layer_freq_pattern(3, 6, "test") == [1, 0, 0, 1, 0, 0]

    def test_list_passthrough(self):
        pattern = [1, 0, 1, 0]
        assert _get_layer_freq_pattern(pattern, 4, "test") == pattern

    def test_list_length_mismatch(self):
        with pytest.raises(AssertionError):
            _get_layer_freq_pattern([1, 0], 4, "test")

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            _get_layer_freq_pattern("bad", 4, "test")


class TestEmbeddingMixerModule:
    """Tests for the EmbeddingMixer module forward pass."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_mixer(self, hidden_size=16, latent_size=8, num_embeddings=4, topk=2, **kwargs):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=2,
            embedding_mixer_latent_size=latent_size,
            embedding_mixer_num_embeddings=num_embeddings,
            embedding_mixer_topk=topk,
            use_cpu_initialization=True,
            **kwargs,
        )
        from megatron.core.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()
        spec = _get_embedding_mixer_spec(config, backend)
        assert isinstance(spec, ModuleSpec)
        mixer = build_module(spec, config=config)
        return mixer, config

    def test_forward_shape(self):
        """Output shape should match input hidden_states shape [s, b, h]."""
        mixer, config = self._make_mixer()
        mixer.cuda()
        s, b, h = 8, 2, config.hidden_size
        hidden_states = torch.randn(s, b, h, device='cuda')
        output = mixer(hidden_states)
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output
        assert out_tensor.shape == (s, b, h)

    def test_forward_gradient_flows(self):
        """Gradients should flow back through the mixer."""
        mixer, config = self._make_mixer()
        mixer.cuda()
        s, b, h = 4, 1, config.hidden_size
        hidden_states = torch.randn(s, b, h, device='cuda', requires_grad=True)
        output = mixer(hidden_states)
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output
        loss = out_tensor.sum()
        loss.backward()
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

    def test_topk_indices_computed_once(self):
        """Smoke test: forward should work with aux loss enabled."""
        mixer, config = self._make_mixer(
            embedding_mixer_aux_loss_coeff=0.01,
            embedding_mixer_z_loss_coeff=0.001,
        )
        mixer.cuda()
        mixer.train()
        s, b, h = 4, 2, config.hidden_size
        hidden_states = torch.randn(s, b, h, device='cuda')
        output = mixer(hidden_states)
        if isinstance(output, tuple):
            out_tensor = output[0]
        else:
            out_tensor = output
        assert out_tensor.shape == (s, b, h)

    def test_latent_size_none_raises(self):
        """EmbeddingMixer should raise if latent_size is None."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            embedding_mixer_latent_size=None,
        )
        from megatron.core.models.backends import LocalSpecProvider

        backend = LocalSpecProvider()
        submodules = EmbeddingMixerSubmodules(
            down_proj=backend.column_parallel_linear(),
            up_proj=backend.row_parallel_linear(),
        )
        with pytest.raises(ValueError, match="embedding_mixer_latent_size must not be None"):
            EmbeddingMixer(config=config, submodules=submodules)


class TestEmbeddingMixerInTransformerLayer:
    """Integration test: EmbeddingMixer wired into a TransformerLayer."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_transformer_layer_with_mixer(self):
        """A TransformerLayer with embedding mixer should produce valid output."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            embedding_mixer_latent_size=8,
            embedding_mixer_num_embeddings=4,
            embedding_mixer_topk=2,
            use_cpu_initialization=True,
        )
        spec = get_gpt_layer_local_spec(config=config)
        layer = TransformerLayer(config, spec.submodules)
        layer.cuda()

        s, b = 8, 2
        hidden_states = torch.randn(s, b, config.hidden_size, device='cuda')
        attention_mask = torch.ones((1, 1, s, s), dtype=bool, device='cuda')

        output, context = layer(hidden_states=hidden_states, attention_mask=attention_mask)
        assert output.shape == (s, b, config.hidden_size)
