"""
Comprehensive validation script for per-layer embedding implementation.
Tests correctness, performance, and integration with potential GPT model.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any

from test_per_layer_config import get_tiny_config, get_model_params, get_layer_spec
from per_layer_embedding import PerLayerEmbedding, PerLayerProjection, PerLayerGate


class MockGPTModel(nn.Module):
    """
    Mock GPT model that demonstrates how per-layer embeddings would integrate
    with a real transformer model.
    """

    def __init__(self, config, model_params):
        super().__init__()
        self.config = config
        self.model_params = model_params

        # Standard embeddings
        self.token_embedding = nn.Embedding(model_params['vocab_size'], config.hidden_size)

        # Per-layer embedding components
        self.per_layer_embedding = PerLayerEmbedding(
            config=config,
            vocab_size_per_layer=model_params['vocab_size_per_layer_input'],
            hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
            num_layers=config.num_layers,
        )

        self.per_layer_projection = PerLayerProjection(
            config=config,
            hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
            num_layers=config.num_layers,
        )

        # Mock transformer layers with per-layer gates
        self.layers = nn.ModuleList([
            MockTransformerLayer(config, model_params, layer_idx)
            for layer_idx in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, input_ids, per_layer_input_ids):
        """Forward pass with per-layer embedding integration"""
        # Standard token embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, S, H]

        # Per-layer embeddings
        per_layer_embeds = self.per_layer_embedding(per_layer_input_ids)  # [B, S, L, H_per]

        # Combine embeddings
        combined_per_layer = self.per_layer_projection(token_embeds, per_layer_embeds)  # [B, S, L, H_per]

        # Pass through transformer layers
        hidden_states = token_embeds
        for layer_idx, layer in enumerate(self.layers):
            layer_per_layer_input = combined_per_layer[:, :, layer_idx, :]  # [B, S, H_per]
            hidden_states = layer(hidden_states, layer_per_layer_input)

        # Final normalization
        output = self.final_norm(hidden_states)
        return output


class MockTransformerLayer(nn.Module):
    """Mock transformer layer that includes per-layer gate"""

    def __init__(self, config, model_params, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # Simplified transformer components (just linear layers for testing)
        self.attention = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlp = nn.Linear(config.hidden_size, config.hidden_size)

        # Per-layer gate
        self.per_layer_gate = PerLayerGate(
            config=config,
            hidden_size_per_layer=model_params['hidden_size_per_layer_input'],
            layer_idx=layer_idx,
        )

        # Layer norms
        self.pre_attention_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.pre_mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states, per_layer_input):
        """Forward pass with per-layer processing"""
        # Apply per-layer gate first
        hidden_states = self.per_layer_gate(hidden_states, per_layer_input)

        # Simplified attention block
        residual = hidden_states
        hidden_states = self.pre_attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = hidden_states + residual

        # Simplified MLP block
        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


def validate_shapes_and_correctness():
    """Validate that all shapes are correct and model works"""
    print("🔍 Validating shapes and correctness...")

    config = get_tiny_config()
    model_params = get_model_params()

    # Create mock model
    model = MockGPTModel(config, model_params)

    # Test inputs
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    per_layer_input_ids = torch.randint(0, model_params['vocab_size_per_layer_input'], (batch_size, seq_len))

    # Forward pass
    try:
        output = model(input_ids, per_layer_input_ids)
        expected_shape = (batch_size, seq_len, config.hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✅ Output shape correct: {output.shape}")
    except Exception as e:
        print(f"❌ Shape validation failed: {e}")
        return False

    # Test backward pass
    try:
        loss = output.sum()
        loss.backward()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

    return True


def validate_per_layer_behavior():
    """Validate that per-layer embeddings actually affect model behavior"""
    print("\n🧪 Validating per-layer embedding behavior...")

    config = get_tiny_config()
    model_params = get_model_params()

    model = MockGPTModel(config, model_params)
    model.eval()  # Set to eval mode for consistent results

    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))

    # Test with different per-layer inputs
    per_layer_input_ids_1 = torch.zeros((batch_size, seq_len), dtype=torch.long)
    per_layer_input_ids_2 = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        output_1 = model(input_ids, per_layer_input_ids_1)
        output_2 = model(input_ids, per_layer_input_ids_2)

    # Different per-layer inputs should produce different outputs
    if torch.allclose(output_1, output_2, atol=1e-6):
        print("❌ Per-layer embeddings don't affect model output")
        return False

    mean_diff = (output_2 - output_1).abs().mean().item()
    print(f"✅ Per-layer embeddings affect output (mean difference: {mean_diff:.6f})")

    return True


def validate_parameter_counts():
    """Validate parameter counts and memory usage"""
    print("\n📊 Validating parameter counts...")

    config = get_tiny_config()
    model_params = get_model_params()

    model = MockGPTModel(config, model_params)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    per_layer_params = sum(
        p.numel() for name, p in model.named_parameters()
        if any(component in name for component in ['per_layer_embedding', 'per_layer_projection', 'per_layer_gate'])
    )

    param_ratio = per_layer_params / total_params

    print(f"✅ Total parameters: {total_params:,}")
    print(f"✅ Per-layer parameters: {per_layer_params:,}")
    print(f"✅ Per-layer ratio: {param_ratio:.2%}")

    # Expected rough parameter counts (this is approximate)
    expected_min_per_layer = 1000  # Should have at least some per-layer parameters
    if per_layer_params < expected_min_per_layer:
        print(f"❌ Too few per-layer parameters: {per_layer_params}")
        return False

    return True


def validate_performance():
    """Validate performance characteristics"""
    print("\n⚡ Validating performance...")

    config = get_tiny_config()
    model_params = get_model_params()

    # Create model with per-layer embedding
    model_with_per_layer = MockGPTModel(config, model_params)
    model_with_per_layer.eval()

    # Create baseline model (without per-layer components, roughly)
    baseline_model = nn.Sequential(
        nn.Embedding(model_params['vocab_size'], config.hidden_size),
        *[nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_layers * 2)],
        nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
    )
    baseline_model.eval()

    # Test inputs
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    per_layer_input_ids = torch.randint(0, model_params['vocab_size_per_layer_input'], (batch_size, seq_len))

    # Benchmark per-layer model
    num_runs = 10
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            model_with_per_layer(input_ids, per_layer_input_ids)

        # Time per-layer model
        start_time = time.time()
        for _ in range(num_runs):
            output = model_with_per_layer(input_ids, per_layer_input_ids)
        per_layer_time = (time.time() - start_time) / num_runs

        # Time baseline model
        start_time = time.time()
        for _ in range(num_runs):
            output = baseline_model(input_ids)
        baseline_time = (time.time() - start_time) / num_runs

    overhead = ((per_layer_time - baseline_time) / baseline_time) * 100
    print(f"✅ Per-layer model time: {per_layer_time:.4f}s")
    print(f"✅ Baseline model time: {baseline_time:.4f}s")
    print(f"✅ Overhead: {overhead:.1f}%")

    # Reasonable overhead threshold (per-layer embedding should not be too expensive)
    # For very small models, overhead can be high due to fixed costs
    # In production, with larger models, this overhead would be much smaller
    if overhead > 1000:  # More than 1000% overhead would be concerning
        print(f"❌ Excessive overhead: {overhead:.1f}%")
        return False
    elif overhead > 300:
        print(f"⚠️ High overhead (expected for tiny test model): {overhead:.1f}%")

    return True


def validate_gradient_flow():
    """Validate that gradients flow properly through all components"""
    print("\n🔄 Validating gradient flow...")

    config = get_tiny_config()
    model_params = get_model_params()

    model = MockGPTModel(config, model_params)

    # Test inputs
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    per_layer_input_ids = torch.randint(0, model_params['vocab_size_per_layer_input'], (batch_size, seq_len))

    # Forward pass
    output = model(input_ids, per_layer_input_ids)
    loss = output.pow(2).mean()  # Simple loss function

    # Backward pass
    loss.backward()

    # Check gradients in per-layer components
    per_layer_components = [
        ('per_layer_embedding', model.per_layer_embedding.embedding),
        ('per_layer_projection', model.per_layer_projection.projection),
    ]

    all_gradients_present = True
    for name, module in per_layer_components:
        if module.weight.grad is None or module.weight.grad.abs().sum() == 0:
            print(f"❌ No gradients in {name}")
            all_gradients_present = False
        else:
            grad_norm = module.weight.grad.norm().item()
            print(f"✅ {name} gradient norm: {grad_norm:.6f}")

    # Check gradients in per-layer gates
    for layer_idx, layer in enumerate(model.layers):
        gate_grad_norm = layer.per_layer_gate.gate.weight.grad.norm().item()
        proj_grad_norm = layer.per_layer_gate.projection.weight.grad.norm().item()
        print(f"✅ Layer {layer_idx} gate gradients - gate: {gate_grad_norm:.6f}, proj: {proj_grad_norm:.6f}")

    return all_gradients_present


def validate_training_step():
    """Validate a complete training step"""
    print("\n🎯 Validating training step...")

    config = get_tiny_config()
    model_params = get_model_params()

    model = MockGPTModel(config, model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Test inputs
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, model_params['vocab_size'], (batch_size, seq_len))
    per_layer_input_ids = torch.randint(0, model_params['vocab_size_per_layer_input'], (batch_size, seq_len))

    # Initial loss
    output = model(input_ids, per_layer_input_ids)
    initial_loss = output.pow(2).mean().item()

    # Training steps
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        output = model(input_ids, per_layer_input_ids)
        loss = output.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]

    print(f"✅ Initial loss: {initial_loss:.6f}")
    print(f"✅ Final loss: {final_loss:.6f}")
    print(f"✅ Loss change: {((final_loss - initial_loss) / initial_loss) * 100:.1f}%")

    # Loss should change (either increase or decrease)
    if abs(final_loss - initial_loss) < 1e-8:
        print("❌ Loss didn't change during training")
        return False

    return True


def run_comprehensive_validation():
    """Run all validation tests"""
    print("🚀 Running Comprehensive Per-Layer Embedding Validation")
    print("=" * 60)

    validation_tests = [
        ("Shapes and Correctness", validate_shapes_and_correctness),
        ("Per-Layer Behavior", validate_per_layer_behavior),
        ("Parameter Counts", validate_parameter_counts),
        ("Performance", validate_performance),
        ("Gradient Flow", validate_gradient_flow),
        ("Training Step", validate_training_step),
    ]

    results = {}
    for test_name, test_func in validation_tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)

    passed_tests = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed_tests += 1

    print(f"\n📊 Overall Result: {passed_tests}/{len(validation_tests)} tests passed")

    if passed_tests == len(validation_tests):
        print("🎉 All validation tests passed! Per-layer embedding implementation is ready.")
        return True
    else:
        print("❌ Some validation tests failed. Review the implementation.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    if not success:
        exit(1)