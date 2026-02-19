# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from typing import Optional, Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import (
    BackendSpecProvider,
    InferenceSpecProvider,
    LocalSpecProvider,
)
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.embedding_mixer import EmbeddingMixer, EmbeddingMixerSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlockSubmodules,
    get_mtp_layer_offset,
    get_mtp_layer_spec_for_backend,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import is_te_min_version

try:
    import transformer_engine as te  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEFusedMLP, TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import nvidia_kitchen  # type: ignore[import-not-found]  # pylint: disable=unused-import

    from megatron.core.extensions.kitchen import KitchenSpecProvider

    HAVE_KITCHEN = True
except ImportError:
    HAVE_KITCHEN = False

try:
    import apex  # type: ignore[import-untyped]  # pylint: disable=unused-import

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    warnings.warn("Apex is not installed. Falling back to Torch Norm")
    LNImpl = WrappedTorchNorm
    HAVE_APEX = False


def _get_embedding_mixer_spec(config, backend: BackendSpecProvider) -> Union[ModuleSpec, type]:
    """Get EmbeddingMixer spec if configured, else IdentityOp.

    Args:
        config: TransformerConfig instance
        backend: BackendSpecProvider instance

    Returns:
        ModuleSpec for EmbeddingMixer if embedding_mixer_latent_size is set, else IdentityOp
    """
    if config.embedding_mixer_latent_size is None:
        return IdentityOp

    return ModuleSpec(
        module=EmbeddingMixer,
        submodules=EmbeddingMixerSubmodules(
            down_proj=backend.column_parallel_linear(),
            up_proj=backend.row_parallel_linear(),
        ),
    )


def get_gpt_layer_with_inference_spec(
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    config: Optional['TransformerConfig'] = None,
) -> ModuleSpec:
    """Use this spec to use inference optimized linear layers.
    Args:
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        multi_latent_attention (bool, optional): To use MLA. Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
    """
    assert HAVE_TE, "--transformer-impl inference_optimized requires transformer engine"
    backend = InferenceSpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=None,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
        use_te_op_fuser=False,
        use_te_activation_func=False,
    )

    # Resolve embedding mixer spec from config (if provided and enabled).
    emb_mixer_spec = _get_embedding_mixer_spec(config, backend) if config else IdentityOp
    pre_emb_mixer_ln = backend.layer_norm() if isinstance(emb_mixer_spec, ModuleSpec) else IdentityOp
    emb_mixer_bda_spec = get_bias_dropout_add if isinstance(emb_mixer_spec, ModuleSpec) else IdentityFuncOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=backend.layer_norm(),
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.linear(),
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_down_proj=backend.linear(),
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
            ),
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_layer_norm_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
                sharded_state_dict_keys_map={
                    "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                    "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                    "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                    "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                    "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                    "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
                },
            ),
        )


def get_gpt_layer_with_transformer_engine_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    qk_l2_norm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_kitchen: bool = False,
    use_te_activation_func: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    config: Optional['TransformerConfig'] = None,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        assert HAVE_KITCHEN
        backend: BackendSpecProvider = KitchenSpecProvider(
            fallback=TESpecProvider(),
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
        if use_te_op_fuser:
            raise AssertionError("use_te_op_fuser not compatible with using kitchen in mlp.")
        if use_te_activation_func:
            raise AssertionError("use_te_activation_func not compatible with using kitchen.")
    else:
        backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
        use_te_activation_func=use_te_activation_func,
    )

    # Resolve embedding mixer spec from config (if provided and enabled).
    emb_mixer_spec = _get_embedding_mixer_spec(config, backend) if config else IdentityOp
    pre_emb_mixer_ln = backend.layer_norm() if isinstance(emb_mixer_spec, ModuleSpec) else IdentityOp
    emb_mixer_bda_spec = get_bias_dropout_add if isinstance(emb_mixer_spec, ModuleSpec) else IdentityFuncOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        linear_q_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        linear_kv_up_proj = (
            backend.column_parallel_layer_norm_linear()
            if qk_layernorm
            else backend.column_parallel_linear()
        )
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=backend.layer_norm(),
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.linear(),
                        linear_q_up_proj=linear_q_up_proj,
                        linear_kv_down_proj=backend.linear(),
                        linear_kv_up_proj=linear_kv_up_proj,
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=IdentityOp,
                        kv_layernorm=IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
            ),
        )
    else:
        qk_norm = backend.layer_norm(for_qk=True)
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_layer_norm_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
                sharded_state_dict_keys_map={
                    "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                    "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                    "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                    "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                    "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                    "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
                },
            ),
        )


def get_gpt_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    multi_latent_attention: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    use_kitchen: bool = False,
    use_kitchen_attention: bool = False,
    kitchen_attention_backend: str = "sdpa",
    config: Optional['TransformerConfig'] = None,
) -> ModuleSpec:
    """Use this spec for an implementation using only modules in Megatron-Core.


    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.

    Returns:
        ModuleSpec: Module specification with Megatron-Core modules
    """

    if use_kitchen:
        assert HAVE_KITCHEN
        backend = KitchenSpecProvider(
            fallback=LocalSpecProvider(),
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
    else:
        backend = LocalSpecProvider()
    # Adjust for RMS norm.
    if normalization == "RMSNorm":
        layer_norm = backend.layer_norm(rms_norm=True, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=True, for_qk=True)
    else:
        layer_norm = backend.layer_norm(rms_norm=False, for_qk=False)
        qk_norm = backend.layer_norm(rms_norm=False, for_qk=True)

    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_local_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    # Resolve embedding mixer spec from config (if provided and enabled).
    emb_mixer_spec = _get_embedding_mixer_spec(config, backend) if config else IdentityOp
    pre_emb_mixer_ln = layer_norm if isinstance(emb_mixer_spec, ModuleSpec) else IdentityOp
    emb_mixer_bda_spec = get_bias_dropout_add if isinstance(emb_mixer_spec, ModuleSpec) else IdentityFuncOp

    if multi_latent_attention:
        assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=MLASelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=MLASelfAttentionSubmodules(
                        linear_q_proj=backend.column_parallel_linear(),
                        linear_q_down_proj=backend.column_parallel_linear(),
                        linear_q_up_proj=backend.column_parallel_linear(),
                        linear_kv_down_proj=backend.column_parallel_linear(),
                        linear_kv_up_proj=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=qk_norm if qk_layernorm else IdentityOp,
                        kv_layernorm=qk_norm if qk_layernorm else IdentityOp,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
            ),
        )
    else:
        return ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=layer_norm,
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=backend.column_parallel_linear(),
                        core_attention=backend.core_attention(),
                        linear_proj=backend.row_parallel_linear(),
                        q_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                        k_layernorm=(
                            L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                        ),
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=layer_norm,
                mlp=mlp,
                mlp_bda=get_bias_dropout_add,
                pre_emb_mixer_layernorm=pre_emb_mixer_ln,
                embedding_mixer=emb_mixer_spec,
                emb_mixer_bda=emb_mixer_bda_spec,
                sharded_state_dict_keys_map={
                    "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                    "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                },
            ),
        )


def _get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
):
    warnings.warn(
        """This private function is on a deprecation track. Please switch to `get_mlp_module_spec`
        since it will be removed in a future release."""
    )

    return get_mlp_module_spec(
        use_te=use_te,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        fp8=fp8,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def get_mlp_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    fp8: Optional[str] = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "_get_mlp_module_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )
    if use_te_op_fuser:
        if not is_te_min_version("1.13.0"):
            raise ValueError(
                "Transformer Engine operation-based API requires Transformer Engine 1.13+"
            )
        if num_experts is not None:
            raise ValueError(
                "Transformer Engine operation-based API does not support mixture-of-experts"
            )

    return get_mlp_module_spec_for_backend(
        backend=TESpecProvider() if use_te else LocalSpecProvider(),
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
    )


def get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_op_fuser: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE"""

    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func() if use_te_activation_func else None

    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        module = TEFusedMLP if use_te_op_fuser else MLP
        if backend.fuse_layernorm_and_linear():
            linear_fc1 = backend.column_parallel_layer_norm_linear()
            assert linear_fc1 is not None
        else:
            linear_fc1 = backend.column_parallel_linear()
        return ModuleSpec(
            module=module,
            submodules=MLPSubmodules(
                linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        return get_moe_module_spec_for_backend(
            backend=backend,
            num_experts=num_experts,
            moe_grouped_gemm=moe_grouped_gemm,
            moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
            use_te_activation_func=use_te_activation_func,
        )


def _get_layer_freq_pattern(freq, num_layers, name):
    """Parse an int or list layer frequency into a per-layer 0/1 pattern."""
    if isinstance(freq, int):
        return [1 if (i % freq == 0) else 0 for i in range(num_layers)]
    elif isinstance(freq, list):
        assert len(freq) == num_layers, (
            f"Invalid length of {name}: {len(freq)}, expected {num_layers}, "
            f"current pattern: {freq}"
        )
        return freq
    else:
        raise ValueError(f"Invalid {name}: {type(freq)}, {freq}")


def get_gpt_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: Optional[str] = None,
    qk_l2_norm: Optional[bool] = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """GPT block spec."""

    # Determine whether any layer needs an embedding mixer.
    emb_mixer_freq = config.embedding_mixer_layer_freq
    emb_mixer_pattern = _get_layer_freq_pattern(
        emb_mixer_freq, config.num_layers, "embedding_mixer_layer_freq"
    )
    need_mixer = any(p == 1 for p in emb_mixer_pattern) and (
        config.embedding_mixer_latent_size is not None
    )
    need_no_mixer = any(p == 0 for p in emb_mixer_pattern) or not need_mixer

    def _build_spec(spec_func, mixer_config, **kwargs):
        return spec_func(**kwargs, config=mixer_config)

    if use_transformer_engine:
        layer_norm_impl = TENorm
        common_te = dict(
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
        dense_kwargs = dict(num_experts=None, moe_grouped_gemm=False, **common_te)
        moe_kwargs = dict(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            **common_te,
        )
        spec_func = get_gpt_layer_with_transformer_engine_spec
    else:
        layer_norm_impl = LNImpl
        common_local = dict(
            qk_layernorm=config.qk_layernorm,
            multi_latent_attention=config.multi_latent_attention,
            moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
            normalization=normalization,
            qk_l2_norm=qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
        dense_kwargs = dict(num_experts=None, moe_grouped_gemm=False, **common_local)
        moe_kwargs = dict(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=config.moe_grouped_gemm,
            **common_local,
        )
        spec_func = get_gpt_layer_local_spec

    # Build up to 4 variants: {dense, moe} × {with_mixer, no_mixer}
    # Only build variants that are actually needed.
    dense_mixer = _build_spec(spec_func, config, **dense_kwargs) if need_mixer else None
    dense_no_mixer = _build_spec(spec_func, None, **dense_kwargs) if need_no_mixer else None
    moe_mixer = _build_spec(spec_func, config, **moe_kwargs) if need_mixer else None
    moe_no_mixer = _build_spec(spec_func, None, **moe_kwargs) if need_no_mixer else None

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    moe_layer_pattern = _get_layer_freq_pattern(
        config.moe_layer_freq, config.num_layers, "moe_layer_freq"
    )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        is_moe = moe_layer_pattern[layer_number] == 1
        has_mixer = need_mixer and emb_mixer_pattern[layer_number] == 1
        if is_moe and has_mixer:
            layer_specs.append(moe_mixer)
        elif is_moe:
            layer_specs.append(moe_no_mixer)
        elif has_mixer:
            layer_specs.append(dense_mixer)
        elif not is_moe:
            layer_specs.append(dense_no_mixer)
        else:
            raise ValueError(
                f"Invalid layer pattern at layer {layer_number}: "
                f"moe={moe_layer_pattern[layer_number]}, "
                f"mixer={emb_mixer_pattern[layer_number]}"
            )

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    if config.pipeline_model_parallel_layout is not None:
        layout = config.pipeline_model_parallel_layout
        assert isinstance(layout, PipelineParallelLayerLayout)
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def get_gpt_mtp_block_spec(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    use_transformer_engine: bool,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    if use_transformer_engine:
        backend: BackendSpecProvider = (
            KitchenSpecProvider(
                fallback=TESpecProvider(),
                use_kitchen_attention=config.use_kitchen_attention,
                kitchen_attention_backend=config.kitchen_attention_backend,
            )
            if config.use_kitchen
            else TESpecProvider()
        )
    else:
        backend = (
            KitchenSpecProvider(
                fallback=LocalSpecProvider(),
                use_kitchen_attention=config.use_kitchen_attention,
                kitchen_attention_backend=config.kitchen_attention_backend,
            )
            if config.use_kitchen
            else LocalSpecProvider()
        )
    return get_gpt_mtp_block_spec_for_backend(
        config=config, spec=spec, backend=backend, vp_stage=vp_stage, pp_rank=pp_rank
    )


def get_gpt_mtp_block_spec_for_backend(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
    backend: BackendSpecProvider,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> MultiTokenPredictionBlockSubmodules:
    """GPT Multi-Token Prediction (MTP) block spec."""
    num_layers_to_build = get_mtp_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    if num_layers_to_build == 0:
        return None

    if isinstance(spec, TransformerBlockSubmodules):
        # get the spec for the last layer of decoder block
        transformer_layer_spec = spec.layer_specs[-1]
    elif isinstance(spec, ModuleSpec) and spec.module == TransformerLayer:
        transformer_layer_spec = spec
    else:
        raise ValueError(f"Invalid spec: {spec}")

    mtp_layer_spec = get_mtp_layer_spec_for_backend(
        transformer_layer_spec=transformer_layer_spec, backend=backend
    )
    mtp_num_layers = config.mtp_num_layers if config.mtp_num_layers else 0
    mtp_layer_specs = [mtp_layer_spec] * mtp_num_layers

    offset = get_mtp_layer_offset(config)
    # split the mtp layer specs to only include the layers that are built in this pipeline stage.
    mtp_layer_specs = mtp_layer_specs[offset : offset + num_layers_to_build]
    if len(mtp_layer_specs) > 0:
        assert (
            len(mtp_layer_specs) == config.mtp_num_layers
        ), f"currently all of the mtp layers must stage in the same pipeline stage."
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(layer_specs=mtp_layer_specs)
    else:
        mtp_block_spec = None

    return mtp_block_spec
