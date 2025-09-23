"""
Argument parsing extensions for per-layer embedding support.
This extends Megatron's argument parsing to include per-layer embedding options.
"""

import argparse
from megatron.training.arguments import _add_network_size_args, _add_regularization_args


def add_per_layer_embedding_args(parser):
    """Add per-layer embedding arguments to argument parser."""
    group = parser.add_argument_group(title='per-layer embedding')

    # Core per-layer embedding options
    group.add_argument('--use-per-layer-embedding', action='store_true',
                       help='Enable per-layer embedding functionality.')

    group.add_argument('--per-layer-vocab-size', type=int, default=10000,
                       help='Size of the per-layer vocabulary. Should be smaller '
                            'than main vocabulary. Default: 10000')

    group.add_argument('--per-layer-hidden-size', type=int, default=1024,
                       help='Hidden size for per-layer embeddings. Should be smaller '
                            'than main hidden size for efficiency. Default: 1024')

    group.add_argument('--per-layer-vocab-file', type=str, default=None,
                       help='Path to per-layer vocabulary file (JSON format).')

    group.add_argument('--per-layer-merge-file', type=str, default=None,
                       help='Path to per-layer merge file for BPE tokenizer.')

    # Data paths for per-layer tokens
    group.add_argument('--per-layer-data-path', type=str, default=None,
                       help='Path to per-layer tokenized data files.')

    group.add_argument('--per-layer-train-data-path', type=str, nargs='*', default=None,
                       help='Path(s) to the training dataset. Accepts multiple paths '
                            'separated by spaces for blended datasets.')

    group.add_argument('--per-layer-valid-data-path', type=str, nargs='*', default=None,
                       help='Path(s) to the validation dataset.')

    group.add_argument('--per-layer-test-data-path', type=str, nargs='*', default=None,
                       help='Path(s) to the test dataset.')

    # Per-layer embedding configuration
    group.add_argument('--per-layer-embed-scale-factor', type=float, default=1.0,
                       help='Scale factor for per-layer embeddings. Default: 1.0')

    group.add_argument('--per-layer-combination-method', type=str,
                       choices=['scaled_add', 'concat', 'gated'], default='scaled_add',
                       help='Method to combine main and per-layer embeddings. '
                            'Options: scaled_add (1/sqrt(2) scaling), concat, gated. '
                            'Default: scaled_add')

    group.add_argument('--per-layer-gate-activation', type=str,
                       choices=['relu', 'gelu', 'silu', 'swish'], default='silu',
                       help='Activation function for per-layer gates. Default: silu')

    group.add_argument('--per-layer-projection-dropout', type=float, default=0.0,
                       help='Dropout rate for per-layer projections. Default: 0.0')

    # Training and optimization options
    group.add_argument('--per-layer-learning-rate-scale', type=float, default=1.0,
                       help='Learning rate scale for per-layer components relative '
                            'to main model. Default: 1.0')

    group.add_argument('--per-layer-weight-decay', type=float, default=None,
                       help='Weight decay for per-layer components. If None, '
                            'uses main weight decay. Default: None')

    group.add_argument('--freeze-per-layer-embedding', action='store_true',
                       help='Freeze per-layer embedding parameters during training.')

    # Logging and monitoring
    group.add_argument('--log-per-layer-stats', action='store_true',
                       help='Log per-layer embedding statistics during training.')

    group.add_argument('--log-per-layer-gradients', action='store_true',
                       help='Log per-layer gradient norms during training.')

    group.add_argument('--per-layer-stats-interval', type=int, default=100,
                       help='Interval for logging per-layer statistics. Default: 100')

    # Initialization options
    group.add_argument('--per-layer-init-method', type=str,
                       choices=['normal', 'xavier', 'kaiming'], default='normal',
                       help='Initialization method for per-layer components. Default: normal')

    group.add_argument('--per-layer-init-std', type=float, default=0.02,
                       help='Standard deviation for normal initialization of '
                            'per-layer components. Default: 0.02')

    # Advanced options
    group.add_argument('--per-layer-sequence-parallel', action='store_true',
                       help='Enable sequence parallelism for per-layer components.')

    group.add_argument('--per-layer-tensor-parallel', action='store_true',
                       help='Enable tensor parallelism for per-layer embedding.')

    group.add_argument('--per-layer-gradient-checkpointing', action='store_true',
                       help='Enable gradient checkpointing for per-layer components.')

    # Experimental options
    group.add_argument('--per-layer-adaptive-vocab', action='store_true',
                       help='Enable adaptive vocabulary for per-layer embeddings.')

    group.add_argument('--per-layer-layer-specific-vocab', action='store_true',
                       help='Use different vocabularies for different layers.')

    return group


def validate_per_layer_args(args):
    """Validate per-layer embedding arguments."""
    if not args.use_per_layer_embedding:
        return

    # Required arguments when per-layer embedding is enabled
    if args.per_layer_vocab_file is None:
        raise ValueError("--per-layer-vocab-file is required when --use-per-layer-embedding is set")

    if args.per_layer_data_path is None and args.per_layer_train_data_path is None:
        raise ValueError("Either --per-layer-data-path or --per-layer-train-data-path is required "
                        "when --use-per-layer-embedding is set")

    # Validate vocabulary size
    if args.per_layer_vocab_size <= 0:
        raise ValueError("--per-layer-vocab-size must be positive")

    if args.per_layer_vocab_size >= args.padded_vocab_size:
        print(f"WARNING: per-layer vocab size ({args.per_layer_vocab_size}) is >= "
              f"main vocab size ({args.padded_vocab_size}). Consider reducing it for efficiency.")

    # Validate hidden size
    if args.per_layer_hidden_size <= 0:
        raise ValueError("--per-layer-hidden-size must be positive")

    if args.per_layer_hidden_size >= args.hidden_size:
        print(f"WARNING: per-layer hidden size ({args.per_layer_hidden_size}) is >= "
              f"main hidden size ({args.hidden_size}). Consider reducing it for efficiency.")

    # Validate scale factors
    if args.per_layer_embed_scale_factor <= 0:
        raise ValueError("--per-layer-embed-scale-factor must be positive")

    if args.per_layer_learning_rate_scale <= 0:
        raise ValueError("--per-layer-learning-rate-scale must be positive")

    # Validate dropout
    if not (0.0 <= args.per_layer_projection_dropout <= 1.0):
        raise ValueError("--per-layer-projection-dropout must be between 0.0 and 1.0")

    # Validate initialization
    if args.per_layer_init_std <= 0:
        raise ValueError("--per-layer-init-std must be positive")

    # Parallelism validation
    if args.per_layer_sequence_parallel and not args.sequence_parallel:
        print("WARNING: --per-layer-sequence-parallel is set but --sequence-parallel is not. "
              "Per-layer sequence parallelism requires main sequence parallelism.")

    if args.per_layer_tensor_parallel and args.tensor_model_parallel_size == 1:
        print("WARNING: --per-layer-tensor-parallel is set but tensor parallelism is disabled.")

    print(f"Per-layer embedding configuration validated:")
    print(f"  Vocab size: {args.per_layer_vocab_size}")
    print(f"  Hidden size: {args.per_layer_hidden_size}")
    print(f"  Combination method: {args.per_layer_combination_method}")
    print(f"  Gate activation: {args.per_layer_gate_activation}")


def get_per_layer_config_from_args(args):
    """Extract per-layer embedding configuration from arguments."""
    if not args.use_per_layer_embedding:
        return None

    config = {
        'enabled': True,
        'vocab_size': args.per_layer_vocab_size,
        'hidden_size': args.per_layer_hidden_size,
        'vocab_file': args.per_layer_vocab_file,
        'merge_file': args.per_layer_merge_file,
        'data_path': args.per_layer_data_path,
        'embed_scale_factor': args.per_layer_embed_scale_factor,
        'combination_method': args.per_layer_combination_method,
        'gate_activation': args.per_layer_gate_activation,
        'projection_dropout': args.per_layer_projection_dropout,
        'learning_rate_scale': args.per_layer_learning_rate_scale,
        'weight_decay': args.per_layer_weight_decay,
        'freeze_embedding': args.freeze_per_layer_embedding,
        'init_method': args.per_layer_init_method,
        'init_std': args.per_layer_init_std,
        'sequence_parallel': args.per_layer_sequence_parallel,
        'tensor_parallel': args.per_layer_tensor_parallel,
        'gradient_checkpointing': args.per_layer_gradient_checkpointing,
        'log_stats': args.log_per_layer_stats,
        'log_gradients': args.log_per_layer_gradients,
        'stats_interval': args.per_layer_stats_interval,
    }

    return config


# Example usage in main training script:
def add_per_layer_args_to_parser(parser):
    """Add per-layer embedding arguments to existing argument parser."""
    add_per_layer_embedding_args(parser)
    return parser

# Example integration with existing argument parsing:
"""
In megatron/training/arguments.py, you would add:

from per_layer_arguments import add_per_layer_embedding_args, validate_per_layer_args

def parse_args(extra_args_provider=None, defaults={}, ignore_unknown_args=False):
    # ... existing code ...

    # Add per-layer embedding arguments
    add_per_layer_embedding_args(parser)

    # ... existing parsing code ...

    # Validate per-layer arguments
    validate_per_layer_args(args)

    return args
"""