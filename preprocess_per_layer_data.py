#!/usr/bin/env python3

"""
Data preprocessing for per-layer embedding training.
Creates per-layer tokenized data alongside standard GPT data.
"""

import json
import argparse
import multiprocessing
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

# Megatron imports
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import parse_args
from tools.preprocess_data import get_args as get_preprocess_args
from tools.preprocess_data import Encoder


def create_per_layer_vocabulary(
    text_files: List[str],
    vocab_size: int = 10000,
    output_path: str = "per_layer_vocab.json"
) -> Dict[str, int]:
    """
    Create a specialized vocabulary for per-layer embeddings.

    This creates a smaller, focused vocabulary that captures:
    1. Common function words and syntax
    2. Frequent content words
    3. Special tokens for layer-specific processing

    Args:
        text_files: List of text files to analyze
        vocab_size: Target vocabulary size
        output_path: Path to save vocabulary

    Returns:
        Dictionary mapping tokens to IDs
    """
    print(f"Creating per-layer vocabulary of size {vocab_size}...")

    # Special tokens for per-layer processing
    special_tokens = [
        "<|startoftext|>", "<|endoftext|>", "<|pad|>", "<|unk|>",
        "<|layer_start|>", "<|layer_end|>", "<|context|>", "<|focus|>",
        "<|syntax|>", "<|semantic|>", "<|attention|>", "<|memory|>"
    ]

    # Common function words (important for syntax/structure)
    function_words = [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "up", "about", "into", "through",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "this", "that", "these", "those", "I", "you", "he", "she", "it",
        "we", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "her", "its", "our", "their", "who", "what", "when",
        "where", "why", "how", "which", "that", "if", "because", "so",
        ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}"
    ]

    # Frequency analysis would go here for real implementation
    # For demo, create a simple vocabulary
    vocab = {}

    # Add special tokens
    for i, token in enumerate(special_tokens):
        vocab[token] = i

    # Add function words
    for token in function_words:
        if token not in vocab:
            vocab[token] = len(vocab)

    # Add numbers and common patterns
    for i in range(100):
        token = str(i)
        if len(vocab) < vocab_size and token not in vocab:
            vocab[token] = len(vocab)

    # Add more common tokens up to vocab_size
    common_tokens = [
        "code", "function", "class", "method", "variable", "return", "import",
        "def", "if", "else", "for", "while", "try", "except", "with", "as",
        "print", "input", "output", "data", "model", "train", "test", "eval",
        "loss", "accuracy", "epoch", "batch", "learning", "rate", "optimizer"
    ]

    for token in common_tokens:
        if len(vocab) < vocab_size and token not in vocab:
            vocab[token] = len(vocab)

    # Fill remaining slots
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for char in alphabet:
        if len(vocab) < vocab_size:
            vocab[char] = len(vocab)
        if len(vocab) < vocab_size:
            vocab[char.upper()] = len(vocab)

    # Save vocabulary
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f"Created per-layer vocabulary with {len(vocab)} tokens")
    print(f"Saved to {output_path}")

    return vocab


def create_per_layer_tokens(
    text: str,
    main_tokenizer,
    per_layer_vocab: Dict[str, int],
    strategy: str = "syntax_focus"
) -> List[int]:
    """
    Create per-layer tokens that complement main tokens.

    Different strategies for creating per-layer tokens:
    1. syntax_focus: Focus on syntactic/structural elements
    2. semantic_focus: Focus on semantic/content elements
    3. attention_guidance: Provide attention guidance tokens
    4. random_subset: Random subset of main tokens (baseline)

    Args:
        text: Input text
        main_tokenizer: Main GPT tokenizer
        per_layer_vocab: Per-layer vocabulary
        strategy: Strategy for creating per-layer tokens

    Returns:
        List of per-layer token IDs
    """
    if strategy == "syntax_focus":
        return _create_syntax_focus_tokens(text, per_layer_vocab)
    elif strategy == "semantic_focus":
        return _create_semantic_focus_tokens(text, per_layer_vocab)
    elif strategy == "attention_guidance":
        return _create_attention_guidance_tokens(text, per_layer_vocab)
    elif strategy == "random_subset":
        return _create_random_subset_tokens(text, main_tokenizer, per_layer_vocab)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _create_syntax_focus_tokens(text: str, vocab: Dict[str, int]) -> List[int]:
    """Create tokens focused on syntax and structure."""
    words = text.split()
    tokens = []

    for word in words:
        # Clean word
        clean_word = word.strip('.,!?;:"()[]{}').lower()

        # Prioritize function words and punctuation
        if clean_word in vocab:
            tokens.append(vocab[clean_word])
        elif word in vocab:  # Check original case
            tokens.append(vocab[word])
        elif any(p in word for p in ".,!?;:()[]{}"):
            # Extract punctuation
            for char in word:
                if char in vocab:
                    tokens.append(vocab[char])
        else:
            # Unknown token
            tokens.append(vocab.get("<|unk|>", 0))

    return tokens


def _create_semantic_focus_tokens(text: str, vocab: Dict[str, int]) -> List[int]:
    """Create tokens focused on semantic content."""
    words = text.split()
    tokens = []

    # Simple semantic focusing (in practice, would use NER, POS tagging, etc.)
    for word in words:
        clean_word = word.strip('.,!?;:"()[]{}').lower()

        # Focus on content words, skip most function words
        function_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        if clean_word not in function_words and clean_word in vocab:
            tokens.append(vocab[clean_word])
        elif clean_word in function_words:
            # Use special context token for function words
            tokens.append(vocab.get("<|context|>", 0))
        else:
            tokens.append(vocab.get("<|unk|>", 0))

    return tokens


def _create_attention_guidance_tokens(text: str, vocab: Dict[str, int]) -> List[int]:
    """Create tokens that guide attention patterns."""
    words = text.split()
    tokens = []

    for i, word in enumerate(words):
        # Provide attention guidance based on position and content
        if i == 0:
            tokens.append(vocab.get("<|layer_start|>", 0))
        elif i == len(words) - 1:
            tokens.append(vocab.get("<|layer_end|>", 0))
        elif i % 10 == 0:  # Every 10th word gets attention marker
            tokens.append(vocab.get("<|attention|>", 0))
        else:
            # Regular processing
            clean_word = word.strip('.,!?;:"()[]{}').lower()
            tokens.append(vocab.get(clean_word, vocab.get("<|unk|>", 0)))

    return tokens


def _create_random_subset_tokens(text: str, main_tokenizer, vocab: Dict[str, int]) -> List[int]:
    """Create random subset of main tokens (baseline method)."""
    # Tokenize with main tokenizer
    main_tokens = main_tokenizer.tokenize(text)

    # Map to per-layer vocab space (simplified)
    per_layer_tokens = []
    for token in main_tokens:
        # Simple mapping (in practice, would use proper vocabulary alignment)
        per_layer_id = hash(token) % len(vocab)
        per_layer_tokens.append(per_layer_id)

    return per_layer_tokens


def process_file_pair(args_tuple):
    """Process a pair of main and per-layer data files."""
    input_file, output_file, per_layer_output_file, args, per_layer_vocab = args_tuple

    print(f"Processing {input_file} -> {output_file}, {per_layer_output_file}")

    # Build tokenizer
    tokenizer = build_tokenizer(args)

    # Process file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    main_data = []
    per_layer_data = []

    for line in lines:
        if line.strip():
            try:
                # Parse JSON line
                data = json.loads(line)
                text = data.get('text', '')

                # Tokenize for main model
                main_tokens = tokenizer.tokenize(text)

                # Create per-layer tokens
                per_layer_tokens = create_per_layer_tokens(
                    text, tokenizer, per_layer_vocab, args.per_layer_strategy
                )

                # Ensure same length (truncate or pad)
                min_len = min(len(main_tokens), len(per_layer_tokens))
                if min_len > 0:
                    main_data.extend(main_tokens[:min_len])
                    per_layer_data.extend(per_layer_tokens[:min_len])

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # Save processed data
    np.array(main_data, dtype=np.int32).tofile(output_file + '.bin')
    np.array(per_layer_data, dtype=np.int32).tofile(per_layer_output_file + '.bin')

    return len(main_data)


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess data for per-layer embedding training')

    # Input/output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file(s) (one text per line)')
    parser.add_argument('--output-prefix', type=str, required=True,
                       help='Output file prefix')
    parser.add_argument('--per-layer-output-prefix', type=str, required=True,
                       help='Per-layer output file prefix')

    # Vocabulary arguments
    parser.add_argument('--vocab-file', type=str, required=True,
                       help='Main vocabulary file')
    parser.add_argument('--per-layer-vocab-file', type=str, required=True,
                       help='Per-layer vocabulary file')
    parser.add_argument('--per-layer-vocab-size', type=int, default=10000,
                       help='Per-layer vocabulary size')

    # Processing arguments
    parser.add_argument('--per-layer-strategy', type=str, default='syntax_focus',
                       choices=['syntax_focus', 'semantic_focus', 'attention_guidance', 'random_subset'],
                       help='Strategy for creating per-layer tokens')
    parser.add_argument('--tokenizer-type', type=str, default='GPT2BPETokenizer',
                       help='Tokenizer type')
    parser.add_argument('--merge-file', type=str, default=None,
                       help='Merge file for BPE tokenizer')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')

    args = parser.parse_args()

    # Create per-layer vocabulary if it doesn't exist
    if not Path(args.per_layer_vocab_file).exists():
        print(f"Creating per-layer vocabulary: {args.per_layer_vocab_file}")
        per_layer_vocab = create_per_layer_vocabulary(
            [args.input],
            args.per_layer_vocab_size,
            args.per_layer_vocab_file
        )
    else:
        print(f"Loading existing per-layer vocabulary: {args.per_layer_vocab_file}")
        with open(args.per_layer_vocab_file, 'r') as f:
            per_layer_vocab = json.load(f)

    # Process input files
    if args.workers == 1:
        # Single process
        total_tokens = process_file_pair((
            args.input,
            args.output_prefix,
            args.per_layer_output_prefix,
            args,
            per_layer_vocab
        ))
    else:
        # Multi-process (for multiple input files)
        # Implementation would split files across workers
        print("Multi-process mode not implemented in this demo")
        total_tokens = process_file_pair((
            args.input,
            args.output_prefix,
            args.per_layer_output_prefix,
            args,
            per_layer_vocab
        ))

    print(f"Preprocessing complete!")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Main data saved to: {args.output_prefix}.bin")
    print(f"Per-layer data saved to: {args.per_layer_output_prefix}.bin")


if __name__ == "__main__":
    main()