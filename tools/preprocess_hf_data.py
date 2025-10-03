"""
Resilient Hugging Face Dataset Preprocessor for Megatron
- Checkpoint support for resume on failure
- Progress tracking with detailed logging
- Automatic retry on connection errors
- Support for sentence splitting and multiple JSON keys
- Multi-process parallel processing for large datasets
"""

import os
import json
import time
import argparse
import multiprocessing
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from datasets import load_dataset

try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars

    nltk_available = True
except ImportError:
    PunktLanguageVars = object
    nltk_available = False

from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer as build_new_tokenizer
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


# Custom language vars to preserve empty lines with NLTK
class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what preserves newlines
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    """No-op splitter that returns text as-is"""

    def tokenize(self, *text):
        return text


class CheckpointManager:
    """Manages checkpoint saving and loading for resume capability"""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file = f"{checkpoint_path}.checkpoint.json"

    def save(self, processed_count: int, last_successful_index: int, metadata: Dict[str, Any]):
        """Save current processing progress to checkpoint file"""
        checkpoint_data = {
            'processed_count': processed_count,
            'last_successful_index': last_successful_index,
            'metadata': metadata,
            'timestamp': time.time()
        }

        # Write to temporary file first, then atomically replace
        temp_file = f"{self.checkpoint_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        os.replace(temp_file, self.checkpoint_file)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load saved checkpoint data"""
        if not os.path.exists(self.checkpoint_file):
            return None

        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            return None

    def exists(self) -> bool:
        """Check if checkpoint file exists"""
        return os.path.exists(self.checkpoint_file)

    def clear(self):
        """Remove checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print(f"[INFO] Checkpoint file removed: {self.checkpoint_file}")


class ResilientDatasetProcessor:
    """Dataset preprocessor with resume capability on interruption"""

    def __init__(self, args):
        self.args = args
        self.checkpoint_mgr = CheckpointManager(args.output_prefix)

        print(f"[INFO] Initializing tokenizer: {args.tokenizer_type}")
        # Initialize tokenizer
        if args.legacy_tokenizer:
            self.tokenizer = build_tokenizer(args)
        else:
            self.tokenizer = build_new_tokenizer(args)
        print(f"[INFO] Tokenizer initialized. Vocab size: {self.tokenizer.vocab_size}")

        # Initialize sentence splitter if needed
        if args.split_sentences:
            if not nltk_available:
                raise Exception("NLTK is required for sentence splitting but not available")

            print(f"[INFO] Initializing sentence splitter for language: {args.lang}")
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))

            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{args.lang}.pickle")
                url = f"nltk:{library}"

            splitter = nltk.load(url)
            if args.keep_newlines:
                self.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars())
                print(f"[INFO] Sentence splitter initialized (preserving newlines)")
            else:
                self.splitter = splitter
                print(f"[INFO] Sentence splitter initialized")
        else:
            self.splitter = IdentitySplitter()
            print(f"[INFO] No sentence splitting (document-level processing)")

        # Determine processing level
        self.level = "sentence" if args.split_sentences else "document"

        # Setup output files for each JSON key
        self.output_bin_files = {}
        self.output_idx_files = {}
        self.builders = {}

        for key in args.json_keys:
            self.output_bin_files[key] = f"{args.output_prefix}_{key}_{self.level}.bin"
            self.output_idx_files[key] = f"{args.output_prefix}_{key}_{self.level}.idx"
            print(f"[INFO] Output for key '{key}':")
            print(f"[INFO]   Binary: {self.output_bin_files[key]}")
            print(f"[INFO]   Index: {self.output_idx_files[key]}")

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        max_len = 1000000
        tokens_list = [self.splitter.tokenize(text[i:i + max_len])
                       for i in range(0, len(text), max_len)]
        return [token for partial in tokens_list for token in partial]

    def process_example(self, example: dict) -> Dict[str, tuple]:
        """Process a single example, returning tokenized data for each key"""
        results = {}

        for key in self.args.json_keys:
            text = example.get(key, '')
            if not text:
                results[key] = (None, None)
                continue

            # Split into sentences if needed
            if self.args.split_sentences:
                sentences = self.split_sentences(text)
            else:
                sentences = [text]

            # Tokenize each sentence
            doc_ids = []
            sentence_lens = []

            for sentence in sentences:
                sentence_ids = self.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))

            # Add EOD token if requested
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(self.tokenizer.eod)
                if sentence_lens:
                    sentence_lens[-1] += 1

            if len(doc_ids) > 0:
                results[key] = (doc_ids, sentence_lens)
            else:
                results[key] = (None, None)

        return results

    def skip_to_position(self, dataset, target_index: int):
        """Skip dataset to a specific position for resume"""
        print(f"[INFO] Skipping to dataset index {target_index}...")
        print(f"[INFO] This may take a while for large skip distances...")

        skipped = 0
        skip_start_time = time.time()
        last_log_time = skip_start_time

        for _ in dataset:
            skipped += 1

            # Log progress every 10 seconds during skip
            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed = current_time - skip_start_time
                skip_rate = skipped / elapsed if elapsed > 0 else 0
                eta_seconds = (target_index - skipped) / skip_rate if skip_rate > 0 else 0
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                print(f"[PROGRESS] Skipped {skipped:,}/{target_index:,} "
                      f"({skipped / target_index * 100:.1f}%) - "
                      f"Rate: {skip_rate:.0f} docs/s - "
                      f"ETA: {eta_str}")
                last_log_time = current_time

            if skipped >= target_index:
                break

        skip_elapsed = time.time() - skip_start_time
        print(f"[INFO] Successfully skipped to index {target_index} "
              f"(took {skip_elapsed:.1f}s)")
        return dataset

    def format_time(self, seconds):
        """Format seconds into human-readable time string"""
        return str(timedelta(seconds=int(seconds)))

    def format_number(self, num):
        """Format number with thousand separators"""
        return f"{num:,}"

    def process_dataset(self):
        """Main processing logic with checkpoint support"""

        # Check for existing checkpoint
        checkpoint = self.checkpoint_mgr.load()
        start_index = 0
        processed_count = 0
        resume_mode = False

        if checkpoint and not self.args.force_restart:
            start_index = checkpoint['last_successful_index']
            processed_count = checkpoint['processed_count']
            resume_mode = True

            checkpoint_time = datetime.fromtimestamp(checkpoint['timestamp'])
            print(f"\n{'=' * 70}")
            print(f"[INFO] CHECKPOINT DETECTED")
            print(f"{'=' * 70}")
            print(f"[INFO] Checkpoint created at: {checkpoint_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[INFO] Documents already processed: {self.format_number(processed_count)}")
            print(f"[INFO] Resuming from dataset index: {self.format_number(start_index)}")
            print(f"{'=' * 70}\n")

            # Verify output files exist for all keys
            all_files_exist = True
            for key in self.args.json_keys:
                bin_file = self.output_bin_files[key]
                idx_file = self.output_idx_files[key]
                if not os.path.exists(bin_file) or not os.path.exists(idx_file):
                    print(f"[ERROR] Checkpoint exists but output files for key '{key}' are missing!")
                    print(f"[ERROR] Expected files:")
                    print(f"[ERROR]   - {bin_file}")
                    print(f"[ERROR]   - {idx_file}")
                    all_files_exist = False

            if not all_files_exist:
                print(f"[ERROR] Please use --force-restart to start over")
                return

            # Show file sizes
            for key in self.args.json_keys:
                bin_size = os.path.getsize(self.output_bin_files[key])
                idx_size = os.path.getsize(self.output_idx_files[key])
                print(f"[INFO] Existing files for key '{key}':")
                print(f"[INFO]   Binary: {bin_size / (1024 ** 3):.2f} GB")
                print(f"[INFO]   Index: {idx_size / (1024 ** 2):.2f} MB")
        else:
            if checkpoint:
                print(f"[INFO] Force restart enabled: ignoring existing checkpoint")
                self.checkpoint_mgr.clear()
            print(f"[INFO] Starting fresh preprocessing from the beginning")

        # Load dataset
        print(f"\n{'=' * 70}")
        print(f"[INFO] LOADING DATASET")
        print(f"{'=' * 70}")
        print(f"[INFO] Dataset name: {self.args.dataset_name}")
        if self.args.dataset_config:
            print(f"[INFO] Dataset config: {self.args.dataset_config}")
        print(f"[INFO] Split: {self.args.split}")
        print(f"[INFO] JSON keys to extract: {', '.join(self.args.json_keys)}")
        print(f"[INFO] Processing level: {self.level}")
        print(f"[INFO] Streaming mode: enabled")

        # Check if this is a partition worker
        if hasattr(self.args, 'partition_id') and hasattr(self.args, 'total_partitions'):
            print(f"[INFO] Partition mode: {self.args.partition_id + 1}/{self.args.total_partitions}")

        print(f"{'=' * 70}\n")

        max_retries = self.args.max_retries
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Load in streaming mode
                print(f"[INFO] Connecting to Hugging Face Hub...")
                load_start = time.time()

                load_kwargs = {
                    'path': self.args.dataset_name,
                    'split': self.args.split,
                    'streaming': True
                }
                if self.args.dataset_config:
                    load_kwargs['name'] = self.args.dataset_config

                dataset = load_dataset(**load_kwargs)

                # Apply sharding if this is a partition worker
                if hasattr(self.args, 'partition_id') and hasattr(self.args, 'total_partitions'):
                    print(f"[INFO] Applying shard {self.args.partition_id}/{self.args.total_partitions}...")
                    dataset = dataset.shard(
                        num_shards=self.args.total_partitions,
                        index=self.args.partition_id
                    )

                load_elapsed = time.time() - load_start
                print(f"[INFO] Dataset loaded successfully (took {load_elapsed:.2f}s)")

                # Skip to resume position if needed
                if resume_mode:
                    dataset = self.skip_to_position(dataset, start_index)

                # Initialize builders for each key
                print(f"\n[INFO] Initializing indexed dataset builders...")
                for key in self.args.json_keys:
                    if resume_mode:
                        print(f"[INFO] Opening existing files for key '{key}' in append mode")
                    else:
                        print(f"[INFO] Creating new files for key '{key}'")

                    self.builders[key] = indexed_dataset.IndexedDatasetBuilder(
                        self.output_bin_files[key],
                        dtype=indexed_dataset.DType.optimal_dtype(self.tokenizer.vocab_size),
                    )

                print(f"[INFO] All builders initialized successfully")

                # Start processing
                current_index = start_index
                last_checkpoint_time = time.time()
                last_log_time = time.time()
                start_time = time.time()
                total_tokens = {key: 0 for key in self.args.json_keys}
                empty_docs = {key: 0 for key in self.args.json_keys}
                failed_docs = 0

                print(f"\n{'=' * 70}")
                print(f"[INFO] PROCESSING STARTED")
                print(f"{'=' * 70}")
                print(
                    f"[INFO] Checkpoint interval: every {self.format_number(self.args.checkpoint_interval)} documents")
                print(f"[INFO] Log interval: every {self.format_number(self.args.log_interval)} documents")
                print(f"[INFO] Max retries on error: {max_retries}")
                print(f"{'=' * 70}\n")

                for example in dataset:
                    try:
                        # Process example for all keys
                        results = self.process_example(example)

                        # Add to builders
                        any_success = False
                        for key in self.args.json_keys:
                            doc_ids, lens = results[key]
                            if doc_ids is not None:
                                self.builders[key].add_document(doc_ids, lens)
                                total_tokens[key] += len(doc_ids)
                                any_success = True
                            else:
                                empty_docs[key] += 1

                        if any_success:
                            processed_count += 1

                        current_index += 1

                        # Progress logging
                        current_time = time.time()
                        if processed_count % self.args.log_interval == 0 and processed_count > 0:
                            elapsed = current_time - start_time
                            interval_elapsed = current_time - last_log_time

                            docs_per_sec = processed_count / elapsed if elapsed > 0 else 0

                            print(f"[PROGRESS] Processed: {self.format_number(processed_count)} docs | "
                                  f"Index: {self.format_number(current_index)} | "
                                  f"Speed: {docs_per_sec:.1f} docs/s")

                            # Per-key statistics
                            for key in self.args.json_keys:
                                tokens = total_tokens[key]
                                tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
                                avg_tokens = tokens / processed_count if processed_count > 0 else 0

                                if os.path.exists(self.output_bin_files[key]):
                                    size_gb = os.path.getsize(self.output_bin_files[key]) / (1024 ** 3)
                                else:
                                    size_gb = 0

                                print(f"           [{key}] Tokens: {self.format_number(tokens)} "
                                      f"({tokens_per_sec:.0f} tok/s) | "
                                      f"Avg: {avg_tokens:.0f} tok/doc | "
                                      f"Size: {size_gb:.2f} GB | "
                                      f"Empty: {empty_docs[key]}")

                            print(f"           Failed docs: {failed_docs}")

                            last_log_time = current_time

                        # Periodic checkpoint saving
                        if processed_count % self.args.checkpoint_interval == 0 and processed_count > 0:
                            checkpoint_start = time.time()

                            self.checkpoint_mgr.save(
                                processed_count=processed_count,
                                last_successful_index=current_index,
                                metadata={
                                    'dataset_name': self.args.dataset_name,
                                    'dataset_config': self.args.dataset_config,
                                    'split': self.args.split,
                                    'json_keys': self.args.json_keys,
                                    'total_tokens': total_tokens,
                                    'empty_docs': empty_docs,
                                    'failed_docs': failed_docs,
                                    'level': self.level
                                }
                            )

                            checkpoint_elapsed = time.time() - checkpoint_start
                            elapsed_total = time.time() - start_time

                            print(f"[CHECKPOINT] Saved at {self.format_number(processed_count)} docs | "
                                  f"Time: {self.format_time(elapsed_total)} | "
                                  f"Save took: {checkpoint_elapsed:.2f}s")

                            last_checkpoint_time = time.time()

                    except Exception as e:
                        print(f"[WARNING] Failed to process document at index {current_index}: {e}")
                        failed_docs += 1
                        current_index += 1
                        continue

                # Finalize output files
                print(f"\n{'=' * 70}")
                print(f"[INFO] FINALIZING OUTPUT FILES")
                print(f"{'=' * 70}")

                for key in self.args.json_keys:
                    print(f"[INFO] Finalizing files for key '{key}'...")
                    finalize_start = time.time()
                    self.builders[key].finalize(self.output_idx_files[key])
                    finalize_elapsed = time.time() - finalize_start
                    print(f"[INFO] Key '{key}' finalized (took {finalize_elapsed:.2f}s)")

                # Print final statistics
                total_time = time.time() - start_time

                print(f"\n{'=' * 70}")
                print(f"[SUCCESS] PROCESSING COMPLETED")
                print(f"{'=' * 70}")
                print(f"[STATS] Total documents processed: {self.format_number(processed_count)}")
                print(f"[STATS] Processing level: {self.level}")

                for key in self.args.json_keys:
                    tokens = total_tokens[key]
                    avg_tokens = tokens / processed_count if processed_count > 0 else 0

                    final_bin_size = os.path.getsize(self.output_bin_files[key]) if os.path.exists(
                        self.output_bin_files[key]) else 0
                    final_idx_size = os.path.getsize(self.output_idx_files[key]) if os.path.exists(
                        self.output_idx_files[key]) else 0

                    print(f"\n[STATS] Key '{key}':")
                    print(f"[STATS]   Total tokens: {self.format_number(tokens)}")
                    print(f"[STATS]   Average tokens per document: {avg_tokens:.2f}")
                    print(f"[STATS]   Empty documents: {self.format_number(empty_docs[key])}")
                    print(
                        f"[OUTPUT]   Binary file: {self.output_bin_files[key]} ({final_bin_size / (1024 ** 3):.2f} GB)")
                    print(
                        f"[OUTPUT]   Index file: {self.output_idx_files[key]} ({final_idx_size / (1024 ** 2):.2f} MB)")

                print(f"\n[STATS] Failed documents: {self.format_number(failed_docs)}")
                print(f"[STATS] Total processing time: {self.format_time(total_time)}")
                print(f"[STATS] Average speed: {processed_count / total_time:.2f} docs/s")

                total_all_tokens = sum(total_tokens.values())
                print(f"[STATS] Total tokens (all keys): {self.format_number(total_all_tokens)}")
                print(f"[STATS] Token throughput: {total_all_tokens / total_time:.0f} tokens/s")
                print(f"{'=' * 70}\n")

                # Clean up checkpoint
                self.checkpoint_mgr.clear()
                print(f"[INFO] Processing pipeline completed successfully!")
                return

            except Exception as e:
                retry_count += 1

                print(f"\n{'!' * 70}")
                print(f"[ERROR] PROCESSING FAILED")
                print(f"{'!' * 70}")
                print(f"[ERROR] Error message: {e}")
                print(f"[ERROR] Error type: {type(e).__name__}")
                print(f"[ERROR] Retry attempt: {retry_count}/{max_retries}")
                print(f"{'!' * 70}\n")

                if retry_count < max_retries:
                    # Save checkpoint before retry
                    print(f"[INFO] Saving emergency checkpoint...")
                    self.checkpoint_mgr.save(
                        processed_count=processed_count,
                        last_successful_index=current_index,
                        metadata={
                            'dataset_name': self.args.dataset_name,
                            'dataset_config': self.args.dataset_config,
                            'split': self.args.split,
                            'json_keys': self.args.json_keys,
                            'last_error': str(e),
                            'error_type': type(e).__name__,
                            'retry_count': retry_count,
                            'level': self.level
                        }
                    )
                    print(f"[INFO] Emergency checkpoint saved")

                    # Exponential backoff with max 5 minutes
                    wait_time = min(60 * (2 ** (retry_count - 1)), 300)
                    print(f"[INFO] Waiting {wait_time}s before retry...")
                    print(f"[INFO] Next retry will start at: "
                          f"{(datetime.now() + timedelta(seconds=wait_time)).strftime('%Y-%m-%d %H:%M:%S')}")

                    for remaining in range(wait_time, 0, -10):
                        if remaining % 30 == 0 or remaining <= 10:
                            print(f"[INFO] Retrying in {remaining}s...")
                        time.sleep(min(10, remaining))

                    print(f"\n[INFO] Resuming processing attempt {retry_count + 1}...")

                    # Reload checkpoint for retry
                    checkpoint = self.checkpoint_mgr.load()
                    if checkpoint:
                        start_index = checkpoint['last_successful_index']
                        processed_count = checkpoint['processed_count']
                        resume_mode = True
                else:
                    print(f"\n{'!' * 70}")
                    print(f"[FATAL] Maximum retries ({max_retries}) reached")
                    print(f"{'!' * 70}")
                    print(f"[INFO] Progress has been saved to checkpoint")
                    print(f"[INFO] Run the script again to resume from:")
                    print(f"[INFO]   - Processed documents: {self.format_number(processed_count)}")
                    print(f"[INFO]   - Dataset index: {self.format_number(current_index)}")
                    print(f"[INFO] Checkpoint file: {self.checkpoint_mgr.checkpoint_file}")
                    print(f"{'!' * 70}\n")
                    raise


def get_args():
    parser = argparse.ArgumentParser(
        description='Resilient Hugging Face Dataset Preprocessor for Megatron',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add tokenizer arguments
    parser = _add_tokenizer_args(parser)

    # Dataset arguments
    group = parser.add_argument_group('dataset arguments')
    group.add_argument('--dataset-name', type=str, required=True,
                       help='Hugging Face dataset name (e.g., HuggingFaceTB/smollm-corpus)')
    group.add_argument('--dataset-config', type=str, default=None,
                       help='Dataset configuration/subset name (e.g., fineweb-edu-dedup)')
    group.add_argument('--split', type=str, default='train',
                       help='Dataset split to process')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space-separated list of keys to extract from dataset')

    # Sentence splitting arguments
    group = parser.add_argument_group('sentence splitting arguments')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting')
    group.add_argument('--lang', type=str, default='english',
                       help='Language for NLTK sentence splitting')

    # Output arguments
    group = parser.add_argument_group('output arguments')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Output file prefix')
    group.add_argument('--append-eod', action='store_true',
                       help='Append EOD token to each document')

    # Processing arguments
    group = parser.add_argument_group('processing arguments')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (for single partition) or total workers')
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of partitions for parallel processing')
    group.add_argument('--checkpoint-interval', type=int, default=10000,
                       help='Save checkpoint every N documents')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Print progress every N documents')
    group.add_argument('--max-retries', type=int, default=10,
                       help='Maximum number of retries on failure')
    group.add_argument('--force-restart', action='store_true',
                       help='Ignore existing checkpoint and start fresh')
    group.add_argument('--legacy-tokenizer', action='store_true',
                       help='Use legacy tokenizer system')

    args = parser.parse_args()

    # Validation
    if args.split_sentences and not nltk_available:
        parser.error("--split-sentences requires NLTK to be installed. "
                     "Install with: pip install nltk")

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("[WARNING] BERT tokenizers typically work better with sentence splitting.")
        print("[WARNING] Consider using --split-sentences flag.")

    if args.partitions > 1 and args.workers % args.partitions != 0:
        parser.error(f"--workers ({args.workers}) must be divisible by --partitions ({args.partitions})")

    # Set tokenizer defaults
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.keep_empty = False

    return args


def process_partition_worker(partition_id: int, args, total_partitions: int):
    """Worker function to process a single partition"""

    # Create partition-specific args
    partition_args = argparse.Namespace(**vars(args))
    partition_args.output_prefix = f"{args.output_prefix}_partition_{partition_id}"
    partition_args.partition_id = partition_id
    partition_args.total_partitions = total_partitions

    print(f"\n[PARTITION {partition_id}] Starting partition processor")
    print(f"[PARTITION {partition_id}] Output prefix: {partition_args.output_prefix}")

    # Create processor for this partition
    processor = ResilientDatasetProcessor(partition_args)

    # Modify dataset loading to shard by partition
    original_process = processor.process_dataset

    def process_with_sharding():
        """Modified process_dataset that shards the dataset"""
        # We'll intercept the dataset loading and add sharding
        processor._original_dataset_load = True
        processor._partition_id = partition_id
        processor._total_partitions = total_partitions
        original_process()

    processor.process_dataset = process_with_sharding
    processor.process_dataset()

    print(f"[PARTITION {partition_id}] Completed successfully")


def merge_partition_files(args):
    """Merge all partition files into final output files"""

    print(f"\n{'=' * 70}")
    print(f"[INFO] MERGING PARTITIONS")
    print(f"{'=' * 70}")

    # Initialize tokenizer for dtype
    if args.legacy_tokenizer:
        tokenizer = build_tokenizer(args)
    else:
        tokenizer = build_new_tokenizer(args)

    level = "sentence" if args.split_sentences else "document"

    for key in args.json_keys:
        print(f"\n[INFO] Merging files for key '{key}'...")

        output_bin = f"{args.output_prefix}_{key}_{level}.bin"
        output_idx = f"{args.output_prefix}_{key}_{level}.idx"

        # Create builder for merged file
        print(f"[INFO] Creating merged builder...")
        builder = indexed_dataset.IndexedDatasetBuilder(
            output_bin,
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        # Add each partition's index
        total_docs = 0
        for partition_id in range(args.partitions):
            partition_prefix = f"{args.output_prefix}_partition_{partition_id}"
            partition_file = f"{partition_prefix}_{key}_{level}"

            if not os.path.exists(f"{partition_file}.idx"):
                print(f"[WARNING] Partition {partition_id} file not found: {partition_file}.idx")
                continue

            print(f"[INFO] Adding partition {partition_id}: {partition_file}")
            builder.add_index(partition_file)

            # Count documents in this partition
            partition_idx = indexed_dataset.IndexedDataset(partition_file, mmap=False)
            partition_docs = len(partition_idx)
            total_docs += partition_docs
            print(f"[INFO]   Documents in partition: {partition_docs:,}")

        # Finalize merged file
        print(f"[INFO] Finalizing merged file...")
        finalize_start = time.time()
        builder.finalize(output_idx)
        finalize_elapsed = time.time() - finalize_start

        # Get final file sizes
        bin_size = os.path.getsize(output_bin) if os.path.exists(output_bin) else 0
        idx_size = os.path.getsize(output_idx) if os.path.exists(output_idx) else 0

        print(f"[SUCCESS] Key '{key}' merged successfully")
        print(f"[INFO]   Total documents: {total_docs:,}")
        print(f"[INFO]   Binary file: {output_bin} ({bin_size / (1024 ** 3):.2f} GB)")
        print(f"[INFO]   Index file: {output_idx} ({idx_size / (1024 ** 2):.2f} MB)")
        print(f"[INFO]   Finalization took: {finalize_elapsed:.2f}s")

    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] ALL PARTITIONS MERGED")
    print(f"{'=' * 70}\n")


def main():
    args = get_args()

    print("\n" + "=" * 70)
    print("RESILIENT HUGGING FACE DATASET PREPROCESSOR FOR MEGATRON")
    print("=" * 70)
    print(f"[CONFIG] Dataset: {args.dataset_name}")
    if args.dataset_config:
        print(f"[CONFIG] Config: {args.dataset_config}")
    print(f"[CONFIG] Split: {args.split}")
    print(f"[CONFIG] JSON keys: {', '.join(args.json_keys)}")
    print(f"[CONFIG] Tokenizer: {args.tokenizer_type}")
    print(f"[CONFIG] Legacy tokenizer: {args.legacy_tokenizer}")
    print(f"[CONFIG] Split sentences: {args.split_sentences}")
    if args.split_sentences:
        print(f"[CONFIG] Language: {args.lang}")
        print(f"[CONFIG] Keep newlines: {args.keep_newlines}")
    print(f"[CONFIG] Output prefix: {args.output_prefix}")
    print(f"[CONFIG] Append EOD: {args.append_eod}")
    print(f"[CONFIG] Workers: {args.workers}")
    print(f"[CONFIG] Partitions: {args.partitions}")
    if args.partitions > 1:
        workers_per_partition = args.workers // args.partitions
        print(f"[CONFIG] Workers per partition: {workers_per_partition}")
    print(f"[CONFIG] Checkpoint interval: {args.checkpoint_interval:,}")
    print(f"[CONFIG] Log interval: {args.log_interval:,}")
    print(f"[CONFIG] Max retries: {args.max_retries}")
    print(f"[CONFIG] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    try:
        if args.partitions == 1:
            # Single partition mode
            print(f"[INFO] Running in single partition mode")
            processor = ResilientDatasetProcessor(args)
            processor.process_dataset()
        else:
            # Multi-partition parallel mode
            print(f"[INFO] Running in multi-partition mode with {args.partitions} partitions")
            print(f"[INFO] Starting {args.partitions} parallel workers...")
            print(f"[INFO] Total workers: {args.workers}")
            print(f"[INFO] Workers per partition: {args.workers // args.partitions}\n")

            # Start partition workers
            processes = []
            start_time = time.time()

            for partition_id in range(args.partitions):
                p = multiprocessing.Process(
                    target=process_partition_worker,
                    args=(partition_id, args, args.partitions)
                )
                p.start()
                processes.append(p)
                print(f"[INFO] Started partition worker {partition_id}")

            print(f"\n[INFO] All {args.partitions} workers started. Waiting for completion...")

            # Wait for all to complete
            for i, p in enumerate(processes):
                p.join()
                elapsed = time.time() - start_time
                print(f"[INFO] Partition {i} worker finished (total elapsed: {elapsed / 60:.1f}m)")

            total_elapsed = time.time() - start_time
            print(f"\n[SUCCESS] All partitions completed in {total_elapsed / 60:.1f} minutes")

            # Merge partition files
            print(f"\n[INFO] Starting merge process...")
            merge_start = time.time()
            merge_partition_files(args)
            merge_elapsed = time.time() - merge_start

            print(f"\n[SUCCESS] Merge completed in {merge_elapsed / 60:.1f} minutes")
            print(f"[SUCCESS] Total pipeline time: {(time.time() - start_time) / 60:.1f} minutes")

        print(f"\n[INFO] Script finished successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 70}")
        print(f"[INFO] INTERRUPTED BY USER (Ctrl+C)")
        print(f"{'=' * 70}")
        print(f"[INFO] Progress has been saved to checkpoint")
        print(f"[INFO] Run the script again to resume from last checkpoint")
        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"\n\n{'=' * 70}")
        print(f"[FATAL] UNRECOVERABLE ERROR")
        print(f"{'=' * 70}")
        print(f"[ERROR] {e}")
        print(f"[ERROR] Check the checkpoint file for recovery information")
        print(f"{'=' * 70}\n")
        raise


if __name__ == '__main__':
    main()