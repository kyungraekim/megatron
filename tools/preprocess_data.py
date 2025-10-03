# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining with HuggingFace datasets support."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

try:
    from datasets import load_from_disk, load_dataset, Dataset
    import pyarrow.parquet as pq

    hf_available = True
except ImportError:
    hf_available = False

from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer as build_new_tokenizer
from megatron.core.tokenizers import MegatronTokenizer
from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        if self.args.legacy_tokenizer:
            tokenizer = build_tokenizer(self.args)
        else:
            Encoder.tokenizer = build_new_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params,
                    lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter
        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i + max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)

    def encode_hf_sample(self, sample):
        """HuggingFace 샘플을 직접 인코딩"""
        ids = {}
        lens = {}
        total_bytes = 0

        for key in self.args.json_keys:
            if key not in sample:
                continue

            text = sample[key]
            total_bytes += len(str(text).encode('utf-8'))

            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]

            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))

            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1

            ids[key] = doc_ids
            lens[key] = sentence_lens

        return ids, lens, total_bytes


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents",
                  f"({count / elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        if self.args.legacy_tokenizer:
            tokenizer = build_tokenizer(self.args)
        else:
            tokenizer = build_new_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        for key in builders.keys():
            builders[key].finalize(output_idx_files[key])

    def process_hf_dataset(self, dataset_info):
        """HuggingFace 데이터셋을 직접 처리 (JSONL 중간 저장 없이)"""
        dataset, start_idx, end_idx, output_prefix = dataset_info

        print(f"Processing HF dataset partition: samples {start_idx} to {end_idx}")

        startup_start = time.time()
        encoder = Encoder(self.args)
        if self.args.legacy_tokenizer:
            tokenizer = build_tokenizer(self.args)
        else:
            tokenizer = build_new_tokenizer(self.args)

        # 인코더 초기화
        encoder.initializer()

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        # 데이터셋 직접 순회 및 처리
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            doc, sentence_lens, bytes_processed = encoder.encode_hf_sample(sample)

            total_bytes_processed += bytes_processed
            for key in doc.keys():
                if len(doc[key]) > 0:  # 빈 문서 스킵
                    builders[key].add_document(doc[key], sentence_lens[key])

            self.print_processing_stats(i - start_idx + 1, proc_start, total_bytes_processed)

        for key in builders.keys():
            builders[key].finalize(output_idx_files[key])

        print(f"Partition {output_prefix} complete!")


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)

    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str,
                       help='Path to input JSON/JSONL file (for legacy mode)')
    group.add_argument('--hf-dataset-path', type=str,
                       help='Path to HuggingFace dataset (local directory or Hub path)')
    group.add_argument('--hf-split', type=str, default='train',
                       help='Dataset split to use (e.g., train, validation, test)')
    group.add_argument('--hf-cache-dir', type=str, default=None,
                       help='Cache directory for HuggingFace datasets')
    group.add_argument('--hf-streaming', action='store_true',
                       help='Use streaming mode for HuggingFace datasets (for very large datasets)')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='Space separated list of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument('--legacy-tokenizer', action='store_true',
                       help='Use legacy tokenizer system.')

    args = parser.parse_args()
    args.keep_empty = False

    # HuggingFace 모드와 일반 모드 중 하나는 반드시 지정되어야 함
    if not args.hf_dataset_path and not args.input:
        parser.error("Either --hf-dataset-path or --input must be specified")

    if args.hf_dataset_path and not hf_available:
        parser.error("HuggingFace datasets library is not installed. "
                     "Install with: pip install datasets pyarrow")

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    if args.input:
        file_name, extension = os.path.splitext(args.input)
        file_name = file_name + "_" + str(file_id)
    else:
        file_name = f"hf_dataset_{file_id}"
        extension = ".jsonl"

    input_file_name = file_name + extension
    sentence_split_file = file_name + "_ss" + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def process_hf_dataset(args):
    """HuggingFace 데이터셋을 직접 처리 (메모리 효율적)"""
    print(f"Loading HuggingFace dataset from {args.hf_dataset_path}")

    # 로컬에 저장된 데이터셋 로드
    if os.path.isdir(args.hf_dataset_path):
        dataset = load_from_disk(args.hf_dataset_path)
        # split 선택
        if isinstance(dataset, dict):
            if args.hf_split not in dataset:
                raise ValueError(f"Split '{args.hf_split}' not found. Available: {list(dataset.keys())}")
            dataset = dataset[args.hf_split]
    else:
        # HuggingFace Hub에서 로드
        dataset = load_dataset(
            args.hf_dataset_path,
            split=args.hf_split,
            cache_dir=args.hf_cache_dir,
            streaming=args.hf_streaming
        )

    total_samples = len(dataset)
    print(f"Dataset loaded. Total samples: {total_samples}")

    # 데이터셋 필드 확인
    sample = dataset[0]
    available_keys = list(sample.keys())
    print(f"Available keys in dataset: {available_keys}")

    for key in args.json_keys:
        if key not in available_keys:
            print(f"Warning: Key '{key}' not found in dataset. Available keys: {available_keys}")

    partition = Partition(args, args.workers // args.partitions)

    if args.partitions == 1:
        # 단일 파티션 처리
        partition.process_hf_dataset((dataset, 0, total_samples, args.output_prefix))
    else:
        # 멀티 파티션 처리
        partition_size = math.ceil(total_samples / args.partitions)
        processes = []

        for idx in range(args.partitions):
            start_idx = idx * partition_size
            end_idx = min((idx + 1) * partition_size, total_samples)
            output_prefix = f"{args.output_prefix}_{idx}"

            p = multiprocessing.Process(
                target=partition.process_hf_dataset,
                args=((dataset, start_idx, end_idx, output_prefix),)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 파티션 병합
        print("Merging partitions...")
        level = "document"
        if args.split_sentences:
            level = "sentence"

        if args.legacy_tokenizer:
            tokenizer = build_tokenizer(args)
        else:
            tokenizer = build_new_tokenizer(args)

        for key in args.json_keys:
            output_bin_file = "{}_{}_{}.bin".format(args.output_prefix, key, level)
            output_idx_file = "{}_{}_{}.idx".format(args.output_prefix, key, level)

            builder = indexed_dataset.IndexedDatasetBuilder(
                output_bin_file,
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

            for idx in range(args.partitions):
                partition_prefix = f"{args.output_prefix}_{idx}"
                full_partition_prefix = "{}_{}_{}".format(partition_prefix, key, level)
                builder.add_index(full_partition_prefix)

            builder.finalize(output_idx_file)

        print("Merge complete!")


def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    # HuggingFace 데이터셋 직접 처리
    if args.hf_dataset_path:
        process_hf_dataset(args)
        return

    # 기존 JSONL 처리 로직
    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples: line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if args.partitions == 1:
            return

    # encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    if args.legacy_tokenizer:
        tokenizer = build_tokenizer(args)
    else:
        tokenizer = build_new_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()