"""
Train a BPE tokenizer using HuggingFace's tokenizers library.
In the style of GPT-4 tokenizer.
"""

import os
import time
import argparse
from pathlib import Path

import torch
import pyarrow.parquet as pq
from tokenizers import Tokenizer, pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# -----------------------------------------------------------------------------
# Configuration

DATA_DIR = os.environ.get("DATA_DIR", "/resource/data")

DATASETS = {
    "fineweb": f"{DATA_DIR}/fineweb-edu-100b-shuffle",
    "fineweb-350bt": f"{DATA_DIR}/fineweb-edu-350b-shuffle",
    "huginn": f"{DATA_DIR}/huginn-dataset",
}

# Special tokens for the tokenizer
SPECIAL_TOKENS = [
    "<|bos|>",  # beginning of sequence
    "<|eos|>",  # end of sequence
    "<|pad|>",  # padding token
]

# GPT-4 style split pattern
# NOTE: Using \p{N}{1,2} instead of \p{N}{1,3} to avoid wasting tokens on numbers for smaller vocabs
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def resolve_data_dir(data_dir):
    """Resolve dataset name to path, or return as-is if it's already a path."""
    if data_dir in DATASETS:
        return DATASETS[data_dir]
    return data_dir

# -----------------------------------------------------------------------------
# Data loading utilities


def list_parquet_files(data_dir):
    """Returns full paths to all parquet files in a directory."""
    data_dir = Path(data_dir)
    parquet_files = sorted(data_dir.glob("*.parquet"))
    return [str(f) for f in parquet_files if not str(f).endswith('.tmp')]


def parquets_iter_batched(data_dir, split="train", start=0, step=1):
    """
    Iterate through parquet files, yielding batches of text.

    Args:
        data_dir: Directory containing parquet files
        split: "train" or "val" - val uses only the last file
        start: Starting row group index (for DDP)
        step: Step between row groups (for DDP)
    """
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        return

    if split == "val":
        parquet_paths = parquet_paths[-1:]
    else:
        parquet_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths

    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            if 'text' in rg.column_names:
                texts = rg.column('text').to_pylist()
                yield texts


# -----------------------------------------------------------------------------
# Tokenizer training


def create_gpt4_style_tokenizer():
    """Create a GPT-4 style BPE tokenizer configuration."""
    tokenizer = Tokenizer(BPE(
        byte_fallback=True,
        unk_token=None,
        fuse_unk=False,
    ))

    # No normalizer (like GPT-4)
    tokenizer.normalizer = None

    # GPT-4 style pre-tokenizer
    gpt4_split_regex = Regex(SPLIT_PATTERN)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])

    # ByteLevel decoder (pairs with ByteLevel pre-tokenizer)
    tokenizer.decoder = decoders.ByteLevel()

    # No post-processor
    tokenizer.post_processor = None

    return tokenizer


def train_tokenizer(text_iterator, vocab_size):
    """Train a BPE tokenizer from a text iterator."""
    tokenizer = create_gpt4_style_tokenizer()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train_from_iterator(text_iterator, trainer)
    return tokenizer


# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
    parser.add_argument('--data-dir', type=str, required=True,
                        help=f'Dataset name ({", ".join(DATASETS.keys())}) or path to parquet directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save the trained tokenizer')
    parser.add_argument('--max-chars', type=int, default=2_000_000_000,
                        help='Maximum characters to train on (default: 2B)')
    parser.add_argument('--doc-cap', type=int, default=10_000,
                        help='Maximum characters per document (default: 10,000)')
    parser.add_argument('--vocab-size', type=int, default=32768,
                        help='Vocabulary size (default: 32768 = 2^15)')
    args = parser.parse_args()

    # Resolve dataset name to path if needed
    data_dir = resolve_data_dir(args.data_dir)

    print(f"Training tokenizer with:")
    print(f"  data_dir: {data_dir}" + (f" ({args.data_dir})" if args.data_dir in DATASETS else ""))
    print(f"  output_dir: {args.output_dir}")
    print(f"  max_chars: {args.max_chars:,}")
    print(f"  doc_cap: {args.doc_cap:,}")
    print(f"  vocab_size: {args.vocab_size:,}")
    print()

    # -------------------------------------------------------------------------
    # Text iterator

    def text_iterator():
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Break when we've seen args.max_chars characters
        """
        nchars = 0
        for batch in parquets_iter_batched(data_dir, split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    print(f"Reached {nchars:,} characters, stopping...")
                    return

    # -------------------------------------------------------------------------
    # Train the tokenizer

    print("Starting tokenizer training...")
    t0 = time.time()
    tokenizer = train_tokenizer(text_iterator(), args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save the tokenizer to disk

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # -------------------------------------------------------------------------
    # Quick inline sanity check

    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""

    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    if decoded == test_text:
        print("✓ Roundtrip test passed!")
    else:
        print(f"✗ Roundtrip test failed!")
        print(f"  Original: {repr(test_text)}")
        print(f"  Decoded:  {repr(decoded)}")

    # -------------------------------------------------------------------------
    # Cache token bytes mapping for bits-per-byte evaluation

    vocab_size = tokenizer.get_vocab_size()
    special_set = set(SPECIAL_TOKENS)
    token_bytes = []

    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)  # special tokens don't count
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(args.output_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # -------------------------------------------------------------------------
    # Print summary statistics

    token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
    print()
    print("Summary:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Num special tokens: {len(special_set)}")
    print(f"  Token bytes (non-special):")
    print(f"    min:  {int(token_bytes_nonzero.min().item())}")
    print(f"    max:  {int(token_bytes_nonzero.max().item())}")
    print(f"    mean: {token_bytes_nonzero.mean().item():.2f}")
    print(f"    std:  {token_bytes_nonzero.std().item():.2f}")
