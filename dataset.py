"""
dataset.py — Dataset Loader & Preprocessing
Supports:
  - Raw .txt files (any plain text)
  - WikiText-2 / WikiText-103 (via HuggingFace datasets)
  - The Pile (streaming, via HuggingFace datasets)
  - Any HuggingFace text dataset

How it works:
  1. Load raw text from source.
  2. Tokenize the entire corpus into a flat token ID array.
  3. Slice into fixed-length chunks of (seq_len + 1) tokens.
  4. Input = chunk[:-1], Target = chunk[1:]  (next-token prediction).
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer


# ─────────────────────────────────────────────
# Core Dataset
# ─────────────────────────────────────────────

class TokenizedDataset(Dataset):
    """
    Pre-tokenized flat array sliced into fixed-length windows.
    All tokenization happens once at construction time.
    """

    def __init__(
        self,
        token_ids: np.ndarray,
        seq_len: int,
        pad_id: int = 0,
    ):
        self.data = token_ids
        self.seq_len = seq_len
        self.pad_id = pad_id
        # Number of complete chunks
        self.n_chunks = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        # Pad if last chunk is short (shouldn't happen with n_chunks calc above)
        if len(chunk) < self.seq_len + 1:
            pad = np.full(self.seq_len + 1 - len(chunk), self.pad_id, dtype=np.int32)
            chunk = np.concatenate([chunk, pad])
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "targets": y}


# ─────────────────────────────────────────────
# Data Loading Helpers
# ─────────────────────────────────────────────

def load_text_file(path: str) -> str:
    """Load a plain .txt file."""
    print(f"[Dataset] Loading text file: {path}")
    return Path(path).read_text(encoding="utf-8", errors="replace")


def load_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    max_samples: Optional[int] = None,
    streaming: bool = False,
) -> List[str]:
    """
    Load text from a HuggingFace dataset.

    Examples:
      load_huggingface_dataset("wikitext", "wikitext-2-raw-v1")
      load_huggingface_dataset("EleutherAI/pile", streaming=True, max_samples=50000)
      load_huggingface_dataset("openwebtext")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(f"[Dataset] Loading {dataset_name}/{split} from HuggingFace...")
    ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)

    texts = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        t = sample.get(text_field, "")
        if t and t.strip():
            texts.append(t)

    print(f"[Dataset] Loaded {len(texts):,} documents.")
    return texts


# ─────────────────────────────────────────────
# Build Dataset Pipeline
# ─────────────────────────────────────────────

def build_dataset(
    tokenizer: BPETokenizer,
    source: Union[str, List[str]],
    seq_len: int = 1024,
    train_split: float = 0.95,
    cache_path: Optional[str] = None,
) -> tuple["TokenizedDataset", "TokenizedDataset"]:
    """
    Full pipeline:
      source → text → tokenize → flat array → train/val split → datasets

    Args:
        tokenizer: trained BPETokenizer
        source: path to .txt file, or list of text strings
        seq_len: context window length
        train_split: fraction for training
        cache_path: if given, cache tokenized array to .npy file

    Returns:
        (train_dataset, val_dataset)
    """

    # ── Load / cache token array ──────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        print(f"[Dataset] Loading cached token array from {cache_path}")
        all_ids = np.load(cache_path)
    else:
        # Get raw text
        if isinstance(source, str) and os.path.isfile(source):
            text = load_text_file(source)
            texts = [text]
        elif isinstance(source, list):
            texts = source
        else:
            raise ValueError(f"source must be a file path or list of strings, got: {type(source)}")

        # Tokenize
        print(f"[Dataset] Tokenizing {len(texts):,} documents...")
        all_ids_list: List[int] = []
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_ids_list.extend(ids)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(texts)} docs | {len(all_ids_list):,} tokens")

        all_ids = np.array(all_ids_list, dtype=np.int32)
        print(f"[Dataset] Total tokens: {len(all_ids):,}")

        if cache_path:
            np.save(cache_path, all_ids)
            print(f"[Dataset] Cached to {cache_path}")

    # ── Split ─────────────────────────────────────────────────────────
    split_idx = int(len(all_ids) * train_split)
    train_ids = all_ids[:split_idx]
    val_ids = all_ids[split_idx:]

    train_ds = TokenizedDataset(train_ids, seq_len, pad_id=tokenizer.pad_id)
    val_ds = TokenizedDataset(val_ids, seq_len, pad_id=tokenizer.pad_id)

    print(f"[Dataset] Train chunks: {len(train_ds):,}  |  Val chunks: {len(val_ds):,}")
    return train_ds, val_ds


def create_dataloaders(
    train_ds: "TokenizedDataset",
    val_ds: "TokenizedDataset",
    batch_size: int = 8,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Wrap datasets in DataLoaders with shuffle and pinned memory."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Sample tiny corpus for quick testing
# ─────────────────────────────────────────────

SAMPLE_TEXT = """
The transformer architecture was introduced in the paper "Attention Is All You Need" in 2017.
It relies entirely on an attention mechanism to draw global dependencies between input and output.
Unlike recurrent neural networks, transformers process all tokens in parallel during training.

Language models are trained to predict the next token in a sequence.
Given a prompt, they generate text by sampling from the probability distribution over the vocabulary.
Temperature controls randomness: lower values make output more focused, higher values more creative.

Deep learning has revolutionized natural language processing. Models like GPT, BERT, and T5
have achieved state-of-the-art results on a wide variety of tasks including translation,
summarization, question answering, and code generation.

The key insight of attention is that tokens can directly attend to any other token in the sequence,
regardless of their distance. This allows the model to capture long-range dependencies efficiently.
""" * 200  # Repeat to create a decently sized sample corpus


if __name__ == "__main__":
    from tokenizer import BPETokenizer

    # Train a small tokenizer on sample text
    corpus = [line for line in SAMPLE_TEXT.split("\n") if line.strip()]
    tok = BPETokenizer(vocab_size=2000)
    tok.train(corpus, verbose=False)

    # Build dataset
    train_ds, val_ds = build_dataset(
        tokenizer=tok,
        source=corpus,
        seq_len=64,
        train_split=0.9,
    )

    train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    print("Input shape:", batch["input_ids"].shape)
    print("Target shape:", batch["targets"].shape)
    print("Sample IDs:", batch["input_ids"][0][:16])
