"""
tokenizer.py — Byte-Pair Encoding (BPE) Tokenizer
Trains on raw text, builds a vocabulary, and encodes/decodes sequences.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BPETokenizer:
    """
    A from-scratch BPE tokenizer.
    
    How it works:
      1. Split text into characters (unicode-safe via UTF-8 bytes).
      2. Iteratively merge the most frequent adjacent pair of tokens.
      3. After `vocab_size` merges, we have a final vocabulary.
      4. Encode text by greedily applying learned merges.
    """

    # Special tokens
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_vocab(self, corpus: List[str]) -> Dict[str, int]:
        """Convert corpus sentences to character-level token frequency map."""
        vocab: Dict[str, int] = defaultdict(int)
        for text in corpus:
            # Represent each word as space-separated characters + end-of-word marker
            words = text.strip().split()
            for word in words:
                chars = " ".join(list(word)) + " </w>"
                vocab[chars] += 1
        return dict(vocab)

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent token pairs across the vocab."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return dict(pairs)

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Replace all occurrences of `pair` with merged token in vocab."""
        new_vocab: Dict[str, int] = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        merged = "".join(pair)
        for word, freq in vocab.items():
            new_word = pattern.sub(merged, word)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, corpus: List[str], verbose: bool = True):
        """Train BPE on a list of text strings."""
        print(f"[Tokenizer] Training BPE with vocab_size={self.vocab_size} on {len(corpus)} documents...")

        vocab = self._get_vocab(corpus)

        # Seed token set: all unique characters
        all_tokens: set = set()
        for word in vocab:
            all_tokens.update(word.split())

        # Reserve IDs for special tokens first
        idx = 0
        for sp in self._special_tokens:
            self.token_to_id[sp] = idx
            self.id_to_token[idx] = sp
            idx += 1

        for tok in sorted(all_tokens):
            if tok not in self.token_to_id:
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
                idx += 1

        num_merges = self.vocab_size - len(self.token_to_id)
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            merged = "".join(best)
            self.merges[best] = merged
            if merged not in self.token_to_id:
                self.token_to_id[merged] = idx
                self.id_to_token[idx] = merged
                idx += 1
            if verbose and (i + 1) % 500 == 0:
                print(f"  Merge {i+1}/{num_merges}: {best} → {merged}")

        print(f"[Tokenizer] Done. Vocabulary size: {len(self.token_to_id)}")

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        symbols = list(word) + ["</w>"]
        for (a, b), merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols = symbols[:i] + [merged] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert string to list of token IDs."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.token_to_id[self.BOS_TOKEN])
        unk_id = self.token_to_id[self.UNK_TOKEN]
        for word in text.strip().split():
            for sym in self._tokenize_word(word):
                tokens.append(self.token_to_id.get(sym, unk_id))
        if add_special_tokens:
            tokens.append(self.token_to_id[self.EOS_TOKEN])
        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back to string."""
        special_ids = {self.token_to_id[t] for t in self._special_tokens}
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            tokens.append(self.id_to_token.get(i, self.UNK_TOKEN))
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        import re
        text = re.sub(r" {2,}", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def vocab_size_actual(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Serialize tokenizer to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": [[list(k), v] for k, v in self.merges.items()],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tok.merges = {(tuple(k)): v for k, v in data["merges"]}
        print(f"[Tokenizer] Loaded from {path}, vocab size: {len(tok.token_to_id)}")
        return tok


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == "__main__":
    sample = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the tokenizer",
        "language models are trained on large text corpora",
        "transformers use attention mechanisms to process sequences",
    ] * 50  # repeat so BPE has enough data

    tok = BPETokenizer(vocab_size=500)
    tok.train(sample, verbose=False)

    text = "the quick brown fox"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Original : {text}")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : {decoded}")
    tok.save("tokenizer.json")
    tok2 = BPETokenizer.load("tokenizer.json")
    print(f"Roundtrip: {tok2.decode(tok2.encode(text))}")