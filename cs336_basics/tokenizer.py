from __future__ import annotations

import os

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Type aliases
Token = int                # internal numeric id
Subword = bytes            # raw byte sequence
Pair   = Tuple[int, int]   # adjacent token pair


class BPETokenizer:
    """Pure‑Python BPE tokenizer ― now with special‑token capability."""

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(
        self,
        vocab: Dict[int, Subword] | None = None,
        merges: List[Pair] | None = None,
        special_tokens: List[str] | None = None,
    ) -> None:
        # 基础词表与 merge 规则
        self.vocab: Dict[int, Subword] = vocab or {}
        self.merges: List[Pair]         = merges or []

        # special token 双向映射（字符串 <-> id）
        self.special_tokens: List[str]          = special_tokens or []
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        # 反向字节查词表
        self._inv_vocab: Dict[Subword, int] = {b: i for i, b in self.vocab.items()}

        # 确保 special token 已插入 vocab
        for tok in self.special_tokens:
            self.add_special_token(tok)

    # ------------------------------------------------------------------
    # Public encode / decode API (签名保持不变)
    # ------------------------------------------------------------------
    def encode(self, string: str) -> List[int]:
        """将 *string* 转成 token id 序列。

        * 若输入整段即为注册的 special token（如 "<pad>"），直接返回对应 id。
        * 否则按字节拆分并应用已学习的 merge 规则。
        """
        # 特殊情况：整段是一个 special token
        if string in self.special_token_to_id:
            return [self.special_token_to_id[string]]

        # 1) 字节级拆分
        indices = [self._inv_vocab[bytes([b])] for b in string.encode("utf-8")]
        # 2) 依次执行 merge 规则
        for pair in self.merges:
            indices = self.merge(indices, pair, self.pair_to_id(pair))
        return indices

    def decode(self, indices: List[int]) -> str:
        """将 token id 序列还原为人类可读文本。
        special id 会被直接替换成其字符串（如 "<eos>")。
        """
        if not indices:
            return ""
        parts: List[str] = []
        for idx in indices:
            if idx in self.id_to_special_token:
                parts.append(self.id_to_special_token[idx])
            else:
                parts.append(self.vocab[idx].decode("utf-8", errors="replace"))
        return "".join(parts)

    

    # ------------------------------------------------------------------
    # Static helpers required by assignment (merge / calc_freq 保留原签名)
    # ------------------------------------------------------------------
    @staticmethod
    def merge(indices: List[int], pair: Pair, new_index: int) -> List[int]:
        new_indices: List[int] = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1
        return new_indices

    @staticmethod
    def calc_freq(dataset: List[List[int]]) -> Counter[Pair]:
        freq: Counter[Pair] = Counter()
        for seq in dataset:
            for a, b in zip(seq, seq[1:]):
                freq[(a, b)] += 1
        return freq

    def train_from_file(self, file_path: str | Path, target_vocab_size: int = 30000) -> None:
        """Load corpus and train *in‑place* until vocab size reaches *target_vocab_size*."""
        text = Path(file_path).read_text(encoding="utf-8")

        # 1) ensure byte‑level base vocab exists
        if not self.vocab:
            for b in sorted(set(text.encode("utf-8"))):
                self.add_token(bytes([b]))

        dataset = [self.encode(story) for story in text.split('<|endoftext|>')]

        # 2) iterative BPE merges
        while len(self.vocab) < target_vocab_size:
            pair_freq = self.calc_freq(dataset)
            if not pair_freq:
                break
            best_pair, _ = max(pair_freq.items(), key=lambda kv: kv[1])
            new_id = self.add_token(self.vocab[best_pair[0]] + self.vocab[best_pair[1]])
            self.merges.append(best_pair)
            dataset = [self.merge(seq, best_pair, new_id) for seq in dataset]
    
    # ------------------------------------------------------------------
    # Token / special-token utilities
    # ------------------------------------------------------------------
    def add_token(self, subword: Subword) -> Token:
        if subword in self._inv_vocab:
            return self._inv_vocab[subword]
        idx = len(self.vocab)
        self.vocab[idx] = subword
        self._inv_vocab[subword] = idx
        return idx

   

    def pair_to_id(self, pair: Pair) -> int:
        return self._inv_vocab[self.vocab[pair[0]] + self.vocab[pair[1]]]

    


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    
    tokenizer = BPETokenizer()
    print("[INFO] Training tokenizer…")
    tokenizer.train_from_file('./tinystory_head_test.txt', 300)

    sample = '''
While Ollie was catching fish, he found a big shiny stone. He thought, "This is not a fish, but it is so pretty!" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner.
'''
    ids = tokenizer.encode(sample)
    print("Round‑trip:")
    print("  text  :", sample)
    print("  ids   :", ids[:20], "…")
    print("  len(ids):",len(ids))
    print("  decode:", tokenizer.decode(ids))

        


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
