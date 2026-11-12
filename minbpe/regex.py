"""
Minimal (byte-level) BPE tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer, RegexTokenizer handles regex splitting and special tokens.
"""

import regex as re

from .base import Tokenizer, get_stats, merge

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """Regex tokenizer"""

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}  # int -> str

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on text.

        Carries out merges on the text using the BPE algorithm until the tokenizer's
        vocab size reaches vocab_size. Splits the text using self.pattern before
        carrying out the merges.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk in ids:
                # Updates stats in place
                get_stats(chunk, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk, pair, idx) for chunk in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        """
        Update self.special_tokens and self.inverse_special_tokens given special_tokens.

        special_tokens is a dictionary of str -> int.
        Example: {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """
        Return the corresponding string from a list of ints representing vocab indices.

        Note that it's possible for the decoding to be lossy due to the byte stream
        corresponding to the list of ints not following utf-8 schema.
        """
        token_bytes = []
        for idx in ids:
            if idx in self.vocab:
                token_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                token_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        text_bytes = b"".join(token_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text, allowed_special="none_raise"):
        """
        Return the list of ints corresponding to text.

        Args:
            text: The text that is processed.
            allowed_special: Specifies how special tokens should be handled. Can be
                "all", "none", "none_raise", or a custom set of special tokens.

        Returns:
            The list of integers corresponding to the text. Each integer represents the
            index of the corresponding token.

        Raises:
            ValueError: If allowed_special is not one of "all", "none", "none_raise", or
            a set; or if allowed_special is "none_raise" and text contains a special
            token.
        """
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(
                        f"Special token {token!r} found in text with allowed_special='none_raise'"
                    )
        elif isinstance(allowed_special, set):
            special = {
                k: v for k, v in self.special_tokens.items() if k in allowed_special
            }
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)

        # Separate the special tokens so they don't get merged with other tokens
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_ordinary(self, text):
        """
        Return the list of ints corresponding to text.

        Treats any special tokens in text as a non-special token.
        """
        text_chunks = re.findall(self.pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
