"""
Minimal (byte-level) BPE tokenizer.

Algorithmically follows along the GPT tokenizer (mostly):
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike the actual GPT tokenizer, BasicTokenizer does not handle the regex splitting
or any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """Basic BPE tokenizer"""

    def train(self, text, vocab_size, verbose=False):
        """
        Train the tokenizer on text.

        Carries out merges on the text using the BPE algorithm until the tokenizer's
        vocab size reaches vocab_size.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        """
        Return the corresponding string from a list of ints representing vocab indices.

        Note that it's possible for the decoding to be lossy due to the byte stream
        corresponding to the list of ints not following utf-8 schema.
        """
        ids = b"".join([self.vocab[idx] for idx in ids])
        text = ids.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        """
        Returning the list of ints corresponding to text.

        Each integer in the list represents a vocab index.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
