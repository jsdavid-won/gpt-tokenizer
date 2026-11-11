from collections import Counter
from itertools import pairwise


def get_counts(tokens):
    return Counter(pairwise(tokens))


def merge(pair, idx, tokens):
    out = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
