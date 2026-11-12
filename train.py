import os
import time

from minbpe.basic import BasicTokenizer
from minbpe.regex import RegexTokenizer

VOCAB_SIZE = 512

with open("tests/taylorswift.txt") as f:
    text = f.read()

os.makedirs("models", exist_ok=True)

t0 = time.perf_counter()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):
    tokenizer = TokenizerClass()
    tokenizer.train(text, VOCAB_SIZE)
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.perf_counter()

print(f"Training took {t1 - t0:.2f} seconds")
