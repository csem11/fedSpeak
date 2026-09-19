#!/usr/bin/env python3
"""
prepare.py - turn data/fedspeak.txt into training tensors.

Character-level "tokenization": every distinct character in the corpus gets an
integer id. There is no BPE and no vocabulary file to download; the vocabulary
*is* the sorted set of characters in the text. This is the same recipe as
nanoGPT's data/shakespeare_char/prepare.py.

Writes to data/:
  train.bin   uint16 token ids, first 90% of the corpus
  val.bin     uint16 token ids, last 10% (chronologically later documents)
  meta.pkl    vocab_size plus the char<->id maps that sample.py needs to decode
"""

import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
TRAIN_FRACTION = 0.9

text = (DATA / "fedspeak.txt").read_text(encoding="ascii")
print(f"corpus: {len(text):,} characters")

chars = sorted(set(text))
vocab_size = len(chars)
print(f"vocab:  {vocab_size} distinct characters: {''.join(chars)!r}")

stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> string

# The corpus is in date order, so this split is chronological: the model is
# validated on documents written after everything it trained on.
n = int(len(text) * TRAIN_FRACTION)
train_ids = np.array([stoi[c] for c in text[:n]], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in text[n:]], dtype=np.uint16)
print(f"train:  {len(train_ids):,} tokens")
print(f"val:    {len(val_ids):,} tokens")

train_ids.tofile(DATA / "train.bin")
val_ids.tofile(DATA / "val.bin")
with (DATA / "meta.pkl").open("wb") as f:
    pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)
print(f"wrote {DATA / 'train.bin'}, {DATA / 'val.bin'}, {DATA / 'meta.pkl'}")
