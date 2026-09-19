#!/usr/bin/env python3
"""
prepare_ft.py - tokenise the FedSpeak corpus for fine-tuning a pretrained model.

Differences from the from-scratch prepare.py:
  * BPE tokens from the pretrained model's tokenizer, not characters
  * every document gets a one-line header, e.g.  [FOMC statement | 2010-01-27]
    so the fine-tuned model can be *asked* for a given source and date
  * documents are separated by the tokenizer's end-of-text token

Writes data/ft/train.bin, data/ft/val.bin (uint16 token ids) and data/ft/meta.json.
The split is chronological, same rule as before: last ~10% of tokens by date.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
CLEAN = HERE / "data" / "clean"
OUT = HERE / "data" / "ft"
LABELS = {"statements": "FOMC statement", "minutes": "FOMC minutes", "beigebook": "Beige Book"}


def header(source: str, date: str) -> str:
    return f"[{LABELS[source]} | {date[:4]}-{date[4:6]}-{date[6:]}]\n"


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
ap.add_argument("--train_fraction", type=float, default=0.9)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
assert len(tok) < 65536, "uint16 storage assumes a vocabulary under 65,536"
eos = tok.eos_token_id

docs = sorted(CLEAN.glob("*/*.txt"), key=lambda p: (p.stem, p.parent.name))
ids_per_doc, sources = [], []
for p in docs:
    text = header(p.parent.name, p.stem) + p.read_text()
    ids_per_doc.append(tok(text)["input_ids"] + [eos])
    sources.append(p.parent.name)

total = sum(map(len, ids_per_doc))
cut = int(total * args.train_fraction)
running, split_at = 0, None
for i, ids in enumerate(ids_per_doc):
    running += len(ids)
    if running > cut:
        split_at = i
        break

train = np.array([t for ids in ids_per_doc[:split_at] for t in ids], dtype=np.uint16)
val = np.array([t for ids in ids_per_doc[split_at:] for t in ids], dtype=np.uint16)
OUT.mkdir(parents=True, exist_ok=True)
train.tofile(OUT / "train.bin")
val.tofile(OUT / "val.bin")
meta = {"model": args.model, "vocab_size": len(tok), "eos_token_id": eos,
        "val_starts_at_doc": docs[split_at].stem, "train_tokens": int(len(train)), "val_tokens": int(len(val)),
        "header_example": header("statements", "20260128").strip(),
        "val_docs": [f"{p.parent.name}/{p.stem}" for p in docs[split_at:]]}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"{len(docs)} documents -> {total:,} tokens; train {len(train):,} / val {len(val):,}; "
      f"validation starts at {meta['val_starts_at_doc']}; header looks like {meta['header_example']!r}")
