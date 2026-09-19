#!/usr/bin/env python3
"""
eval_by_source.py - how hard is each source for the trained model?

Scores the checkpoint separately on the validation portion of each source.
The validation set is the last 10% of the date-ordered corpus, so this script
finds the date where that split begins and evaluates every clean document from
each source on or after it, in non-overlapping windows of block_size.

Loss is mean cross-entropy in nats per character; lower is easier to predict.
"""

import argparse
import csv
from pathlib import Path

import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SOURCES = ("statements", "minutes", "beigebook")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/ckpt.pt")
ap.add_argument("--out", default="results/loss_by_source.csv")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--device", default=None)
args = ap.parse_args()
if args.device is None:
    args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)

ckpt = torch.load(HERE / args.ckpt, map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt["model_args"])).to(device).eval()
model.load_state_dict(ckpt["model"])
stoi = ckpt["meta"]["stoi"]
block = ckpt["model_args"]["block_size"]

# Where does the validation split start? Walk the same date-ordered document
# list that scrape_fed.py concatenated and find the document containing the
# 90% character boundary.
docs = sorted(DATA.glob("clean/*/*.txt"), key=lambda p: (p.stem, p.parent.name))
lengths = [len(p.read_text()) + 1 for p in docs]        # +1 for the separator newline
boundary = int(sum(lengths) * 0.9)
running, val_start = 0, None
for p, n in zip(docs, lengths):
    running += n
    if running > boundary:
        val_start = p.stem
        break
print(f"validation split starts at document date {val_start}")


@torch.no_grad()
def score(text: str) -> tuple[float, int]:
    ids = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    n_windows = (len(ids) - 1) // block
    ids = ids[: n_windows * block + 1]
    x = ids[:-1].view(n_windows, block)
    y = ids[1:].view(n_windows, block)
    total, count = 0.0, 0
    for i in range(0, n_windows, args.batch_size):
        xb, yb = x[i:i + args.batch_size].to(device), y[i:i + args.batch_size].to(device)
        _, loss = model(xb, yb)
        total += loss.item() * xb.numel()
        count += xb.numel()
    return total, count


rows = []
for src in SOURCES:
    val_docs = [p for p in sorted((DATA / "clean" / src).glob("*.txt")) if p.stem >= val_start]
    total, count = 0.0, 0
    for p in val_docs:
        t, c = score(p.read_text())
        total += t; count += c
    loss = total / count
    rows.append([src, len(val_docs), count, f"{loss:.4f}", f"{loss / 0.6931:.3f}"])
    print(f"{src:<11} {len(val_docs):3d} docs  {count:>10,} chars scored  loss {loss:.4f} nats/char  = {loss/0.6931:.3f} bits/char")

with (HERE / args.out).open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "val_docs", "chars_scored", "loss_nats_per_char", "bits_per_char"])
    w.writerows(rows)
print("wrote", args.out)
