#!/usr/bin/env python3
"""
eval_common_context.py - compare models trained at different context lengths
on identical predictions.

The chunk-size sweep scores each model in windows of its own block size, which
conflates two different things: whether long context helps a model *learn*, and
whether long context helps it *predict*. A model trained at 16 characters is
also only ever tested with 16 characters of context.

This scores every model on exactly the same target characters, each using as
much left-context as it has. For a stride S and a grid of positions t:

    window  = text[t - B : t]        B = that model's block size
    scored  = the last S predictions in the window, i.e. text[t-S+1 : t+1]

Every model is asked to predict the same characters; the only difference is how
much history it was allowed to condition on. With S small relative to the
shortest block size, each model uses close to its full context on every scored
position.

Writes results/common_context_eval.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--runs", default="ctx0016,ctx0032,ctx0064,ctx0128,ctx0256,ctx0512,ctx1024")
ap.add_argument("--chars", type=int, default=60000, help="validation characters covered by the grid")
ap.add_argument("--stride", type=int, default=4, help="predictions scored per window")
ap.add_argument("--start", type=int, default=1024, help="first scored position; must be >= the largest block size")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--device", default=None)
ap.add_argument("--out", default="results/common_context_eval.csv")
args = ap.parse_args()
if args.device is None:
    args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)

val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
S = args.stride
grid = list(range(args.start, min(args.start + args.chars, len(val) - 1), S))
print(f"scoring {len(grid) * S:,} identical target characters ({len(grid):,} windows of stride {S})")

rows = []
for name in args.runs.split(","):
    ckpt_path = HERE / "runs" / name / "ckpt.pt"
    if not ckpt_path.exists():
        print(f"{name}: no checkpoint, skipped")
        continue
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    B = ckpt["model_args"]["block_size"]
    model = GPT(GPTConfig(**ckpt["model_args"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    assert B >= S, "stride must fit inside the block"

    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(grid), args.batch_size):
            chunk = grid[i:i + args.batch_size]
            x = torch.stack([torch.from_numpy(val[t - B:t].astype(np.int64)) for t in chunk]).to(device)
            # Targets for the whole window, then everything but the last S
            # positions set to -1. nanoGPT's loss uses ignore_index=-1, so only
            # the scored positions contribute, and it skips the inference-time
            # shortcut that would otherwise return logits for one position only.
            y = torch.stack([torch.from_numpy(val[t - B + 1:t + 1].astype(np.int64)) for t in chunk]).to(device)
            y[:, :B - S] = -1
            _, loss = model(x, y)                      # mean over the non-ignored positions
            n = len(chunk) * S
            total += loss.item() * n
            count += n
    own = ckpt["best_val_loss"]
    rows.append([name, B, f"{total / count:.4f}", f"{own:.4f}", count])
    print(f"{name:<9} block {B:>5}  common-grid loss {total / count:.4f}  (its own-window loss was {own:.4f})", flush=True)

with (HERE / args.out).open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "block_size", "common_grid_loss", "own_window_loss", "chars_scored"])
    w.writerows(rows)
print("wrote", args.out)
