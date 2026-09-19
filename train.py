#!/usr/bin/env python3
"""
train.py - train a small GPT from random initialisation on the FedSpeak corpus.

A deliberately simplified rewrite of nanoGPT's train.py: one process, one
device, no wandb, no torch.compile, no gradient accumulation. Everything that
matters for understanding pretraining is still here:

  1. sample random (x, y) windows from train.bin, y = x shifted one char right
  2. forward pass -> cross-entropy over next-character predictions
  3. backward pass, gradient clipping, AdamW step
  4. cosine learning-rate schedule with linear warmup
  5. periodic evaluation on held-out val.bin, checkpoint on improvement

Logs:
  out/step_log.csv   iter, train_loss, lr, ms_per_iter        (every --log_interval)
  out/eval_log.csv   iter, train_loss, val_loss, elapsed_s     (every --eval_interval)
  out/ckpt.pt        best-val-loss checkpoint (model weights + config + vocab)
"""

import argparse
import csv
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ---------------------------------------------------------------------------
# Config. Defaults are nanoGPT's shakespeare_char settings adjusted for a
# corpus ~30x larger (dropout off: there is no overfitting risk at 1-2 epochs).
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out_dir", default="out")
ap.add_argument("--max_iters", type=int, default=2000)
ap.add_argument("--eval_interval", type=int, default=200)
ap.add_argument("--eval_iters", type=int, default=40, help="batches averaged per evaluation")
ap.add_argument("--log_interval", type=int, default=10)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--block_size", type=int, default=256, help="context length in characters")
ap.add_argument("--n_layer", type=int, default=6)
ap.add_argument("--n_head", type=int, default=6)
ap.add_argument("--n_embd", type=int, default=384)
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--learning_rate", type=float, default=1e-3)
ap.add_argument("--min_lr", type=float, default=1e-4)
ap.add_argument("--warmup_iters", type=int, default=100)
ap.add_argument("--weight_decay", type=float, default=0.1)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--device", default=None, help="mps | cuda | cpu (default: best available)")
ap.add_argument("--save_every_eval", action="store_true",
                help="also keep a checkpoint at every evaluation (ckpt_0200.pt, ...) to sample from later")
args = ap.parse_args()

if args.device is None:
    args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
torch.manual_seed(args.seed)
out_dir = HERE / args.out_dir
out_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data. The .bin files are flat arrays of uint16 character ids. A "batch" is
# batch_size random windows of block_size+1 characters; the model sees the
# first block_size (x) and is scored on predicting the next one at every
# position (y). np.memmap means the 66 MB train file is never fully loaded.
# ---------------------------------------------------------------------------
train_data = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
meta = pickle.load((DATA / "meta.pkl").open("rb"))
vocab_size = meta["vocab_size"]


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    starts = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in starts])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model and optimiser.
# ---------------------------------------------------------------------------
model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  block_size=args.block_size, bias=False, vocab_size=vocab_size, dropout=args.dropout)
model = GPT(GPTConfig(**model_args)).to(device)
optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.99), args.device)
print(f"device={device}  params={model.get_num_params() / 1e6:.2f}M  vocab={vocab_size}  "
      f"tokens/iter={args.batch_size * args.block_size:,}  "
      f"epochs planned={args.max_iters * args.batch_size * args.block_size / len(train_data):.2f}")


def get_lr(it: int) -> float:
    """Linear warmup, then cosine decay from learning_rate down to min_lr."""
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    if it >= args.max_iters:
        return args.min_lr
    progress = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss() -> dict:
    """Average loss over several batches of each split. Dropout off during eval."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
step_file = (out_dir / "step_log.csv").open("w", newline="", buffering=1)   # line-buffered
step_log = csv.writer(step_file)
step_log.writerow(["iter", "train_loss", "lr", "ms_per_iter"])
eval_file = (out_dir / "eval_log.csv").open("w", newline="", buffering=1)
eval_log = csv.writer(eval_file)
eval_log.writerow(["iter", "train_loss", "val_loss", "elapsed_s"])

best_val_loss = float("inf")
t_start = t_last = time.time()

for it in range(args.max_iters + 1):
    lr = get_lr(it)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if it % args.eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - t_start
        print(f"--- eval @ {it}: train {losses['train']:.4f}  val {losses['val']:.4f}  ({elapsed / 60:.1f} min)", flush=True)
        eval_log.writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.0f}"])
        checkpoint = {"model": model.state_dict(), "model_args": model_args, "iter_num": it,
                      "best_val_loss": min(best_val_loss, losses["val"]), "train_args": vars(args), "meta": meta}
        if args.save_every_eval:
            torch.save(checkpoint, out_dir / f"ckpt_{it:04d}.pt")
        if losses["val"] < best_val_loss and it > 0:
            best_val_loss = losses["val"]
            torch.save(checkpoint, out_dir / "ckpt.pt")
    if it == args.max_iters:
        break

    # The actual learning step.
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    if it % args.log_interval == 0:
        now = time.time()
        ms = (now - t_last) * 1000 / args.log_interval
        t_last = now
        print(f"iter {it}: loss {loss.item():.4f}  lr {lr:.2e}  {ms:.0f} ms/iter", flush=True)
        step_log.writerow([it, f"{loss.item():.4f}", f"{lr:.2e}", f"{ms:.0f}"])

print(f"done. best val loss {best_val_loss:.4f}; checkpoint at {out_dir / 'ckpt.pt'}")
