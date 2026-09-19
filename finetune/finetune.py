#!/usr/bin/env python3
"""
finetune.py - continue pretraining a small open model on the FedSpeak corpus.

Same loop shape as train.py, with the model swapped for a pretrained one:
  1. random 1024-token windows from data/ft/train.bin
  2. forward in bfloat16 autocast, loss = next-token cross-entropy
  3. gradients accumulated over --accum micro-batches, clipped, AdamW step
  4. warmup + cosine schedule; periodic eval on fixed validation windows
  5. best checkpoint saved with save_pretrained() so sample_ft.py can load it

The step-0 evaluation is the *base* model's loss on Fed text, before any
training: that is the "before" number.

Sized for a 16 GB Apple Silicon laptop: micro-batch 1 x 1024, gradient
checkpointing on, ~360 tokens/s for the 360M model.

Logs: finetune/out/eval_log.csv, finetune/out/step_log.csv
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "ft"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default=None, help="default: the model data/ft was tokenised with")
ap.add_argument("--out_dir", default="finetune/out")
ap.add_argument("--max_steps", type=int, default=320)
ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimiser step")
ap.add_argument("--seq_len", type=int, default=1024)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--min_lr", type=float, default=3e-6)
ap.add_argument("--warmup_steps", type=int, default=10)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--grad_clip", type=float, default=1.0)
ap.add_argument("--eval_interval", type=int, default=40)
ap.add_argument("--eval_windows", type=int, default=24, help="fixed windows per split per eval")
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--device", default="mps")
args = ap.parse_args()

meta = json.loads((DATA / "meta.json").read_text())
model_name = args.model or meta["model"]
device = torch.device(args.device)
torch.manual_seed(args.seed)
out_dir = HERE / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

train_data = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")


def window(data, start):
    """One window of seq_len tokens. Hugging Face models shift labels internally,
    so x doubles as its own target; there is no separate y as in train.py."""
    x = torch.from_numpy(data[start:start + args.seq_len].astype(np.int64))
    return x[None].to(device), None


def random_window(data):
    return window(data, int(torch.randint(len(data) - args.seq_len - 1, (1,))))


# Fixed, evenly spaced evaluation windows so every eval scores the same text.
eval_starts = {split: np.linspace(0, len(d) - args.seq_len - 1, args.eval_windows, dtype=int)
               for split, d in (("train", train_data), ("val", val_data))}

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
model.config.use_cache = False
model.gradient_checkpointing_enable()
n_params = sum(p.numel() for p in model.parameters())
print(f"{model_name}: {n_params / 1e6:.0f}M params on {device}; {args.accum * args.seq_len:,} tokens/step; "
      f"{args.max_steps} steps = {args.max_steps * args.accum * args.seq_len / 1e6:.2f}M tokens "
      f"({100 * args.max_steps * args.accum * args.seq_len / len(train_data):.0f}% of the training split)", flush=True)

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))


def get_lr(step):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, progress))) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for s in eval_starts[split]:
            x, y = window(data, int(s))
            with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                losses.append(model(input_ids=x, labels=x).loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


step_file = (out_dir / "step_log.csv").open("w", newline="", buffering=1)
step_log = csv.writer(step_file); step_log.writerow(["step", "train_loss", "lr", "s_per_step", "tokens_seen"])
eval_file = (out_dir / "eval_log.csv").open("w", newline="", buffering=1)
eval_log = csv.writer(eval_file); eval_log.writerow(["step", "train_loss", "val_loss", "elapsed_s", "tokens_seen"])

best_val = float("inf")
t_start = time.time()
model.train()
for step in range(args.max_steps + 1):
    if step % args.eval_interval == 0 or step == args.max_steps:
        losses = estimate_loss()
        elapsed = time.time() - t_start
        tokens = step * args.accum * args.seq_len
        print(f"--- eval @ {step}: train {losses['train']:.4f}  val {losses['val']:.4f}  "
              f"({elapsed / 60:.1f} min, {tokens / 1e6:.2f}M tokens)", flush=True)
        eval_log.writerow([step, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{elapsed:.0f}", tokens])
        if losses["val"] < best_val:
            best_val = losses["val"]
            if step > 0:
                model.save_pretrained(out_dir / "model")
                tok.save_pretrained(out_dir / "model")
    if step == args.max_steps:
        break

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    t0 = time.time()
    total = 0.0
    for _ in range(args.accum):
        x, y = random_window(train_data)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss / args.accum
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    dt = time.time() - t0
    tokens = (step + 1) * args.accum * args.seq_len
    print(f"step {step}: loss {total:.4f}  lr {lr:.2e}  {dt:.1f} s/step", flush=True)
    step_log.writerow([step, f"{total:.4f}", f"{lr:.2e}", f"{dt:.1f}", tokens])

print(f"done. best val loss {best_val:.4f}; model at {out_dir / 'model'}", flush=True)
