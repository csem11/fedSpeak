#!/usr/bin/env python3
"""
eval_ft_by_source.py - score a pretrained or fine-tuned model on each source's
validation documents, in nats per token and bits per character.

Bits per character is the unit that puts a BPE model and the from-scratch
character model on the same axis:  bits/char = (nats/token / ln 2) / (chars/token).

  python finetune/eval_ft_by_source.py                                  # fine-tuned
  python finetune/eval_ft_by_source.py --model HuggingFaceTB/SmolLM2-360M --tag base
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent


def load_model(name, dtype, device):
    """Load a full model, or a LoRA adapter directory (which records its base model)."""
    if (Path(name) / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        return AutoPeftModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
DATA = HERE / "data"
LABELS = {"statements": "FOMC statement", "minutes": "FOMC minutes", "beigebook": "Beige Book"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="finetune/out/model")
ap.add_argument("--tag", default="finetuned")
ap.add_argument("--seq_len", type=int, default=1024)
ap.add_argument("--device", default="mps")
ap.add_argument("--data_dir", default="data/ft", help="whose meta.json lists the validation documents")
ap.add_argument("--out", default="results/finetune/loss_by_source.csv", help="CSV to append rows to")
args = ap.parse_args()

meta = json.loads((HERE / args.data_dir / "meta.json").read_text())
path = HERE / args.model
name = str(path) if path.exists() else args.model
tok = AutoTokenizer.from_pretrained(name)
# fp32 weights with bfloat16 autocast in score(): the same numerics as the
# SmolLM2 evaluation, so the two model families are scored identically.
model = load_model(name, torch.float32, args.device)


@torch.no_grad()
def score(text: str):
    """Sum of token losses and counts over consecutive windows of up to seq_len
    tokens. The last window may be short: statements are ~500 tokens, shorter
    than one full window, and must still count."""
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    nats, count = 0.0, 0
    for start in range(0, len(ids), args.seq_len):
        row = ids[start:start + args.seq_len]
        if len(row) < 2:
            continue
        x = row[None].to(args.device)
        with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            nats += model(input_ids=x, labels=x).loss.item() * (len(row) - 1)
        count += len(row) - 1
    return nats, count


rows = []
for src in LABELS:
    docs = [d for d in meta["val_docs"] if d.startswith(src + "/")]
    nats = toks = chars = 0
    for d in docs:
        p = DATA / "clean" / (d + ".txt")
        date = p.stem
        text = f"[{LABELS[src]} | {date[:4]}-{date[4:6]}-{date[6:]}]\n" + p.read_text()
        a, b = score(text)
        nats += a; toks += b; chars += len(text)
    npt = nats / toks
    cpt = chars / len(tok("".join((DATA / "clean" / (d + ".txt")).read_text() for d in docs))["input_ids"])
    bpc = npt / math.log(2) / cpt
    rows.append([args.tag, src, len(docs), toks, f"{npt:.4f}", f"{cpt:.2f}", f"{bpc:.3f}"])
    print(f"{args.tag:<10} {src:<11} {len(docs):3d} docs  {toks:>9,} tokens  {npt:.4f} nats/token  {cpt:.2f} chars/token  = {bpc:.3f} bits/char", flush=True)

out = HERE / args.out
out.parent.mkdir(parents=True, exist_ok=True)
new = not out.exists()
with out.open("a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["model", "source", "val_docs", "tokens_scored", "nats_per_token", "chars_per_token", "bits_per_char"])
    w.writerows(rows)
print("appended to", out.relative_to(HERE))
