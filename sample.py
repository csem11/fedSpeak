#!/usr/bin/env python3
"""
sample.py - generate text from a trained FedSpeak checkpoint.

The model only ever learned "given these characters, which character comes
next?". Generation is that question asked repeatedly: feed the prompt, sample
one character from the predicted distribution, append it, repeat.

  --temperature  <1 sharpens the distribution (safer, more repetitive),
                 >1 flattens it (more surprising, more nonsense)
  --top_k        only sample among the k most likely next characters

Usage:
  python sample.py
  python sample.py --prompt "The Committee decided to" --temperature 0.7
"""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ckpt", default="out/ckpt.pt")
ap.add_argument("--prompt", default="\n", help="starting text; default is a newline (start of a document)")
ap.add_argument("--num_samples", type=int, default=3)
ap.add_argument("--max_new_tokens", type=int, default=600, help="characters to generate per sample")
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=40)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--device", default=None)
args = ap.parse_args()

if args.device is None:
    args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
torch.manual_seed(args.seed)

ckpt = torch.load(HERE / args.ckpt, map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.to(device).eval()
stoi, itos = ckpt["meta"]["stoi"], ckpt["meta"]["itos"]
print(f"loaded {args.ckpt}: iter {ckpt['iter_num']}, val loss {ckpt['best_val_loss']:.4f}, "
      f"{model.get_num_params() / 1e6:.2f}M params\n")

unknown = sorted(set(args.prompt) - set(stoi))
if unknown:
    print(f"note: dropping characters not in the vocabulary: {unknown!r}")
prompt_ids = [stoi[c] for c in args.prompt if c in stoi] or [stoi["\n"]]
x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    for k in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"=============== sample {k + 1} ===============")
        print("".join(itos[int(i)] for i in y[0]))
        print()
