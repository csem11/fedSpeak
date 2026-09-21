#!/usr/bin/env python3
"""
sample_ft.py - generate from the fine-tuned model (or the untouched base model, for comparison).

The fine-tuning data gave every document a header, so the most useful prompt
is a header:   python finetune/sample_ft.py --prompt "[FOMC statement | 2026-10-28]"
Pass --model HuggingFaceTB/SmolLM2-360M to see what the base model does with the same prompt.
"""

import argparse
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
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", default="finetune/out/model", help="local directory or Hub id")
ap.add_argument("--prompt", default="[FOMC statement | 2026-10-28]\n")
ap.add_argument("--num_samples", type=int, default=1)
ap.add_argument("--max_new_tokens", type=int, default=250)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--repetition_penalty", type=float, default=1.1)
ap.add_argument("--seed", type=int, default=1337)
ap.add_argument("--device", default="mps")
args = ap.parse_args()

path = HERE / args.model
name = str(path) if path.exists() else args.model
tok = AutoTokenizer.from_pretrained(name)
model = load_model(name, torch.bfloat16, args.device)
torch.manual_seed(args.seed)
prompt = args.prompt if args.prompt.endswith("\n") or not args.prompt.startswith("[") else args.prompt + "\n"
ids = tok(prompt, return_tensors="pt").input_ids.to(args.device)
print(f"model: {name}\nprompt: {prompt!r}\n")
with torch.no_grad():
    for k in range(args.num_samples):
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                             pad_token_id=tok.eos_token_id)
        print(f"=============== sample {k + 1} ===============")
        print(tok.decode(out[0], skip_special_tokens=True))
        print()
