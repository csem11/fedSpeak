#!/usr/bin/env python3
"""
generate_playground.py - precompute the prompt playground for the portfolio page.

Runs the same prompts through all three models and stores the outputs, so the
page can show them instantly without a model in the browser:

  char       phase 1, 10.65M-parameter character model  (out/ckpt.pt)
  base       SmolLM2-360M, untouched
  finetuned  SmolLM2-360M after finetune/finetune.py    (finetune/out/model)

Writes results/playground.json. Everything in it is generated text.
"""

import json
import pickle
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from model import GPT, GPTConfig  # noqa: E402

PROMPTS = [
    {"id": "decided", "label": "Statement opener", "text": "The Committee decided to"},
    {"id": "minutes", "label": "Minutes title", "text": "Minutes of the Federal Open Market Committee"},
    {"id": "participants", "label": "Minutes discussion", "text": "In their discussion of the economic situation and the outlook, meeting participants"},
    {"id": "consumer", "label": "Beige Book heading", "text": "Consumer Spending"},
    {"id": "boston", "label": "District heading", "text": "Federal Reserve Bank of Boston"},
    {"id": "hdr_statement", "label": "Header: statement", "text": "[FOMC statement | 2026-10-28]"},
    {"id": "hdr_beige", "label": "Header: Beige Book", "text": "[Beige Book | 2026-10-15]"},
    {"id": "question", "label": "A real question", "text": "What will the Fed do at its next meeting?"},
]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 1337
# --only char,base,... regenerates just those models and merges them into the
# existing results/playground.json, so adding a model doesn't re-run the others.
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else {"char", "base", "finetuned"}
OUT_PATH = HERE / "results" / "playground.json"
out = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {"prompts": PROMPTS, "models": {}, "settings": {}}
out["prompts"] = PROMPTS

# ---- phase 1 character model
if "char" in ONLY:
  ckpt = torch.load(HERE / "out" / "ckpt.pt", map_location=DEVICE, weights_only=False)
  gpt = GPT(GPTConfig(**ckpt["model_args"])).to(DEVICE).eval()
  gpt.load_state_dict(ckpt["model"])
  stoi, itos = ckpt["meta"]["stoi"], ckpt["meta"]["itos"]
  out["settings"]["char"] = {"temperature": 0.8, "top_k": 40, "max_new_tokens": 600}
  out["models"]["char"] = {}
  for p in PROMPTS:
      torch.manual_seed(SEED)
      ids = [stoi[c] for c in p["text"] if c in stoi]
      x = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None]
      with torch.no_grad():
          y = gpt.generate(x, 600, temperature=0.8, top_k=40)
      out["models"]["char"][p["id"]] = "".join(itos[int(i)] for i in y[0])
      print("char     ", p["id"], flush=True)
  del gpt

# ---- pretrained models, full or LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
HF_MODELS = {
    "base": "HuggingFaceTB/SmolLM2-360M",
    "finetuned": str(HERE / "finetune" / "out" / "model"),
    "qwen_base": "Qwen/Qwen3-1.7B-Base",
    "qwen_finetuned": str(HERE / "finetune" / "out_qwen3-1.7b" / "model"),
}
for tag, name in HF_MODELS.items():
    if tag not in ONLY:
        continue
    tok = AutoTokenizer.from_pretrained(name)
    if (Path(name) / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM
        m = AutoPeftModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEVICE).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEVICE).eval()
    out["settings"][tag] = {"temperature": 0.8, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.1, "max_new_tokens": 200}
    out["models"][tag] = {}
    for p in PROMPTS:
        torch.manual_seed(SEED)
        text = p["text"] + ("\n" if p["text"].startswith("[") else "")
        ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            y = m.generate(ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_k=50, top_p=0.95,
                           repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
        out["models"][tag][p["id"]] = tok.decode(y[0], skip_special_tokens=True)
        print(tag.ljust(9), p["id"], flush=True)
    del m

OUT_PATH.write_text(json.dumps(out, indent=1))
print("wrote results/playground.json")
