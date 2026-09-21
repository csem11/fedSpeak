#!/usr/bin/env python3
"""
build.py - assemble docs/index.html from site/template.html and results/.

The page is a single static file: every chart, sample and table is read from
the CSV/JSON/text files under results/ at build time and embedded as one JSON
object, so the page needs no server and stays in sync with the experiments.

  python3 site/build.py     ->  docs/index.html
"""

import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
LN2 = math.log(2)


def read_csv(path, numeric=True):
    rows = []
    for r in csv.DictReader(path.open()):
        if numeric:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
        rows.append(r)
    return rows


def parse_blocks(path, marker):
    """Split a samples file on '====' separators; return [(header line, body)]."""
    text = path.read_text()
    parts = re.split(r"\n=+\n", text)
    out = []
    for i, part in enumerate(parts):
        m = re.match(rf"\s*{marker}(.*)", part.strip())
        if m and i + 1 < len(parts):
            out.append((m.group(1).strip(), parts[i + 1].strip()))
    return out


data = {}

# ---- corpus
data["corpus"] = {
    "per_year": read_csv(RES / "corpus" / "docs_per_year.csv"),
    "shares": read_csv(RES / "corpus" / "source_shares.csv"),
    "vocab": [{"id": int(r["id"]), "char": eval(r["char"]), "count": int(r["count"])} for r in read_csv(RES / "corpus" / "vocab.csv", numeric=False)],
}
ex = (RES / "corpus" / "cleaning_example.md").read_text()
raw = re.search(r"```html\n(.*?)\n```", ex, re.S).group(1)
clean = re.search(r"```text\n(.*?)\n```", ex, re.S).group(1)
data["corpus"]["cleaning_example"] = {"raw": raw[:1400], "clean": clean[:900]}

# ---- main run + stages
data["main_run"] = {"eval": read_csv(RES / "eval_log.csv"), "step": read_csv(RES / "step_log.csv")}
stages = []
for hdr, body in parse_blocks(RES / "samples_by_iteration.txt", "ITERATION"):
    it, vl = re.match(r"(\d+)\s+val loss\s+([\d.]+)", hdr).groups()
    stages.append({"iter": int(it), "val_loss": float(vl), "text": body})
data["stages"] = stages

# ---- ablations
AXES = {
    "context": ("Context length", [("ctx0016", "16"), ("ctx0064", "64"), ("ctx0256", "256"), ("ctx1024", "1024")]),
    "size": ("Model size", [("size0.4M", "0.40M"), ("ctx0256", "3.17M"), ("size10.6M", "10.65M")]),
    "lr": ("Peak learning rate", [("lr3e-4", "3e-4"), ("ctx0256", "1e-3"), ("lr3e-3", "3e-3")]),
}
# The full chunk-size sweep: one point per context length, for the U-curve.
# The over-time chart keeps four representative sizes so the categorical
# palette stays within its validated range.
CONTEXT_RUNS = [("ctx0016", 16), ("ctx0032", 32), ("ctx0064", 64), ("ctx0128", 128),
                ("ctx0256", 256), ("ctx0512", 512), ("ctx1024", 1024)]
curve = []
for run, ctx in CONTEXT_RUNS:
    p = RES / "experiments" / run / "eval_log.csv"
    if p.exists():
        ev = read_csv(p)
        curve.append({"run": run, "ctx": ctx, "final": ev[-1]["val_loss"],
                      "minutes": round(ev[-1]["elapsed_s"] / 60, 1), "seqs_per_step": 16384 // ctx})
fair = RES / "common_context_eval.csv"
if fair.exists():
    by_run = {r["run"]: r["common_grid_loss"] for r in read_csv(fair)}
    for c in curve:
        c["fair"] = by_run.get(c["run"])
data["context_curve"] = curve
lr_full = RES / "experiments" / "main_lr3e-3" / "eval_log.csv"
data["lr_transfer"] = {"lr1e-3": 0.698, "lr3e-3": read_csv(lr_full)[-1]["val_loss"]} if lr_full.exists() else None

data["ablations"] = {}
for axis, (title, members) in AXES.items():
    runs = []
    for run, label in members:
        ev = read_csv(RES / "experiments" / run / "eval_log.csv")
        runs.append({"run": run, "label": label, "baseline": run == "ctx0256",
                     "eval": [{"iter": r["iter"], "val_loss": r["val_loss"]} for r in ev],
                     "final": ev[-1]["val_loss"], "minutes": round(ev[-1]["elapsed_s"] / 60, 1)})
    data["ablations"][axis] = {"title": title, "runs": runs}

# ---- per-source, all three models, bits/char
by_source = {"char": {}, "base": {}, "finetuned": {}, "qwen_base": {}, "qwen_finetuned": {}}
for r in read_csv(RES / "loss_by_source.csv"):
    by_source["char"][r["source"]] = round(r["loss_nats_per_char"] / LN2, 3)
for f in (RES / "finetune" / "loss_by_source.csv", RES / "finetune_qwen" / "loss_by_source.csv"):
    if f.exists():
        for r in read_csv(f):
            by_source[r["model"]][r["source"]] = r["bits_per_char"]
data["by_source"] = by_source

# ---- temperature sweep (char model)
data["temperature"] = [{"t": float(h), "text": b} for h, b in parse_blocks(RES / "temperature_sweep.txt", "TEMPERATURE")]

# ---- fine-tune
CPT = 5.39
ft = read_csv(RES / "finetune" / "eval_log.csv")
# Qwen3's tokenizer is a little more efficient on this corpus: 5.55 characters
# per token against SmolLM2's 5.39 (36.8M characters / 6.64M vs 6.84M tokens).
CPT_QWEN = 5.55
qf = RES / "finetune_qwen" / "eval_log.csv"
data["finetune_qwen"] = [{"step": r["step"], "tokens": r["tokens_seen"], "val": r["val_loss"],
                          "bpc": round(r["val_loss"] / LN2 / CPT_QWEN, 3)} for r in read_csv(qf)] if qf.exists() else None
data["finetune"] = {"chars_per_token": CPT, "char_model_bpc": round(0.698 / LN2, 3),
                    "eval": [{"step": r["step"], "tokens": r["tokens_seen"], "val": r["val_loss"], "train": r["train_loss"],
                              "bpc": round(r["val_loss"] / LN2 / CPT, 3)} for r in ft]}

# ---- playground
pg = RES / "playground.json"
data["playground"] = json.loads(pg.read_text()) if pg.exists() else None

# ---- scale table
data["scale"] = read_csv(RES / "scale_table.csv", numeric=False)

# ---- annotated sample: spans of the phase-1 minutes sample with tags
data["annotated"] = [
    ("Minutes of the Federal Open Market Committee\n", "form", "The prompt. Every real minutes document since 1994 opens this way."),
    ("\nJanuary 1, 2100\n", "fabricated", "Right shape, impossible value. The corpus has years 1994-2026; the model learned 'month, day, comma, four digits', not what a year is."),
    ("\nA joint meeting of the Federal Open Market Committee and the Board of Governors of the Federal Reserve System ", "form", "Verbatim template from real minutes."),
    ("in April and ", "drift", "'was held in the offices ... on' collapsed into 'in April and'."),
    ("May 38, 2002, at 9:00 p.m.", "fabricated", "No month has 38 days; 240 meetings in the corpus start at 9:00 a.m. and none in the evening."),
    ("\n\nPresent:\n\nMr. Greenspan, Chairman\n", "form", "Roster block in the pre-2011 format. Greenspan is on 97 of the 261 rosters, always as Chairman: memorised co-occurrence, accidentally true."),
    ("Mr. ki meeting on Secretary\n", "bug", "A surname that failed to form: a lowercase 'k' after 'Mr. ' and no way back into a name."),
    ("Mr. Gillum, on August 18, 1997, until Procedure\n", "fabricated", "Gillum is 'Assistant Secretary' on all 74 of his real roster lines."),
    ("Mr. Gramlich, Manager, System Open Market Account\n", "fabricated", "A real title assigned to the wrong person; Gramlich is always listed as a bare voting member."),
    ("Mr. General CounselVrey Deputy Secretary\n", "bug", "A title where a surname should be, then a fused token."),
    ("Ms. Oliner, Deputy Secretary\nMr. Boehne, General Counsel\n", "fabricated", "Oliner is a Senior Adviser; Boehne a voting member. Neither held these roles."),
    ("Ms. Bridge, Deputy Secretary\n", "fabricated", "'Bridge' appears on no roster in the corpus: an invented surname that looks like one."),
    ("Mr. Coyne, Assistant Secretary\nMr. Alvarez, General Counsel\n", "form", "Both correct: Coyne is Assistant Secretary on 34 lines, Alvarez General Counsel on 43. Note this is the second General Counsel in one roster."),
]

out = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
template = (HERE / "site" / "template.html").read_text()
assert "/*__DATA__*/" in template
html = template.replace("/*__DATA__*/", out, 1)
# Two outputs from one template: a complete standalone page for local use and
# GitHub Pages, and the bare fragment the Artifact publisher wraps itself.
skeleton = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + html.split("</style>", 1)[0] + "</style>\n</head>\n<body>\n" + html.split("</style>", 1)[1] + "\n</body>\n</html>\n")
(HERE / "docs").mkdir(exist_ok=True)
(HERE / "docs" / "index.html").write_text(skeleton)
(HERE / "site" / "artifact.html").write_text(html)
print(f"docs/index.html: {len(skeleton) / 1e6:.2f} MB; playground {'included' if data['playground'] else 'MISSING'}")
