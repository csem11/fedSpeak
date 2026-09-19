#!/usr/bin/env python3
"""
plot_results.py - figures for the write-up, from the CSV logs.

  results/figures/loss_curve.png         main 10.65M run: train and val loss
  results/figures/ablation_context.png   val loss vs iteration for each context length
  results/figures/ablation_size.png      ... for each model size
  results/figures/ablation_lr.png        ... for each learning rate
  results/figures/ablation_summary.png   final val loss of every run, one panel per axis

Reads results/eval_log.csv, results/step_log.csv and results/experiments/<run>/eval_log.csv.
Runs that have not finished are skipped, so this can be re-run at any time.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

# Palette: validated categorical slots (see dataviz reference); ink for all text.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def style_axes(ax, ylabel="validation loss (nats / char)"):
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


def spread(values, min_gap):
    """Nudge label positions apart so end labels never overlap; keeps their order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = []
    for i in order:
        y = values[i]
        if placed and y - placed[-1][1] < min_gap:
            y = placed[-1][1] + min_gap
        placed.append((i, y))
    out = [0.0] * len(values)
    for i, y in placed:
        out[i] = y
    return out


def label_line_ends(ax, ends, min_gap):
    """ends: list of (x, y, text, color). Marker at the true point, label at a spread-out y."""
    ys = spread([e[1] for e in ends], min_gap)
    for (x, y, text, color), ly in zip(ends, ys):
        ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5)
        ax.annotate(text, (x, ly), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))


# ---------------------------------------------------------------- main run
def plot_main():
    ev = read_csv(RES / "eval_log.csv")
    st = read_csv(RES / "step_log.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot([r["iter"] for r in st], [r["train_loss"] for r in st], color=SERIES[0], lw=0.8, alpha=0.25)
    ax.plot([r["iter"] for r in ev], [r["train_loss"] for r in ev], color=SERIES[0], lw=2, label="train")
    ax.plot([r["iter"] for r in ev], [r["val_loss"] for r in ev], color=SERIES[1], lw=2, label="validation")
    last = ev[-1]
    label_line_ends(ax, [(last["iter"], last["train_loss"], f"train {last['train_loss']:.3f}", SERIES[0]),
                         (last["iter"], last["val_loss"], f"validation {last['val_loss']:.3f}", SERIES[1])], min_gap=0.22)
    ax.axhline(4.454, color=INK2, lw=1, ls=(0, (4, 4)))
    ax.text(last["iter"] * 1.15, 4.454, "uniform guess = ln(86)  ", va="bottom", ha="right", fontsize=8.5, color=INK2)
    ax.set_title("Main run: 10.65M params, one pass over 33M characters")
    style_axes(ax, "loss (nats / char)")
    ax.set_xlim(0, last["iter"] * 1.15)
    ax.set_ylim(0, 5)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(FIG / "loss_curve.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- ablations
AXES = {
    "context": ("Context length (characters the model can see)",
                [("ctx0016", "16"), ("ctx0064", "64"), ("ctx0256", "256"), ("ctx1024", "1024")]),
    "size":    ("Model size (parameters)",
                [("size0.4M", "0.40M"), ("ctx0256", "3.17M"), ("size10.6M", "10.65M")]),
    "lr":      ("Peak learning rate",
                [("lr3e-4", "3e-4"), ("ctx0256", "1e-3"), ("lr3e-3", "3e-3")]),
}


def load_runs(axis):
    title, members = AXES[axis]
    out = []
    for run, label in members:
        p = RES / "experiments" / run / "eval_log.csv"
        if p.exists():
            out.append((label, read_csv(p)))
    return title, out


def plot_axis(axis):
    title, runs = load_runs(axis)
    if len(runs) < 2:
        print(f"skip {axis}: only {len(runs)} run(s) finished")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ends = []
    for (label, ev), color in zip(runs, SERIES):
        xs, ys = [r["iter"] for r in ev], [r["val_loss"] for r in ev]
        ax.plot(xs, ys, color=color, lw=2, label=label)
        ends.append((xs[-1], ys[-1], f"{label}  {ys[-1]:.3f}", color))
    label_line_ends(ax, ends, min_gap=0.07)
    ax.set_title(title)
    style_axes(ax)
    xmax = max(r[1][-1]["iter"] for r in runs)
    ax.set_xlim(0, xmax * 1.2)
    ax.set_ylim(0.6, 2.0)
    ax.legend(frameon=False, loc="upper right", title=None)
    fig.savefig(FIG / f"ablation_{axis}.png", dpi=160)
    plt.close(fig)


def plot_context_curve():
    """The chunk-size sweep as a curve: final loss against context length."""
    runs = [("ctx0016", 16), ("ctx0032", 32), ("ctx0064", 64), ("ctx0128", 128),
            ("ctx0256", 256), ("ctx0512", 512), ("ctx1024", 1024)]
    pts = []
    for run, ctx in runs:
        p = RES / "experiments" / run / "eval_log.csv"
        if p.exists():
            pts.append((ctx, read_csv(p)[-1]["val_loss"]))
    if len(pts) < 4:
        return
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    best = ys.index(min(ys))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(xs, ys, color=SERIES[0], lw=2, zorder=2)
    ax.plot(xs, ys, "o", ms=6, mfc=SURFACE, mec=SERIES[0], mew=2, zorder=3)
    ax.plot([xs[best]], [ys[best]], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=1.5, zorder=4)
    for i in (0, best, len(xs) - 1):
        ax.annotate(f"{ys[i]:.3f}", (xs[i], ys[i]), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.set_xlabel("chunk size (characters of context, log scale)")
    ax.set_ylabel("val loss @ 1000 iters (nats / char)")
    ax.set_title("Same characters per step, seven chunk sizes")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.09)
    fig.savefig(FIG / "context_curve.png", dpi=160)
    plt.close(fig)


def plot_summary():
    panels = [(axis, *load_runs(axis)) for axis in AXES]
    panels = [p for p in panels if len(p[2]) >= 2]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels) + 1, 3.6), constrained_layout=True, sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    ymax = max(ev[-1]["val_loss"] for _, _, runs in panels for _, ev in runs) * 1.18
    for ax, (axis, title, runs) in zip(axes, panels):
        labels = [l for l, _ in runs]
        vals = [ev[-1]["val_loss"] for _, ev in runs]
        bars = ax.bar(labels, vals, width=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title.split(" (")[0])
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("val loss after 1000 iterations")
    fig.suptitle("What moves the loss: one knob per panel, everything else held at the baseline", x=0.01, ha="left", fontweight="bold")
    fig.savefig(FIG / "ablation_summary.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- fine-tune
def plot_finetune():
    """Phase 2 curve in bits/char, on the same axis as the phase 1 character model."""
    p = RES / "finetune" / "eval_log.csv"
    if not p.exists():
        return
    import math
    CHARS_PER_TOKEN = 5.39                       # SmolLM2 tokenizer on this corpus
    ev = read_csv(p)
    xs = [r["tokens_seen"] / 1e6 for r in ev]
    bpc = [r["val_loss"] / math.log(2) / CHARS_PER_TOKEN for r in ev]
    char_model = 0.698 / math.log(2)            # phase 1 best val loss, nats/char -> bits/char
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(xs, bpc, color=SERIES[0], lw=2, label="SmolLM2-360M, fine-tuning on Fed text")
    ax.axhline(char_model, color=SERIES[1], lw=2, ls=(0, (4, 3)), label="10.65M char model, trained from scratch")
    label_line_ends(ax, [(xs[-1], bpc[-1], f"fine-tuned  {bpc[-1]:.2f}", SERIES[0]),
                         (xs[-1], char_model, f"char model  {char_model:.2f}", SERIES[1])], min_gap=0.03)
    ax.annotate(f"base model, untouched  {bpc[0]:.2f}", (xs[0], bpc[0]), xytext=(8, 6), textcoords="offset points",
                fontsize=9, color=INK)
    ax.plot([xs[0]], [bpc[0]], "o", ms=5, color=SERIES[0], mec=SURFACE, mew=1.5)
    ax.set_title("Fine-tuning vs from scratch, in bits per character")
    ax.set_xlabel("Fed-text tokens seen during fine-tuning (millions)")
    ax.set_ylabel("validation loss (bits / char)")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.02, xs[-1] * 1.3)
    ax.set_ylim(0.6, 1.1)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "finetune_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_main()
    plot_finetune()
    plot_context_curve()
    for axis in AXES:
        plot_axis(axis)
    plot_summary()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
