"""Plots for one training run from its train.jsonl.

    python plot_run.py log/<run_name>            # reads <dir>/train.jsonl, writes loss.png + diagnostics.png next to it
    python plot_run.py train.jsonl --out plots   # explicit file and output dir
"""
import os
import json
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 1.6,
})


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    return df[df.split == "train"].set_index("step"), df[df.split == "val"].set_index("step")


def plot_loss(train, val, out):
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    y = train["ce"] if "ce" in train else train["loss"]          # ce = cross-entropy only (MoE runs add aux terms into loss)
    ax.plot(y.index, y, color=BLUE, linewidth=0.8, alpha=0.45, label="train (per step)")
    ax.plot(y.index, y.rolling(100, min_periods=1).mean(), color=BLUE, label="train (100-step mean)")
    vy = val["ce"] if "ce" in val else val["loss"]
    ax.plot(vy.index, vy, color=ORANGE, marker="o", markersize=3, label="val")
    ax.annotate(f"{vy.iloc[-1]:.3f}", (vy.index[-1], vy.iloc[-1]), textcoords="offset points", xytext=(6, 4), color=INK2, fontsize=9)
    ax.annotate(f"{y.rolling(100).mean().iloc[-1]:.3f}", (y.index[-1], y.rolling(100).mean().iloc[-1]),
                textcoords="offset points", xytext=(6, -10), color=INK2, fontsize=9)
    ax.set_ylim(min(vy.min(), y.rolling(100, min_periods=1).mean().min()) - 0.15, min(y.max(), 6.0))
    ax.set_xlabel("step"); ax.set_ylabel("loss (cross-entropy)"); ax.set_title("Loss")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def plot_diagnostics(train, out):
    panels = [("lr", "learning rate", None), ("grad_norm", "grad norm (after clipping)", "log"),
              ("tok_per_sec", "tokens / sec", None), ("dt", "step time, s", "log")]
    titles = {"lb": "load-balancing loss (sum over layers)", "z": "router z-loss (sum over layers)", "mem_gb": "peak GPU memory, GB"}
    panels += [(c, titles[c], "log" if c == "z" else None) for c in ("lb", "z", "mem_gb") if c in train]
    n = len(panels); cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.2 * rows), dpi=150)
    for ax, (col, title, scale) in zip(axes.flat, panels):
        s = train[col]
        ax.plot(s.index, s, color=BLUE, linewidth=0.8, alpha=0.5)
        ax.plot(s.index, s.rolling(100, min_periods=1).median(), color=BLUE)
        if scale: ax.set_yscale(scale)
        if col == "tok_per_sec": ax.set_ylim(0, s.quantile(0.99) * 1.1)        # first step (compile) would flatten the rest
        if col == "dt": ax.set_ylim(s.quantile(0.01) * 0.8, s.quantile(0.99) * 3)
        ax.set_title(title); ax.set_xlabel("step")
    for ax in list(axes.flat)[n:]: ax.axis("off")
    fig.suptitle("Diagnostics (thin: per step, bold: 100-step median)", color=INK2, fontsize=10)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="run directory containing train.jsonl, or the jsonl file itself")
    p.add_argument("--out", default=None, help="output directory (default: next to the jsonl)")
    a = p.parse_args()
    jsonl = a.path if a.path.endswith(".jsonl") else os.path.join(a.path, "train.jsonl")
    out = a.out or os.path.dirname(os.path.abspath(jsonl))
    os.makedirs(out, exist_ok=True)
    train, val = load(jsonl)
    plot_loss(train, val, os.path.join(out, "loss.png"))
    plot_diagnostics(train, os.path.join(out, "diagnostics.png"))
    ce = "ce" if "ce" in val else "loss"
    print(f"steps {len(train)} | final val {ce} {val[ce].iloc[-1]:.4f} | train {ce} (100-step mean) {train[ce].tail(100).mean():.4f}")
    print(f"written: {out}/loss.png, {out}/diagnostics.png")
