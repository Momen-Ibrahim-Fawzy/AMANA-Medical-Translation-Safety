"""Figures for the README, built from results/*.json only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
BLUE, GRAY, INK, MUTED, GRID = "#1E6BFF", "#6B7280", "#111827", "#4B5563", "#E5E7EB"
ERROR_ORDER = ["dose", "strength", "unit", "frequency", "duration", "timing", "route", "laterality",
               "negation", "omission", "drug"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})


def is_ours(name: str) -> bool:
    return name.startswith("AMANA")


def fig_blindness(audit: dict):
    names = [k for k in audit if not k.startswith("_")]
    names.sort(key=lambda k: audit[k]["contrastive_minimal"])
    vals = [audit[k]["contrastive_minimal"] for k in names]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(names) + 1.2))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.62, color=[BLUE if is_ours(n) else GRAY for n in names])
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, -0.75, "coin flip", color=MUTED, fontsize=8, va="bottom")
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.9, len(names) - 0.5)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("How often the metric scores the safe translation above the dangerous one")
    ax.set_title("Which metrics notice a clinically critical translation error?", loc="left",
                 color=INK, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "metric_blindness.png")
    plt.close(fig)


def fig_heatmap(audit: dict):
    names = [k for k in audit if not k.startswith("_")]
    names.sort(key=lambda k: audit[k]["contrastive_minimal"])
    types = [t for t in ERROR_ORDER if t in audit[names[0]]["per_type"]]
    m = np.array([[audit[n]["per_type"][t]["minimal"] for t in types] for n in names])
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(names) + 1.6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(types)):
            ax.text(j, i, f"{m[i, j]*100:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(types)), types, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Hardest case: one-word clinical error vs. a safe rewording (% safe preferred)",
                 loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "blindness_by_error_type.png")
    plt.close(fig)


def fig_mt(mt: dict):
    systems = sorted(mt["summary"], key=lambda s: -mt["summary"][s]["1-best"]["chrF"])
    before = [mt["summary"][s]["1-best"]["critical_flag_rate"] for s in systems]
    after = [mt["summary"][s]["Guard+QE rerank"]["critical_flag_rate"] for s in systems]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(systems))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, before, w, color=GRAY, label="Standard output (1-best)")
    ax.bar(x + w / 2 + 0.01, after, w, color=BLUE, label="AMANA safety reranking (same 5 candidates)")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - w / 2, b + 0.01, f"{b:.0%}", ha="center", fontsize=9, color=INK)
        ax.text(xi + w / 2, a + 0.01, f"{a:.0%}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{s}\nchrF {mt['summary'][s]['1-best']['chrF']:.1f}" for s in systems])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylim(0, max(before) * 1.25)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Instructions flagged by AMANA-Guard")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Open MT systems on generated instructions for held-out drugs (EN → AR)", loc="left",
                 color=INK, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "mt_audit.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    audit = json.loads((RES / "metric_audit.json").read_text(encoding="utf-8"))
    fig_blindness(audit)
    fig_heatmap(audit)
    if (RES / "mt_audit.json").exists():
        fig_mt(json.loads((RES / "mt_audit.json").read_text(encoding="utf-8")))
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
