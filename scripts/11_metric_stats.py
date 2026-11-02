"""Uncertainty for the metric audit: bootstrap intervals, a sign test against chance, ties and score
margins, overall and per error type. Uses the scores cached by 03_metric_audit.py.

    python scripts/11_metric_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = {"BLEU": "BLEU", "TER (neg)": "TER_neg", "BERTScore": "BERTScore", "chrF": "chrF", "chrF++": "chrF++",
           "COMET-22": "COMET-22", "LaBSE-ref": "LaBSE-ref", "LaBSE-QE": "LaBSE-QE"}
N_BOOT = 2000


def block(good: np.ndarray, bad: np.ndarray, rng: np.random.Generator) -> dict:
    pref = (good > bad) + 0.5 * (good == bad)
    boot = pref[rng.integers(0, len(pref), (N_BOOT, len(pref)))].mean(axis=1)
    wins, losses = int((good > bad).sum()), int((good < bad).sum())
    margin = good - bad
    return {"safe_preferred": round(float(pref.mean()), 4),
            "ci95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "ties": round(float((good == bad).mean()), 4),
            "p_vs_chance": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else None,
            "median_margin": round(float(np.median(margin)), 4),
            "iqr_margin": [round(float(x), 4) for x in np.percentile(margin, [25, 75])], "n": int(len(pref))}


def main():
    df = pd.read_json(RES / "audit_set.jsonl", lines=True)
    rng = np.random.default_rng(7)
    out = {}
    for name, fname in METRICS.items():
        cache = json.loads((RES / "cache" / f"{fname}.json").read_text(encoding="utf-8"))
        get = lambda col: np.array([cache["\t".join((r.src_en, getattr(r, col), r.ref_ar))] for r in df.itertuples()])  # noqa: E731
        good, bad, badm = get("good_ar"), get("bad_ar"), get("bad_matched_ar")
        out[name] = {"reworded": block(good, bad, rng), "same_wording": block(good, badm, rng), "per_type": {}}
        for t, idx in df.groupby("error_type").groups.items():
            idx = np.asarray(idx)
            out[name]["per_type"][t] = {"reworded": block(good[idx], bad[idx], rng),
                                        "same_wording": block(good[idx], badm[idx], rng)}
    (RES / "metric_stats.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, v in out.items():
        a, b = v["reworded"], v["same_wording"]
        print(f"{name:10s} reworded {a['safe_preferred']:.3f} {a['ci95']} ties {a['ties']:.3f} p={a['p_vs_chance']:.1e} "
              f"| same wording {b['safe_preferred']:.3f} {b['ci95']} ties {b['ties']:.3f}")


if __name__ == "__main__":
    main()
