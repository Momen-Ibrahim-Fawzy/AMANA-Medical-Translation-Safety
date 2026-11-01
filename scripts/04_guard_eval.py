"""AMANA-Guard as a detector: recall on critical translations and false alarms on safe ones.

Reported on held-out drugs of AMANA-Bench (in domain) and on AMANA-TICO (real professional
translations). False alarms on TICO-19 led to guard fixes, so TICO numbers are development results.

    python scripts/04_guard_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.guard import check  # noqa: E402


def rates(df: pd.DataFrame) -> dict:
    flagged = lambda col: df.apply(lambda r: not check(r.src_en, r[col]).ok, axis=1)  # noqa: E731
    bad = flagged("bad_ar")
    safe = pd.concat([flagged("ref_ar"), flagged("good_ar")])
    return {"recall": float(bad.mean()), "false_alarm": float(safe.mean()), "n_bad": int(len(df)),
            "n_safe": int(len(safe))}


def main():
    bench = pd.read_json(ROOT / "data" / "processed" / "amana_bench.jsonl", lines=True)
    tico = pd.read_json(ROOT / "data" / "processed" / "amana_tico.jsonl", lines=True)
    out = {"bench_test_heldout_drugs": rates(bench[bench.split == "test"]),
           "tico19_real_text": rates(tico)}
    for t, g in tico.groupby("error_type"):
        out[f"tico19_real_text/{t}"] = rates(g)
    (ROOT / "results" / "guard_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"{k:32s} recall={v['recall']:.3f} false_alarm={v['false_alarm']:.3f} n_bad={v['n_bad']}")


if __name__ == "__main__":
    main()
