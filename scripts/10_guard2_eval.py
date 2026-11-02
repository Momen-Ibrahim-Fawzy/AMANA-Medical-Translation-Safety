"""AMANA-Guard 2 against the guard.py baseline. A translation "passes" only with no flag and no
review item.

    python scripts/10_guard2_eval.py             # development data
    python scripts/10_guard2_eval.py --holdout   # frozen hard cases; report once, do not tune on it
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana import guard as G, guard2 as G2  # noqa: E402

HOLDOUT = ROOT / "data" / "hard_cases_holdout.jsonl"
HOLDOUT_SHA256 = "db6ebbab36d5a48ec437171a20e458f45467925e1a1012def551c503502de80e"
CHECKERS = {"baseline": G.check, "guard2": G2.check}


def rates(df: pd.DataFrame, chk) -> dict:
    passed = lambda col: df.apply(lambda r: chk(r.src_en, r[col]).ok, axis=1)  # noqa: E731
    bad = ~passed("bad_ar")
    safe = ~pd.concat([passed("ref_ar"), passed("good_ar")])
    return {"recall": round(float(bad.mean()), 4), "false_alarm": round(float(safe.mean()), 4)}


def case_scores(cases: list[tuple[str, bool, str, str]], chk) -> dict:
    """cases: (name, should_pass, src, hyp)"""
    wrong = {"missed_errors": [], "false_alarms": []}
    for name, should_pass, s, h in cases:
        if chk(s, h).ok != should_pass:
            wrong["false_alarms" if should_pass else "missed_errors"].append(name)
    n_err = sum(not c[1] for c in cases)
    return {"errors_caught": [n_err - len(wrong["missed_errors"]), n_err],
            "safe_accepted": [len(cases) - n_err - len(wrong["false_alarms"]), len(cases) - n_err], **wrong}


def garbled(chk) -> int:
    """Source names drug A, translation has drug B with one interior letter dropped. Counts pairs
    that pass with nothing raised about the drug."""
    table = G._ar_drug_table()
    name_of = {k: n for n, k in table.items()}
    accepted = 0
    for a, b in itertools.permutations(sorted(set(table.values())), 2):
        n = name_of[b]
        for g in (n[:i] + n[i + 1:] for i in range(2, len(n) - 1)):
            rep = chk(f"Take {a}.", f"خذ {g}.")
            if " " not in g and not any(f.kind == "drug" for f in rep.flags + getattr(rep, "review", [])):
                accepted += 1
                break
    return accepted


def mt_rates(chk) -> dict:
    rows = [json.loads(l) for l in (ROOT / "results" / "mt_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    out = {}
    for name in sorted({r["system"] for r in rows}):
        reps = [chk(r["src_en"], r["cands"][0]) for r in rows if r["system"] == name]
        flags = [f.kind for x in reps for f in x.flags]
        review = [f.kind for x in reps for f in getattr(x, "review", [])]
        out[name] = {"not_passed": round(sum(not x.ok for x in reps) / len(reps), 4),
                     "flagged": round(sum(bool(x.flags) for x in reps) / len(reps), 4),
                     "flag_types": dict(Counter(flags)), "review_types": dict(Counter(review))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    if args.holdout:
        digest = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
        assert digest == HOLDOUT_SHA256, "hold-out file changed since it was frozen"
        rows = [json.loads(l) for l in HOLDOUT.read_text(encoding="utf-8").splitlines()]
        cases = [(r["id"], r["should_pass"], r["src_en"], r["hyp_ar"]) for r in rows]
        out = {n: case_scores(cases, c) for n, c in CHECKERS.items()}
        (ROOT / "results" / "guard2_holdout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=1))
        return
    spec = importlib.util.spec_from_file_location("hard_cases", ROOT / "scripts" / "09_hard_cases.py")
    hc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hc)
    dev = [(n, not bad, s, h) for n, bad, s, h in hc.CASES]
    bench = pd.read_json(ROOT / "data" / "processed" / "amana_bench.jsonl", lines=True)
    tico = pd.read_json(ROOT / "data" / "processed" / "amana_tico.jsonl", lines=True)
    out = {}
    for name, chk in CHECKERS.items():
        out[name] = {"bench_test": rates(bench[bench.split == "test"], chk), "tico": rates(tico, chk),
                     "dev_hard_cases": case_scores(dev, chk), "garbled_wrong_drug_accepted": garbled(chk),
                     "mt_1best": mt_rates(chk)}
    (ROOT / "results" / "guard2_dev.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
