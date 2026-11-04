"""Test the transliteration comparison on Wikidata's English and Arabic drug names, which no one on this
project wrote. One name in five, chosen by hash, is sealed: the threshold is chosen on the rest.

    python scripts/15_drug_names_eval.py          # development names: threshold sweep
    python scripts/15_drug_names_eval.py --test   # sealed names, at the fixed threshold

Each drug is tested with its own Arabic label (should match), a random other drug's label (should not),
and the label of the drug whose English name is closest to it (should not, and is much harder). ISMP
look-alike pairs from the lexicon are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana import drugnames as D, lexicon as L  # noqa: E402
from amana.arabic import normalize  # noqa: E402

THRESHOLDS = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def load() -> list[tuple[str, str]]:
    with D.TABLE.open(encoding="utf-8", newline="") as f:
        return [(r["en"], normalize(r["ar"])) for r in csv.DictReader(f, delimiter="\t") if " " not in r["ar"]]


def sealed(en: str) -> bool:
    return int(hashlib.sha256(en.encode()).hexdigest(), 16) % 5 == 0


def nearest(en: str, names: list[str]) -> str:
    return max((n for n in names if n != en), key=lambda n: SequenceMatcher(None, en, n).ratio())


def rates(rows: list[tuple[str, str]], t: float) -> dict:
    labels = dict(rows)
    names = list(labels)
    rng = random.Random(13)
    own = random_other = hard = 0
    hard_examples = []
    for en, ar in rows:
        own += D.similarity(en, ar) >= t
        other = rng.choice([n for n in names if n != en])
        random_other += D.similarity(en, labels[other]) >= t
        near = NEAREST[en]
        if D.similarity(en, labels[near]) >= t:
            hard += 1
            hard_examples.append((en, near))
    n = len(rows)
    return {"threshold": t, "n": n, "own_label_accepted": round(own / n, 4), "random_drug_accepted": round(random_other / n, 4),
            "closest_name_accepted": round(hard / n, 4), "examples": hard_examples[:8]}


def lasa(t: float) -> dict:
    drugs = {L.base_name(d): normalize(d.ar) for d in L.DRUGS} if hasattr(L, "base_name") else {d.en: normalize(d.ar) for d in L.DRUGS}
    rejected, pairs = 0, []
    for a, b in L.LASA_PAIRS:
        if a in drugs and b in drugs:
            ok = D.similarity(a, drugs[b]) < t and D.similarity(b, drugs[a]) < t
            rejected += ok
            pairs.append({"pair": [a, b], "rejected": bool(ok)})
    return {"both_directions_rejected": rejected, "n": len(pairs), "pairs": pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    rows = [r for r in load() if sealed(r[0]) == args.test]
    names = [r[0] for r in rows]
    global NEAREST
    NEAREST = {en: nearest(en, names) for en in names}
    out_path = ROOT / "results" / "drug_names_eval.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if args.test:
        out["sealed_test"] = rates(rows, D.MATCH_THRESHOLD)
        print(json.dumps(out["sealed_test"], ensure_ascii=False, indent=1))
    else:
        out["development"] = [rates(rows, t) for t in THRESHOLDS]
        out["lasa_pairs"] = lasa(D.MATCH_THRESHOLD)
        out["threshold"] = D.MATCH_THRESHOLD
        for r in out["development"]:
            print({k: v for k, v in r.items() if k != "examples"})
        print(json.dumps(out["lasa_pairs"], ensure_ascii=False)[:900])
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    if args.test:
        from sealed_log import record
        record("drug names, sealed fifth", out_path)


if __name__ == "__main__":
    main()
