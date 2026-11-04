"""Score your own metric on the AMANA contrastive pairs.

    python -m amana benchmark my_module:my_metric
    python -m amana benchmark my_module:my_metric --pairs path/to/audit_set.jsonl   # outside a clone

The pairs ship with the repository (results/audit_set.jsonl), not with the installed package.

The function takes lists of sources, hypotheses and references, and returns one score per hypothesis, where
higher means better. The two numbers to read are the same two as the README: how often the metric prefers
the safe translation when it is reworded, and when both share the same wording.
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

PAIRS = Path(__file__).resolve().parents[1] / "results" / "audit_set.jsonl"


def score(fn, pairs_path: Path = PAIRS) -> dict:
    rows = [json.loads(x) for x in pairs_path.read_text(encoding="utf-8").splitlines()]
    src, ref = [r["src_en"] for r in rows], [r["ref_ar"] for r in rows]
    good = fn(src, [r["good_ar"] for r in rows], ref)
    bad = fn(src, [r["bad_ar"] for r in rows], ref)
    matched = fn(src, [r["bad_matched_ar"] for r in rows], ref)
    prefer = lambda a, b: 1.0 if a > b else 0.5 if a == b else 0.0  # noqa: E731
    types = defaultdict(list)
    totals = {"reworded": 0.0, "same_wording": 0.0}
    for r, g, b, m in zip(rows, good, bad, matched):
        types[r["error_type"]].append(prefer(g, b))
        totals["reworded"] += prefer(g, b)
        totals["same_wording"] += prefer(g, m)
    n = len(rows)
    return {"n": n, "reworded": round(totals["reworded"] / n, 4), "same_wording": round(totals["same_wording"] / n, 4),
            "reworded_by_error_type": {t: round(sum(v) / len(v), 4) for t, v in sorted(types.items())}}


def run(args) -> int:
    module, _, name = args.metric.partition(":")
    if not name:
        raise SystemExit("name the function as module:function, for example my_metric:score")
    pairs = Path(args.pairs) if args.pairs else PAIRS
    if not pairs.exists():
        raise SystemExit(f"pairs file not found: {pairs}. Run from a clone of the repository, or pass --pairs with the "
                         "path to results/audit_set.jsonl")
    sys.path.insert(0, str(Path.cwd()))
    print(json.dumps(score(getattr(importlib.import_module(module), name), pairs), indent=1))
    return 0
