"""Recurring clinical error patterns in the 1-best MT outputs, found in the manual review.

    python scripts/08_error_patterns.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.guard import check  # noqa: E402

PATTERNS = {
    # name: (applies to source, error in 1-best output)
    "mcg_rendered_as_mg": (lambda r, rep: "mcg" in r["src_en"],
                           lambda r, rep: any(f.kind == "unit" and re.search(r"mcg rendered as [\d.]+ mg", f.detail)
                                              for f in rep.flags)),
    "safety_warning_dropped": (lambda r, rep: rep.src.prohibitions > 0,
                               lambda r, rep: any(f.kind == "omission" for f in rep.flags)),
    "glimepiride_as_contraceptive": (lambda r, rep: "glimepiride" in r["src_en"],
                                     lambda r, rep: "منع الحمل" in r["cands"][0]),
    "rash_as_stool": (lambda r, rep: "rash" in r["src_en"],
                      lambda r, rep: "البراز" in r["cands"][0]),
    "instill_as_still": (lambda r, rep: r["src_en"].startswith("Instill"),
                         lambda r, rep: bool(re.search("لا [تي]زال", r["cands"][0]))),
}


def main():
    rows = [json.loads(l) for l in (ROOT / "results" / "mt_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    out = {}
    for r in rows:
        rep = check(r["src_en"], r["cands"][0])
        for name, (applies, error) in PATTERNS.items():
            d = out.setdefault(r["system"], {}).setdefault(name, {"errors": 0, "applicable": 0})
            if applies(r, rep):
                d["applicable"] += 1
                d["errors"] += error(r, rep)
    (ROOT / "results" / "mt_error_patterns.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for system, d in out.items():
        print(system, {k: f"{v['errors']}/{v['applicable']}" for k, v in d.items()})


if __name__ == "__main__":
    main()
