"""Check one translation from the command line.

    python -m amana "Take 1 tablet of warfarin 3 mg by mouth once daily." "<Arabic translation>"

Exit code: 0 passes, 1 flagged, 2 needs review.
"""
from __future__ import annotations

import argparse
import sys

from .guard2 import check


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m amana", description=__doc__.splitlines()[0])
    ap.add_argument("source_en")
    ap.add_argument("translation_ar")
    args = ap.parse_args()
    report = check(args.source_en, args.translation_ar)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report.status)
    for f in report.flags:
        print(f"  flag   {f}")
    for f in report.review:
        print(f"  review {f}")
    return {"ok": 0, "flag": 1, "needs_review": 2}[report.status]


if __name__ == "__main__":
    sys.exit(main())
