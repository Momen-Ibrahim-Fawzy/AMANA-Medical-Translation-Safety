"""Check medication translations from the command line.

    python -m amana "Take 1 tablet of warfarin 3 mg by mouth once daily." "<Arabic translation>"
    python -m amana batch translations.csv --report report.html
    python -m amana benchmark my_module:my_metric

Exit code for a check: 0 passes, 1 flagged, 2 needs review. For a batch: 1 if anything is flagged, otherwise 2 if
anything needs review, otherwise 0. --allow-review exits 0 when nothing is flagged; use it only if every review
item is read by a person.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import batch, benchmark
from .guard2 import check


def _single(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m amana", description=__doc__)
    ap.formatter_class = argparse.RawDescriptionHelpFormatter
    ap.add_argument("source_en")
    ap.add_argument("translation_ar")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    report = check(args.source_en, args.translation_ar)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        print(report.status)
        for f in report.flags:
            print(f"  flag   {f}")
        for f in report.review:
            print(f"  review {f}")
    return {"ok": 0, "flag": 1, "needs_review": 2}[report.status]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    if argv[:1] == ["batch"]:
        ap = argparse.ArgumentParser(prog="python -m amana batch", description=batch.__doc__.splitlines()[0])
        ap.add_argument("input")
        ap.add_argument("--report", help="write a reviewer's HTML report here")
        ap.add_argument("--json", help="write one JSON result per line here")
        ap.add_argument("--allow-review", action="store_true", help="exit 0 when rows need review but none is flagged")
        return batch.run(ap.parse_args(argv[1:]))
    if argv[:1] == ["benchmark"]:
        ap = argparse.ArgumentParser(prog="python -m amana benchmark", description=benchmark.__doc__.splitlines()[0])
        ap.add_argument("metric", help="module:function")
        ap.add_argument("--pairs", help="path to audit_set.jsonl, if not running from a clone of the repository")
        return benchmark.run(ap.parse_args(argv[1:]))
    return _single(argv)


if __name__ == "__main__":
    sys.exit(main())
