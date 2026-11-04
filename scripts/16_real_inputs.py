"""Real inputs: patient-facing sentences from US FDA drug labels (openFDA, public domain), translated by
the open MT systems. There are no reference translations and no error labels, so this measures how much
real text the checker can verify and what it says about it. It is not an error rate.

Labels are split by hash into development and sealed sets: the checker may be changed after reading
development output, and the sealed set is scored once.

    python scripts/16_real_inputs.py fetch
    python scripts/16_real_inputs.py translate
    python scripts/16_real_inputs.py evaluate            # development
    python scripts/16_real_inputs.py evaluate --sealed
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.arabic import number_words_to_digits, normalize, to_western  # noqa: E402
from amana.guard2 import check  # noqa: E402

SENTENCES = ROOT / "data" / "processed" / "fda_sentences.jsonl"
OUTPUTS = ROOT / "results" / "fda_mt_outputs.jsonl"
API = "https://api.fda.gov/drug/label.json"
SEARCH = '_exists_:spl_patient_package_insert AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
PAGES, PAGE_SIZE, PER_LABEL, PER_SPLIT = 12, 100, 3, 300
SYSTEMS = ("NLLB-200-600M", "M2M100-418M", "OPUS-MT-en-ar")
START = re.compile(r"^(?:take|use|apply|inject|inhale|give|do not|don't|never|avoid|stop|keep|swallow|store|if you miss|"
                   r"the (?:usual|recommended) dose)\b", re.I)
DOSE = re.compile(r"\b\d+(?:\.\d+)? ?(?:mg|mcg|g|ml|units?|tablets?|capsules?|times|hours?|days?|weeks?)\b", re.I)


def sealed(set_id: str) -> bool:
    return int(hashlib.sha256(set_id.encode()).hexdigest(), 16) % 2 == 0


def sentences_of(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text))
    return [p.strip(" •-") for p in parts]


def keep(s: str) -> bool:
    n = len(s.split())
    return 5 <= n <= 35 and s.endswith(".") and not re.search(r"https?:|www\.|1-\d{3}|\bFDA\b", s) \
        and bool(START.search(s) or DOSE.search(s))


def fetch():
    rows, seen = [], set()
    for page in range(PAGES):
        url = API + "?" + urllib.parse.urlencode({"search": SEARCH, "limit": PAGE_SIZE, "skip": page * PAGE_SIZE})
        with urllib.request.urlopen(url, timeout=120) as r:
            results = json.load(r)["results"]
        for label in results:
            set_id = label.get("set_id") or label.get("id")
            found = []
            for s in sentences_of(" ".join(label.get("spl_patient_package_insert", []))):
                if keep(s) and s not in seen:
                    seen.add(s)
                    found.append(s)
            rng = random.Random(set_id)
            for s in rng.sample(found, min(PER_LABEL, len(found))):
                rows.append({"set_id": set_id, "generic_name": (label.get("openfda", {}).get("generic_name") or [""])[0].lower(),
                             "src_en": s, "sealed": sealed(set_id)})
        print(f"page {page + 1}/{PAGES}: {len(rows)} sentences", flush=True)
        time.sleep(0.5)
    rng = random.Random(13)
    picked = []
    for flag in (False, True):
        part = [r for r in rows if r["sealed"] == flag]
        picked += rng.sample(part, min(PER_SPLIT, len(part)))
    with SENTENCES.open("w", encoding="utf-8", newline="\n") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(picked)} of {len(rows)} sentences, {len({r['set_id'] for r in picked})} labels -> {SENTENCES}")


def translate():
    spec = importlib.util.spec_from_file_location("mt", ROOT / "scripts" / "06_mt_audit.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    rows = [json.loads(x) for x in SENTENCES.read_text(encoding="utf-8").splitlines()]
    with OUTPUTS.open("w", encoding="utf-8", newline="\n") as f:
        for system in SYSTEMS:
            cands, revision = mt.translate(system, [r["src_en"] for r in rows], nbest=1)
            for r, c in zip(rows, cands):
                f.write(json.dumps({**r, "system": system, "revision": revision, "hyp_ar": c[0]}, ensure_ascii=False) + "\n")


def numbers_lost(src: str, hyp: str) -> bool:
    """Parser-free: a digit sequence in the source that is nowhere in the translation."""
    want = set(re.findall(r"\d+(?:\.\d+)?", src.replace(",", "")))
    have = set(re.findall(r"\d+(?:\.\d+)?", number_words_to_digits(normalize(to_western(hyp))).replace(",", "")))
    return bool(want - have)


def evaluate(is_sealed: bool):
    rows = [json.loads(x) for x in OUTPUTS.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["sealed"] == is_sealed]
    out = {}
    for system in SYSTEMS:
        part = [r for r in rows if r["system"] == system]
        reports = [check(r["src_en"], r["hyp_ar"]) for r in part]
        n = len(part)
        kinds = Counter(f.kind for x in reports for f in x.flags)
        review = Counter(f.kind for x in reports for f in x.review)
        out[system] = {"n": n, "passed": round(sum(x.ok for x in reports) / n, 4),
                       "flagged": round(sum(bool(x.flags) for x in reports) / n, 4),
                       "review_only": round(sum(not x.flags and bool(x.review) for x in reports) / n, 4),
                       "mean_coverage": round(sum(x.coverage for x in reports) / n, 4),
                       "fully_covered": round(sum(not x.unverified for x in reports) / n, 4),
                       "general_path": round(sum(x.general for x in reports) / n, 4),
                       "passed_on_general_path": round(sum(x.ok and x.general for x in reports) / n, 4),
                       "numbers_lost_parser_free": sum(numbers_lost(r["src_en"], r["hyp_ar"]) for r in part),
                       "flag_kinds": dict(kinds.most_common()), "review_kinds": dict(review.most_common())}
        print(system, {k: v for k, v in out[system].items()}, flush=True)
    path = ROOT / "results" / ("fda_sealed.json" if is_sealed else "fda_dev.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "translate", "evaluate"])
    ap.add_argument("--sealed", action="store_true")
    args = ap.parse_args()
    if args.step == "fetch":
        fetch()
    elif args.step == "translate":
        translate()
    else:
        evaluate(args.sealed)


if __name__ == "__main__":
    main()
