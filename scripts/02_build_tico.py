"""AMANA-TICO: real-text subset built from the TICO-19 English->Arabic test set (CC0).

Critical errors are injected into the professional Arabic reference:
  number   - one number that the source also contains is changed (x10, /10 or +/- a small amount)
  negation - a negation particle (لا / لم / لن) is removed, flipping the meaning
Benign variants are orthographic only (Arabic-Indic digits, hamza-free spelling, no final punctuation).

    python scripts/02_build_tico.py
"""
from __future__ import annotations

import csv
import json
import hashlib
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.arabic import fmt_num, strip_hamza, to_indic, to_western  # noqa: E402

URL = "https://huggingface.co/datasets/gmnlp/tico19/resolve/main/test/test.en-ar.tsv"
EN_NEG = re.compile(r"\b(?:not|no|never|don't|doesn't|didn't|can't|cannot|won't|isn't|aren't)\b", re.I)
AR_NEG_PARTICLE = re.compile(r"(?<![ء-ي])(لا|لم|لن)\s+(?=[ء-ي])")


def load() -> pd.DataFrame:
    path = ROOT / "data" / "raw" / "tico19" / "test.en-ar.tsv"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
    df["src"] = df.sourceString.str.strip()
    df["ref"] = df.targetString.str.strip()
    return df


def benign(rng: random.Random, ar: str) -> str:
    out = ar
    ops = rng.sample(["indic", "hamza", "punct"], rng.randint(1, 2))
    if "indic" in ops and re.search(r"\d", out):
        out = to_indic(out)
    if "hamza" in ops or out == ar:
        out = strip_hamza(out)
    if "punct" in ops:
        out = out.rstrip(".؟?!")
    return out


def change_number(rng: random.Random, en: str, ar: str) -> tuple[str, str] | None:
    ar_w = to_western(ar)
    en_nums = set(re.findall(r"\d+(?:\.\d+)?", en.replace(",", "")))
    cands = [m for m in re.finditer(r"\d+(?:\.\d+)?", ar_w) if m.group() in en_nums]
    # skip disease/strain names such as COVID-19, H1N1, and 4-digit years (mostly not clinical)
    cands = [m for m in cands if not re.search(r"[-A-Za-z]$", ar_w[:m.start()])
             and not (len(m.group()) == 4 and m.group().startswith(("19", "20")))]
    if not cands:
        return None
    m = rng.choice(cands)
    v = float(m.group())
    new = rng.choice([v * 10, v / 10 if v >= 10 else v + 2, v + 1 if v < 10 else v + 10])
    new_s = fmt_num(round(new, 2))
    return ar_w[:m.start()] + new_s + ar_w[m.end():], f"number {m.group()} -> {new_s}"


def drop_negation(rng: random.Random, en: str, ar: str) -> tuple[str, str] | None:
    if not EN_NEG.search(en):
        return None
    ms = list(AR_NEG_PARTICLE.finditer(ar))
    if not ms:
        return None
    m = rng.choice(ms)
    return ar[:m.start()] + ar[m.end():], f"negation '{m.group(1)}' removed"


def main():
    rng = random.Random(11)
    df = load()
    rows = []
    for r in df.itertuples():
        for kind, fn in (("number", change_number), ("negation", drop_negation)):
            res = fn(rng, r.src, r.ref)
            if res is None:
                continue
            bad, desc = res
            good = benign(rng, r.ref)
            rows.append({"id": f"{r.stringID}-{kind}", "error_type": kind, "error": desc, "src_en": r.src,
                         "ref_ar": r.ref, "good_ar": good, "bad_ar": bad,
                         "bad_matched_ar": bad if kind == "number" and re.search(r"[٠-٩]", good) is None
                         else benign(random.Random(hashlib.sha256(r.stringID.encode()).hexdigest()), bad),
                         "medical_conversation": str(r.url).endswith("medical_domain_sentences.en")})
    out = ROOT / "data" / "processed" / "amana_tico.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    s = pd.DataFrame(rows)
    print(f"{len(s)} rows -> {out}\n{s.error_type.value_counts().to_dict()}")
    print("from the CMU doctor-patient sentences:", int(s.medical_conversation.sum()))


if __name__ == "__main__":
    main()
