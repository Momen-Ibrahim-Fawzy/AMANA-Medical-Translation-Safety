"""Build AMANA-Bench: contrastive English -> Arabic medication-instruction translations.

Each row = one source instruction, its reference translation, a meaning-preserving paraphrase
('good') and two versions with one clinically critical error ('bad' = minimal edit of the reference,
'bad_matched' = the same paraphrase style as 'good' plus the error).

    python scripts/01_build_bench.py --per-drug 60 --seed 13
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana import lexicon as L  # noqa: E402
from amana.rx import Style, applicable_errors, inject_error, render_ar, render_en, sample_rx  # noqa: E402


def drug_splits(seed: int) -> dict[str, str]:
    """Hold out whole drugs (with their LASA partner) for dev/test."""
    groups, seen = [], set()
    for d in L.DRUGS:
        if d.en in seen:
            continue
        group = [d.en] + ([L.LASA[d.en]] if d.en in L.LASA else [])
        seen.update(group)
        groups.append(group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    split = {}
    n = len(groups)
    for i, g in enumerate(groups):
        name = "test" if i < round(0.25 * n) else "dev" if i < round(0.35 * n) else "train"
        for en in g:
            split[en] = name
    # every dosage form must appear in train
    for form in L.FORMS:
        members = [d.en for d in L.DRUGS if d.form == form]
        if all(split[m] != "train" for m in members):
            split[members[0]] = "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-drug", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "amana_bench.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = drug_splits(args.seed)
    rows, seen_src = [], set()
    for drug in L.DRUGS:
        made = tries = 0
        while made < args.per_drug and tries < args.per_drug * 20:
            tries += 1
            rx = sample_rx(rng, drug)
            src = render_en(rx)
            if src in seen_src:
                continue
            seen_src.add(src)
            made += 1
            ref = render_ar(rx)
            style = Style.random_benign(rng, rx)
            good = render_ar(rx, style)
            base_id = f"{drug.en.replace(' ', '_')}-{made:03d}"
            for kind in applicable_errors(rx):
                bad_rx, desc = inject_error(rng, rx, kind)
                rows.append({
                    "id": f"{base_id}-{kind}", "base_id": base_id, "split": split[drug.en],
                    "drug": drug.en, "form": drug.form, "error_type": kind, "error": desc,
                    "lasa": "(LASA)" in desc, "src_en": src, "ref_ar": ref, "good_ar": good,
                    "bad_ar": render_ar(bad_rx), "bad_matched_ar": render_ar(bad_rx, style),
                    "style": {k: v for k, v in asdict(style).items() if v}, "slots": rx.slots(),
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows from {len(seen_src)} instructions -> {out}")
    print("splits:", Counter(r["split"] for r in rows))
    print("errors:", Counter(r["error_type"] for r in rows))
    print("test drugs:", sorted({r["drug"] for r in rows if r["split"] == "test"}))


if __name__ == "__main__":
    main()
