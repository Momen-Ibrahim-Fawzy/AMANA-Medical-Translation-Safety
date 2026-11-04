"""Two controls for AMANA-Bench, both proposed by an outside review.

1. Source-blind baseline: a character n-gram classifier trained on train-drug Arabic only, never the English
   source, scored on held-out drugs. If it separates safe from critical translations, the labels leak through the
   Arabic text, and any learned model has to beat this, not 0.5.
2. Normalised surface metrics: the metric audit scored raw strings. Here BLEU, chrF, chrF++ and TER are scored on the
   same pairs after amana.arabic.normalize (digits, hamza, letter variants), broken down by the kind of benign
   change: orthographic only, lexical only, or both.

    python scripts/18_benchmark_controls.py      # CPU; needs data/processed/amana_bench.jsonl and results/audit_set.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.arabic import normalize  # noqa: E402

ORTHOGRAPHIC = {"indic_digits", "hamza_free"}


def source_blind(rows: list[dict]) -> dict:
    by_split = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    def pairs(split):
        texts, labels, types, seen = [], [], [], set()
        for r in by_split[split]:
            for c in ("ref_ar", "good_ar"):
                if (r["base_id"], c) not in seen:  # each safe text once per instruction
                    seen.add((r["base_id"], c))
                    texts.append(r[c]); labels.append(0); types.append("safe")
            for c in ("bad_ar", "bad_matched_ar"):
                texts.append(r[c]); labels.append(1); types.append(r["error_type"])
        return texts, np.array(labels), types

    xtr, ytr, _ = pairs("train")
    xte, yte, tte = pairs("test")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, C=4.0).fit(vec.fit_transform(xtr), ytr)
    p = clf.predict_proba(vec.transform(xte))[:, 1]
    safe = p[yte == 0]
    per_type = {}
    for t in sorted(set(tte) - {"safe"}):
        idx = np.array([x == t for x in tte])
        per_type[t] = round(float(roc_auc_score(np.r_[np.zeros(len(safe)), np.ones(idx.sum())], np.r_[safe, p[idx]])), 3)
    # words that occur in critical translations but in no safe translation anywhere in the benchmark
    safe_words = Counter(w for r in rows for c in ("ref_ar", "good_ar") for w in re.findall(r"[^\s.،]+", r[c]))
    tells, total = Counter(), Counter(r["error_type"] for r in rows)
    for r in rows:
        for c in ("bad_ar", "bad_matched_ar"):
            tells[r["error_type"]] += any(w not in safe_words for w in re.findall(r"[^\s.،]+", r[c]))
    return {"auroc_heldout_drugs": round(float(roc_auc_score(yte, p)), 3), "n": int(len(yte)),
            "auroc_by_error_type": per_type,
            "critical_with_a_word_no_safe_text_has": {t: round(tells[t] / (2 * n), 3) for t, n in sorted(total.items())}}


def normalised_metrics(rows: list[dict]) -> dict:
    bleu, chrf, chrfpp, ter = BLEU(effective_order=True), CHRF(), CHRF(word_order=2), TER()
    metrics = {"BLEU": lambda h, r: bleu.sentence_score(h, [r]).score, "chrF": lambda h, r: chrf.sentence_score(h, [r]).score,
               "chrF++": lambda h, r: chrfpp.sentence_score(h, [r]).score, "TER": lambda h, r: -ter.sentence_score(h, [r]).score}
    prefer = lambda g, b: 1.0 if g > b else 0.5 if g == b else 0.0  # noqa: E731
    out = {}
    for name, m in metrics.items():
        res = {}
        for label, f in (("raw", lambda x: x), ("normalised", normalize)):
            rew, same, kinds = [], [], defaultdict(list)
            for r in rows:
                ref, good, bad, badm = (f(r[k]) for k in ("ref_ar", "good_ar", "bad_ar", "bad_matched_ar"))
                g = m(good, ref)
                p = prefer(g, m(bad, ref))
                rew.append(p)
                same.append(prefer(g, m(badm, ref)))
                knobs = set(r["style"])
                kinds["orthographic only" if knobs <= ORTHOGRAPHIC else "lexical only" if not knobs & ORTHOGRAPHIC
                      else "both"].append(p)
            res[label] = {"reworded": round(float(np.mean(rew)), 4), "same_wording": round(float(np.mean(same)), 4),
                          "reworded_by_change": {k: {"rate": round(float(np.mean(v)), 4), "n": len(v)}
                                                 for k, v in sorted(kinds.items())}}
        out[name] = res
    return out


def main():
    bench = [json.loads(x) for x in (ROOT / "data" / "processed" / "amana_bench.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = [json.loads(x) for x in (ROOT / "results" / "audit_set.jsonl").read_text(encoding="utf-8").splitlines()]
    out = {"source_blind": source_blind(bench), "normalised_metrics": normalised_metrics(audit)}
    (ROOT / "results" / "benchmark_controls.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
