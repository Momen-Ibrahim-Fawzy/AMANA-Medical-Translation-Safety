"""Metric-blindness audit: do MT metrics prefer a safe paraphrase over a clinically critical error?

Runs on the held-out TEST drugs of AMANA-Bench, balanced per error type.

    python scripts/03_metric_audit.py --per-type 150
    python scripts/03_metric_audit.py --metrics AMANA-QE   # add one metric to the cached audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.metrics import METRICS, contrastive_accuracy  # noqa: E402

CACHE = ROOT / "results" / "cache"
HYPS = ("ref_ar", "good_ar", "bad_ar", "bad_matched_ar")


def audit_set(bench: Path, per_type: int, seed: int) -> pd.DataFrame:
    df = pd.read_json(bench, lines=True)
    df = df[df.split == "test"]
    parts = [g.sample(min(len(g), per_type), random_state=seed) for _, g in df.groupby("error_type")]
    return pd.concat(parts).reset_index(drop=True)


def qe_scores(name: str, src, hyp, ref):
    """AMANA-QE predictions are produced by 05_train_qe.py and looked up here."""
    preds = json.loads((ROOT / "results" / "amana_qe_predictions.json").read_text(encoding="utf-8"))
    return [preds[f"{s}\t{h}"] for s, h in zip(src, hyp)]


def score_metric(name: str, df: pd.DataFrame) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    triples = {(s, r[h], r["ref_ar"]) for _, r in df.iterrows() for s in [r["src_en"]] for h in HYPS}
    todo = [t for t in triples if "\t".join(t) not in cache]
    if todo:
        fn = qe_scores if name == "AMANA-QE" else METRICS[name]
        src, hyp, ref = map(list, zip(*todo))
        for t, v in zip(todo, fn(name, src, hyp, ref) if name == "AMANA-QE" else fn(src, hyp, ref)):
            cache["\t".join(t)] = v
        path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def summarise(name: str, df: pd.DataFrame, cache: dict) -> dict:
    get = lambda col: np.array([cache["\t".join((r.src_en, getattr(r, col), r.ref_ar))] for r in df.itertuples()])  # noqa: E731
    ref, good, bad, badm = (get(c) for c in HYPS)
    out = {
        "contrastive_minimal": contrastive_accuracy(good, bad),
        "contrastive_matched": contrastive_accuracy(good, badm),
        "auroc": float(roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(badm))], -np.r_[good, badm])),
        "mean_score": {"ref": float(ref.mean()), "good": float(good.mean()), "bad_matched": float(badm.mean())},
        "per_type": {},
    }
    # a 'blind' metric penalises a dangerous edit less than a harmless rewording
    out["critical_vs_benign_drop"] = float(np.median(ref - badm) / max(np.median(ref - good), 1e-9))
    for t, idx in df.groupby("error_type").groups.items():
        idx = np.asarray(idx)
        out["per_type"][t] = {"minimal": contrastive_accuracy(good[idx], bad[idx]),
                              "matched": contrastive_accuracy(good[idx], badm[idx])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "data" / "processed" / "amana_bench.jsonl"))
    ap.add_argument("--per-type", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metrics", nargs="*", default=list(METRICS))
    args = ap.parse_args()

    df = audit_set(Path(args.bench), args.per_type, args.seed)
    print(f"audit set: {len(df)} rows ({df.error_type.value_counts().to_dict()})")
    out_path = ROOT / "results" / "metric_audit.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name in args.metrics:
        try:
            cache = score_metric(name, df)
        except Exception as e:  # noqa: BLE001  (e.g. a gated checkpoint)
            print(f"  {name}: skipped ({type(e).__name__}: {str(e)[:120]})")
            continue
        results[name] = summarise(name, df, cache)
        r = results[name]
        print(f"  {name:20s} minimal={r['contrastive_minimal']:.3f} matched={r['contrastive_matched']:.3f} "
              f"auroc={r['auroc']:.3f}")
    results["_meta"] = {"rows": len(df), "per_type": args.per_type, "seed": args.seed,
                        "split": "test (held-out drugs)"}
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    df.to_json(ROOT / "results" / "audit_set.jsonl", orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
