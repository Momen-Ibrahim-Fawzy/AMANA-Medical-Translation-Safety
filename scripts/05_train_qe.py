"""AMANA-QE: a reference-free cross-encoder that scores P(translation is clinically safe | source, translation).

Trained only on TRAIN-split drugs of AMANA-Bench. Evaluated on held-out drugs (test split) and on
real TICO-19 sentences (out of domain: never seen in training).

    python scripts/05_train_qe.py --epochs 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.arabic import strip_hamza, to_indic  # noqa: E402


def pairs(df: pd.DataFrame, rng: random.Random) -> list[tuple[str, str, int]]:
    """label 1 = safe, 0 = critical error. Safe examples are augmented to balance the classes."""
    out = []
    for _, g in df.groupby("base_id"):
        r0 = g.iloc[0]
        safe = {r0.ref_ar, r0.good_ar, to_indic(r0.ref_ar), strip_hamza(r0.good_ar)}
        bad = [r.bad_ar if rng.random() < 0.5 else r.bad_matched_ar for r in g.itertuples()]
        safe = list(safe)
        out += [(r0.src_en, h, 1) for h in safe] + [(r0.src_en, h, 0) for h in bad]
        out += [(r0.src_en, rng.choice(safe), 1) for _ in range(max(0, len(bad) - len(safe)))]
    rng.shuffle(out)
    return out


@torch.no_grad()
def predict(model, tok, src, hyp, device, bs=64) -> np.ndarray:
    model.eval()
    scores = []
    for i in range(0, len(src), bs):
        enc = tok(src[i:i + bs], hyp[i:i + bs], truncation=True, max_length=192, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            logits = model(**enc).logits.float()
        scores.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def fit(args, bench, rng, device):
    train = pairs(bench[bench.split == "train"], rng)
    dev = pairs(bench[bench.split == "dev"], rng)
    print(f"train pairs {len(train)}  dev pairs {len(dev)}  device {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)

    def collate(batch):
        s, h, y = zip(*batch)
        enc = tok(list(s), list(h), truncation=True, max_length=192, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y)
        return enc

    loader = DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    for ep in range(args.epochs):
        model.train()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % 200 == 0:
                print(f"epoch {ep} step {i}/{len(loader)} loss {loss.item():.4f}", flush=True)
        s, h, y = zip(*dev)
        p = predict(model, tok, list(s), list(h), device)
        print(f"epoch {ep} dev AUROC {roc_auc_score(y, p):.4f}  acc {((p > 0.5) == np.array(y)).mean():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/LaBSE")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "amana-qe"))
    ap.add_argument("--eval-only", action="store_true", help="score with the saved checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = pd.read_json(ROOT / "data" / "processed" / "amana_bench.jsonl", lines=True)
    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(args.out)
        model = AutoModelForSequenceClassification.from_pretrained(args.out).to(device)
    else:
        tok, model = fit(args, bench, rng, device)

    # predictions for every (src, hyp) the audits need: held-out test drugs + TICO-19
    tico = pd.read_json(ROOT / "data" / "processed" / "amana_tico.jsonl", lines=True)
    test = bench[bench.split == "test"]
    cols = ["ref_ar", "good_ar", "bad_ar", "bad_matched_ar"]
    keys = {(r.src_en, getattr(r, c)) for d in (test, tico) for r in d.itertuples() for c in cols}
    keys = sorted(keys)
    p = predict(model, tok, [k[0] for k in keys], [k[1] for k in keys], device)
    preds = {f"{s}\t{h}": float(v) for (s, h), v in zip(keys, p)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "amana_qe_predictions.json").write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")

    summary = {}
    for name, d in (("bench_test_heldout_drugs", test), ("tico19_real_text", tico)):
        safe = np.r_[[preds[f"{r.src_en}\t{r.ref_ar}"] for r in d.itertuples()],
                     [preds[f"{r.src_en}\t{r.good_ar}"] for r in d.itertuples()]]
        bad = np.array([preds[f"{r.src_en}\t{r.bad_matched_ar}"] for r in d.itertuples()])
        y = np.r_[np.ones(len(safe)), np.zeros(len(bad))]
        sc = np.r_[safe, bad]
        summary[name] = {"auroc": float(roc_auc_score(y, sc)), "recall_at_0.5": float((bad < 0.5).mean()),
                         "false_alarm_at_0.5": float((safe < 0.5).mean()), "n_bad": int(len(bad))}
        print(name, summary[name])
    (ROOT / "results" / "amana_qe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
