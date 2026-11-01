"""Real MT audit: how often do open MT systems make clinically critical errors on medication
instructions, and does safety-aware reranking of their n-best lists fix them?

    python scripts/06_mt_audit.py translate --n 300
    python scripts/06_mt_audit.py evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.guard import check  # noqa: E402

OUT = ROOT / "results" / "mt_outputs.jsonl"
SYSTEMS = {
    "NLLB-200-600M": ("facebook/nllb-200-distilled-600M", {"src_lang": "eng_Latn"}, "arb_Arab"),
    "NLLB-200-1.3B": ("facebook/nllb-200-distilled-1.3B", {"src_lang": "eng_Latn"}, "arb_Arab"),
    "M2M100-418M": ("facebook/m2m100_418M", {"src_lang": "en"}, "ar"),
    "OPUS-MT-en-ar": ("Helsinki-NLP/opus-mt-en-ar", {}, ">>ara<<"),
}
NBEST, MAX_NEW_TOKENS = 5, 160


def sources(n: int, seed: int) -> pd.DataFrame:
    bench = pd.read_json(ROOT / "data" / "processed" / "amana_bench.jsonl", lines=True)
    test = bench[bench.split == "test"].drop_duplicates("base_id")
    return test.sample(min(n, len(test)), random_state=seed)[["base_id", "drug", "form", "src_en", "ref_ar"]]


@torch.no_grad()
def translate(name: str, srcs: list[str], nbest: int = NBEST, bs: int = 16) -> tuple[list[list[str]], str | None]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    path, tok_kw, tgt = SYSTEMS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path, **tok_kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.float16 if device == "cuda" else None).to(device).eval()
    gen_kw = {"num_beams": nbest, "num_return_sequences": nbest, "max_new_tokens": MAX_NEW_TOKENS}
    if tgt.startswith(">>"):
        srcs = [f"{tgt} {s}" for s in srcs]
    else:
        gen_kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(tgt) if "nllb" in path else tok.get_lang_id(tgt)
    out = []
    for i in range(0, len(srcs), bs):
        enc = tok(srcs[i:i + bs], return_tensors="pt", padding=True, truncation=True).to(device)
        dec = tok.batch_decode(model.generate(**enc, **gen_kw), skip_special_tokens=True)
        out += [dec[j:j + nbest] for j in range(0, len(dec), nbest)]
        print(f"  {name}: {min(i + bs, len(srcs))}/{len(srcs)}", flush=True)
    revision = getattr(model.config, "_commit_hash", None)
    del model
    torch.cuda.empty_cache()
    return out, revision


def run_config(name: str, n: int, seed: int) -> dict:
    return {"model": SYSTEMS[name][0], "n": n, "seed": seed, "nbest": NBEST, "max_new_tokens": MAX_NEW_TOKENS}


def check_cached(name: str, rows: list[dict], config: dict):
    """Stop if cached outputs came from other settings or other source sentences."""
    df = sources(config["n"], config["seed"])
    same = (all(r.get("config") == config for r in rows)
            and [(r["base_id"], r["src_en"]) for r in rows] == list(zip(df.base_id, df.src_en)))
    if not same:
        sys.exit(f"{name}: cached rows in {OUT.name} do not match this run's config or sources. "
                 "Delete that system's rows to translate it again.")


def cmd_translate(args):
    df = sources(args.n, args.seed)
    done: dict[str, list[dict]] = {}
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            row = json.loads(l)
            done.setdefault(row["system"], []).append(row)
    with OUT.open("a", encoding="utf-8") as f:
        for name in args.systems:
            config = run_config(name, args.n, args.seed)
            if name in done:
                check_cached(name, done[name], config)
                print(f"{name}: cached")
                continue
            cands, revision = translate(name, df.src_en.tolist())
            for r, c in zip(df.itertuples(), cands):
                f.write(json.dumps({"system": name, "config": config, "revision": revision, "base_id": r.base_id,
                                    "drug": r.drug, "form": r.form, "src_en": r.src_en, "ref_ar": r.ref_ar,
                                    "cands": c}, ensure_ascii=False) + "\n")


def _qe_scorer():
    ckpt = ROOT / "checkpoints" / "amana-qe"
    if not ckpt.exists():
        return None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    from importlib import import_module
    predict = import_module("05_train_qe").predict
    return lambda s, h: predict(model, tok, s, h, device)


def cmd_evaluate(args):
    from amana.metrics import chrf, comet22
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()]
    for name in {r["system"] for r in rows}:
        sys_rows = [r for r in rows if r["system"] == name]
        check_cached(name, sys_rows, sys_rows[0].get("config") or run_config(name, 0, 0))
    qe = _qe_scorer()
    flat_src = [r["src_en"] for r in rows for _ in r["cands"]]
    flat_hyp = [c for r in rows for c in r["cands"]]
    qe_flat = qe(flat_src, flat_hyp) if qe else np.full(len(flat_hyp), np.nan)
    k = 0
    for r in rows:
        n = len(r["cands"])
        r["qe"] = [float(x) for x in qe_flat[k:k + n]]
        k += n
        r["reports"] = [check(r["src_en"], c) for c in r["cands"]]
        r["pick"] = {
            "1-best": 0,
            "Guard rerank": next((i for i, rep in enumerate(r["reports"]) if rep.ok), 0),
            "QE rerank": int(np.argmax(r["qe"])) if qe else 0,
            "Guard+QE rerank": (max((i for i, rep in enumerate(r["reports"]) if rep.ok), key=lambda i: r["qe"][i],
                                    default=int(np.argmax(r["qe"])) if qe else 0)),
        }
    strategies = ["1-best", "Guard rerank", "QE rerank", "Guard+QE rerank"]
    picks = [(r, s, r["cands"][r["pick"][s]]) for r in rows for s in strategies]
    srcs, hyps, refs = zip(*[(r["src_en"], h, r["ref_ar"]) for r, _, h in picks])
    chrf_s = chrf(srcs, hyps, refs)
    comet_s = comet22(srcs, hyps, refs) if not args.no_comet else [np.nan] * len(hyps)
    table, examples = [], []
    for (r, s, h), c1, c2 in zip(picks, chrf_s, comet_s):
        rep = r["reports"][r["pick"][s]]
        table.append({"system": r["system"], "strategy": s, "guard_flagged": not rep.ok,
                      "flag_types": [f.kind for f in rep.flags], "qe": r["qe"][r["pick"][s]],
                      "chrF": c1, "COMET-22": c2, "any_clean_candidate": any(x.ok for x in r["reports"])})
        if s == "1-best" and not rep.ok:
            examples.append({"system": r["system"], "src_en": r["src_en"], "mt_ar": h, "ref_ar": r["ref_ar"],
                             "flags": [str(f) for f in rep.flags], "chrF": round(c1, 1),
                             "COMET-22": None if np.isnan(c2) else round(c2, 3)})
    t = pd.DataFrame(table)
    summary = {}
    for (sysname, strat), g in t.groupby(["system", "strategy"], sort=False):
        summary.setdefault(sysname, {})[strat] = {
            "critical_flag_rate": float(g.guard_flagged.mean()),
            "qe_unsafe_rate": float((g.qe < 0.5).mean()) if qe else None,
            "chrF": float(g.chrF.mean()), "COMET-22": float(g["COMET-22"].mean()),
            "n": int(len(g)),
        }
    by_type = {sysname: dict(Counter(ft for fts in g[g.strategy == "1-best"].flag_types for ft in fts))
               for sysname, g in t.groupby("system")}
    (ROOT / "results" / "mt_audit.json").write_text(json.dumps({"summary": summary, "flag_types_1best": by_type},
                                                               indent=2), encoding="utf-8")
    pd.DataFrame(examples).to_json(ROOT / "results" / "mt_flagged_examples.jsonl", orient="records",
                                   lines=True, force_ascii=False)
    for sysname, d in summary.items():
        print(sysname)
        for strat, v in d.items():
            print(f"   {strat:16s} flagged={v['critical_flag_rate']:.3f} qe_unsafe={v['qe_unsafe_rate']} "
                  f"chrF={v['chrF']:.1f} COMET={v['COMET-22']:.4f}")
    print(json.dumps(by_type, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("translate")
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--seed", type=int, default=5)
    a.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    b = sub.add_parser("evaluate")
    b.add_argument("--no-comet", action="store_true")
    args = ap.parse_args()
    {"translate": cmd_translate, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
