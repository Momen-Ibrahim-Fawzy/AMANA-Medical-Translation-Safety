"""Independent check of reranking: a blinded packet for a pharmacist or clinical translator.

Each item shows an English instruction and two Arabic candidates, the system's normal output and
the reranked one, in random order. System names and checker results are hidden. Run `build`, give
adjudication/packet.html to a reviewer, then run `score` on the CSV they download.

    python scripts/12_adjudication_packet.py build --n 60
    python scripts/12_adjudication_packet.py score answers.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "adjudication"
KEY = ROOT / "_private" / "adjudication_key.json"  # kept out of the repo so the packet stays blind
STRATEGY = "Guard2+QE rerank"

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Translation Review</title>
<style>:root{--bg:#f6f7f9;--card:#fff;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--blue:#1e6bff}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--ink:#e5e7eb;--muted:#9ca3af;--line:#2a2f38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;flex:1}main{max-width:860px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
.ar{direction:rtl;text-align:right;font-size:19px;line-height:1.8;padding:8px 10px;background:var(--bg);border-radius:6px;margin:6px 0}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;margin:3px}
button.sel{background:var(--blue);border-color:var(--blue);color:#fff}textarea{width:100%;margin-top:8px;font:inherit;padding:6px;background:var(--bg);color:var(--ink);border:1px solid var(--line);border-radius:6px}
.m{color:var(--muted);font-size:13px}</style></head><body>
<header><h1>Medication translation review</h1><span id="c" class="m"></span><button id="x">Download my answers</button></header>
<main><p>For each English instruction, judge each Arabic version on whether a patient following it would receive
the instruction as written. Ignore style. Answers are saved in this browser. This is a research exercise: no patient
will receive these texts. Download your answers when finished.</p><div id="l"></div></main><script>
const I=/*ITEMS*/[];const K="amana-adj-v1";let S={};try{S=JSON.parse(localStorage.getItem(K))||{}}catch(e){}
const sv=()=>{try{localStorage.setItem(K,JSON.stringify(S))}catch(e){}};const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const O=[["safe","Safe"],["minor","Minor problem"],["critical","Clinically critical"]];
const row=(it,w)=>`<div class="m">Version ${w}</div><div class="ar">${esc(it[w])}</div>`+O.map(([v,t])=>`<button data-i="${it.id}" data-w="${w}" data-v="${v}" class="${(S[it.id]||{})[w]===v?"sel":""}">${t}</button>`).join("");
document.getElementById("l").innerHTML=I.map(it=>`<div class="card"><div class="m">#${it.id}</div><div>${esc(it.en)}</div>${row(it,"A")}${row(it,"B")}<textarea data-n="${it.id}" placeholder="Optional note">${esc((S[it.id]||{}).note||"")}</textarea></div>`).join("");
const p=()=>{document.getElementById("c").textContent=I.filter(i=>(S[i.id]||{}).A&&(S[i.id]||{}).B).length+" / "+I.length+" done"};p();
document.addEventListener("click",e=>{const b=e.target.closest("button[data-v]");if(!b)return;const{i,w,v}=b.dataset;S[i]={...(S[i]||{}),[w]:v};sv();
document.querySelectorAll(`button[data-i="${i}"][data-w="${w}"]`).forEach(x=>x.classList.toggle("sel",x===b));p()});
document.addEventListener("input",e=>{if(e.target.dataset.n){const i=e.target.dataset.n;S[i]={...(S[i]||{}),note:e.target.value};sv()}});
document.getElementById("x").onclick=()=>{const r=[["id","A","B","note"],...I.map(i=>[i.id,(S[i.id]||{}).A||"",(S[i.id]||{}).B||"",(S[i.id]||{}).note||""])];
const c="\\ufeff"+r.map(x=>x.map(y=>`"${String(y).replace(/"/g,'""')}"`).join(",")).join("\\n");const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([c],{type:"text/csv;charset=utf-8"}));a.download="adjudication_answers.csv";a.click()};
</script></body></html>"""


def build(args):
    rows = [json.loads(l) for l in (ROOT / "results" / "mt_picks.jsonl").read_text(encoding="utf-8").splitlines()]
    changed = [r for r in rows if r["pick"][STRATEGY] != 0]
    rng = random.Random(args.seed)
    rng.shuffle(changed)
    items, key = [], {}
    for i, r in enumerate(changed[:args.n], 1):
        base, reranked = r["cands"][0], r["cands"][r["pick"][STRATEGY]]
        a_is_base = rng.random() < 0.5
        items.append({"id": i, "en": r["src_en"], "A": base if a_is_base else reranked, "B": reranked if a_is_base else base})
        key[str(i)] = {"base": "A" if a_is_base else "B", "system": r["system"]}
    OUT.mkdir(exist_ok=True)
    (OUT / "packet.html").write_text(PAGE.replace("/*ITEMS*/[]", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
    KEY.parent.mkdir(exist_ok=True)
    KEY.write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} of {len(changed)} changed instructions -> {OUT / 'packet.html'}")


def score(args):
    key = json.loads(KEY.read_text(encoding="utf-8"))
    tally = {"base": {"safe": 0, "minor": 0, "critical": 0}, "reranked": {"safe": 0, "minor": 0, "critical": 0}}
    with open(args.answers, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = key[row["id"]]
            other = "B" if k["base"] == "A" else "A"
            if row[k["base"]] and row[other]:
                tally["base"][row[k["base"]]] += 1
                tally["reranked"][row[other]] += 1
    (ROOT / "results" / "adjudication.json").write_text(json.dumps(tally, indent=2), encoding="utf-8")
    print(tally)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--seed", type=int, default=31)
    s = sub.add_parser("score")
    s.add_argument("answers")
    args = ap.parse_args()
    {"build": build, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
