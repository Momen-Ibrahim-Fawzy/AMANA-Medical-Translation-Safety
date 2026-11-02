"""Check a file of translations and write a report a reviewer can work through.

    python -m amana batch translations.csv --report report.html --json results.jsonl

The input is .csv, .tsv or .jsonl with a source column (source, src_en, en) and a translation column
(translation, hyp_ar, ar). The report lists what most needs a person first, and says what was not checked.
"""
from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from .guard2 import check

SOURCE = ("source", "src_en", "en", "english")
TRANSLATION = ("translation", "hyp_ar", "ar", "arabic")
_ORDER = {"flag": 0, "needs_review": 1, "ok": 2}
_LABEL = {"flag": "Mismatch found", "needs_review": "Needs a person", "ok": "No problem detected"}


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t" if path.suffix == ".tsv" else ","))


def _column(row: dict, names: tuple, path: Path) -> str:
    lower = {k.lower().strip(): k for k in row}
    for n in names:
        if n in lower:
            return lower[n]
    raise SystemExit(f"{path}: no column named any of {', '.join(names)}; found {', '.join(row)}")


def check_rows(rows: list[dict], path: Path) -> list[dict]:
    if not rows:
        raise SystemExit(f"{path}: no rows")
    src, hyp = _column(rows[0], SOURCE, path), _column(rows[0], TRANSLATION, path)
    return [{"line": i, "source": r[src], "translation": r[hyp], **check(r[src], r[hyp]).to_dict()}
            for i, r in enumerate(rows, 1)]


def render_html(results: list[dict], title: str) -> str:
    counts = Counter(r["status"] for r in results)
    tiles = "".join(f'<div class="tile {s}"><b>{counts.get(s, 0)}</b><span>{_LABEL[s]}</span></div>' for s in _ORDER)
    rows = []
    for r in sorted(results, key=lambda x: (_ORDER[x["status"]], x["line"])):
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in r["flags"] + r["review"])
        rows.append(f'<tr data-status="{r["status"]}"><td class="num">{r["line"]}</td>'
                    f'<td><span class="pill {r["status"]}">{_LABEL[r["status"]]}</span></td>'
                    f'<td>{html.escape(r["source"])}</td><td dir="rtl" lang="ar">{html.escape(r["translation"])}</td>'
                    f'<td><ul>{notes}</ul></td></tr>')
    return _PAGE.format(title=html.escape(title), tiles=tiles, rows="".join(rows), n=len(results))


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AMANA report</title><style>
:root{{--bg:#fff;--ink:#14181f;--muted:#5b6472;--line:#e3e6eb;--flag:#b42318;--flagbg:#fdecea;--rev:#8a5a00;--revbg:#fdf3dc;
--ok:#1a6e3c;--okbg:#e6f4ea}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181f;--ink:#e8eaee;--muted:#a3abb8;--line:#2d333d;--flag:#ff9b90;--flagbg:#3a1d1a;
--rev:#f0c674;--revbg:#33290f;--ok:#7fd6a0;--okbg:#14301f}}}}
*{{box-sizing:border-box}}body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto}}h1{{font-size:22px;margin:0 0 4px}}p{{color:var(--muted);margin:0 0 16px}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}}.tile{{flex:1 1 160px;border:1px solid var(--line);border-radius:8px;padding:12px}}
.tile b{{display:block;font-size:26px}}.tile span{{color:var(--muted);font-size:13px}}
.wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:760px}}th,td{{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left;vertical-align:top}}
th{{font-size:13px;color:var(--muted)}}td[dir=rtl]{{text-align:right;font-size:16px}}ul{{margin:0;padding-left:16px;font-size:13px}}.num{{color:var(--muted)}}
.pill{{white-space:nowrap;font-size:12px;font-weight:600;padding:2px 8px;border-radius:999px}}.pill.flag{{color:var(--flag);background:var(--flagbg)}}
.pill.needs_review{{color:var(--rev);background:var(--revbg)}}.pill.ok{{color:var(--ok);background:var(--okbg)}}
</style></head><body><main><h1>{title}</h1>
<p>{n} translations checked. "No problem detected" does not mean safe: the checker reads only what its rules cover, and it is
not clinically validated. A qualified person must review translations before patients use them.</p>
<div class="tiles">{tiles}</div><div class="wrap"><table><thead><tr><th>#</th><th>Result</th><th>Source</th><th>Translation</th>
<th>Why</th></tr></thead><tbody>{rows}</tbody></table></div></main></body></html>"""


def run(args) -> int:
    path = Path(args.input)
    results = check_rows(read_rows(path), path)
    counts = Counter(r["status"] for r in results)
    print(f"{len(results)} checked: {counts.get('flag', 0)} mismatch, {counts.get('needs_review', 0)} for review, "
          f"{counts.get('ok', 0)} no problem detected")
    if args.report:
        Path(args.report).write_text(render_html(results, f"AMANA report: {path.name}"), encoding="utf-8")
        print(f"report -> {args.report}")
    if args.json:
        with Path(args.json).open("w", encoding="utf-8", newline="\n") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"results -> {args.json}")
    if counts.get("flag"):
        return 1
    return 2 if counts.get("needs_review") and args.strict else 0
