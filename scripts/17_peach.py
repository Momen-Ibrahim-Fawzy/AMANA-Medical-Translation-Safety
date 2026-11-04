"""PEACH: real human English->Arabic patient leaflet sentences (Al-Sabbagh, 2024).

Two questions, on sentences about taking a medicine:
  clean    - what does the checker say about published translations? A flag is not automatically a false
             alarm: leaflets can really differ from their English, and nobody has adjudicated them.
  injected - if one number, unit or negation is broken, does a sentence that passed when clean stop passing?

Leaflets are split by hash into development and sealed sets, a sentence in both goes to sealed, and the sealed
set is scored once. The corpus is not redistributed here: results hold counts and file/row references, no text.
Download the zip from https://data.mendeley.com/datasets/5k6yrrhng7/1 and save it as data/raw/peach/peach.zip.

    python scripts/17_peach.py evaluate                    # development
    python scripts/17_peach.py evaluate --sealed
    python scripts/17_peach.py evaluate --dump out.jsonl   # write development rows that did not pass, to read
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amana.guard import check as check_baseline  # noqa: E402
from amana.guard2 import check  # noqa: E402

ZIP = ROOT / "data" / "raw" / "peach" / "peach.zip"
LEAFLETS = "Patient Information Leaflets"
MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
START = re.compile(r"^(?:take|use|apply|inject|inhale|give|do not|don't|never|avoid|stop|keep|swallow|store|if you miss|"
                   r"the (?:usual|recommended) dose)\b", re.I)
DOSE = re.compile(r"\b\d+(?:\.\d+)? ?(?:mg|mcg|g|ml|units?|tablets?|capsules?|times|hours?|days?|weeks?)\b", re.I)
AR_MG = re.compile(r"(?<![ء-ي])(?:ملغم|ملغ|ملجم|ملج|مجم|مغ)(?![ء-ي])")
EN_MG = re.compile(r"\d ?(?:mg|milligrams?)\b", re.I)
BOOTSTRAPS = 1000


def read_sheet(data: bytes) -> list[tuple[str, str]]:
    """The first two columns of the first sheet of an .xlsx file."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(MAIN + "si"):
                shared.append("".join(t.text or "" for t in si.iter(MAIN + "t")))
        rows = []
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(MAIN + "row"):
            cells = {}
            for c in row.findall(MAIN + "c"):
                value = c.find(MAIN + "v")
                if c.get("t") == "s" and value is not None:
                    cells[c.get("r")[0]] = shared[int(value.text)]
                elif c.get("t") == "inlineStr":
                    cells[c.get("r")[0]] = "".join(t.text or "" for t in c.iter(MAIN + "t"))
                elif value is not None:
                    cells[c.get("r")[0]] = value.text
            rows.append((cells.get("A", "").strip(), cells.get("B", "").strip()))
    return rows


def is_sealed(doc: str) -> bool:
    return int(hashlib.sha256(f"peach:{doc}".encode()).hexdigest(), 16) % 2 == 0


def keep(en: str, ar: str) -> bool:
    return 5 <= len(en.split()) <= 35 and bool(re.search("[ء-ي]", ar)) and not re.search(r"https?:|www\.", en) \
        and bool(START.search(en) or DOSE.search(en))


def select(pairs: list[dict]) -> list[dict]:
    """Keep each English sentence once. One that occurs in any sealed leaflet belongs to the sealed set."""
    sealed_text = {p["en"].lower() for p in pairs if p["sealed"]}
    seen, out = set(), []
    for p in sorted(pairs, key=lambda x: not x["sealed"]):
        key = p["en"].lower()
        if key not in seen and keep(p["en"], p["ar"]) and (p["sealed"] or key not in sealed_text):
            seen.add(key)
            out.append(p)
    return out


def load(path: Path = ZIP) -> list[dict]:
    pairs = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            p = PurePosixPath(name)
            if p.parent.name == LEAFLETS and p.suffix == ".xlsx":
                for i, (en, ar) in enumerate(read_sheet(z.read(name)), 1):
                    pairs.append({"doc": p.stem, "row": i, "en": en, "ar": ar, "sealed": is_sealed(p.stem)})
    return select(pairs)


def change_unit(rng: random.Random, en: str, ar: str) -> tuple[str, str] | None:
    if not EN_MG.search(en):
        return None
    found = list(AR_MG.finditer(ar))
    if not found:
        return None
    m = rng.choice(found)
    return ar[:m.start()] + "ميكروغرام" + ar[m.end():], "unit mg -> mcg"


def injectors():
    spec = importlib.util.spec_from_file_location("tico", ROOT / "scripts" / "02_build_tico.py")
    tico = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tico)
    return {"number": tico.change_number, "negation": tico.drop_negation, "unit": change_unit}


def cluster_ci(values: list[tuple[str, bool]]) -> list[float] | None:
    """95% interval from resampling leaflets, because sentences from one leaflet are not independent."""
    if not values:
        return None
    docs: dict[str, list[int]] = {}
    for doc, v in values:
        docs.setdefault(doc, []).append(int(v))
    keys = list(docs)
    rng = random.Random(13)
    rates = []
    for _ in range(BOOTSTRAPS):
        hits = total = 0
        for k in rng.choices(keys, k=len(keys)):
            hits += sum(docs[k])
            total += len(docs[k])
        rates.append(hits / total)
    rates.sort()
    return [round(rates[int(0.025 * BOOTSTRAPS)], 4), round(rates[int(0.975 * BOOTSTRAPS) - 1], 4)]


def rate(values: list[tuple[str, bool]]) -> dict:
    n = len(values)
    return {"n": n, "rate": round(sum(v for _, v in values) / n, 4) if n else None, "ci95_by_leaflet": cluster_ci(values)}


def evaluate(rows: list[dict]) -> tuple[dict, list[dict]]:
    tools = injectors()
    reports = [check(r["en"], r["ar"]) for r in rows]
    baseline = [check_baseline(r["en"], r["ar"]) for r in rows]
    out = {"sentences": len(rows), "leaflets": len({r["doc"] for r in rows}),
           "clean": {"passed": rate([(r["doc"], x.ok) for r, x in zip(rows, reports)]),
                     "flagged": rate([(r["doc"], bool(x.flags)) for r, x in zip(rows, reports)]),
                     "review_only": rate([(r["doc"], not x.flags and bool(x.review)) for r, x in zip(rows, reports)]),
                     "fully_covered": rate([(r["doc"], not x.unverified) for r, x in zip(rows, reports)]),
                     "general_path": rate([(r["doc"], x.general) for r, x in zip(rows, reports)]),
                     "passed_on_general_path": rate([(r["doc"], x.ok and x.general) for r, x in zip(rows, reports)]),
                     "mean_coverage": round(sum(x.coverage for x in reports) / len(rows), 4),
                     "baseline_passed": rate([(r["doc"], x.ok) for r, x in zip(rows, baseline)]),
                     "flag_kinds": dict(Counter(f.kind for x in reports for f in x.flags).most_common()),
                     "review_kinds": dict(Counter(f.kind for x in reports for f in x.review).most_common())},
           "injected": {}}
    for kind, fn in tools.items():
        applied, caught, flagged, baseline_caught = 0, [], [], []
        for r, x, b in zip(rows, reports, baseline):
            found = fn(random.Random(f"{kind}:{r['doc']}:{r['row']}"), r["en"], r["ar"])
            if found is None:
                continue
            applied += 1
            if x.ok:
                bad = check(r["en"], found[0])
                caught.append((r["doc"], not bad.ok))
                flagged.append((r["doc"], bool(bad.flags)))
            if b.ok:
                baseline_caught.append((r["doc"], not check_baseline(r["en"], found[0]).ok))
        out["injected"][kind] = {"applied": applied, "passed_when_clean": len(caught), "stopped_passing": rate(caught),
                                 "flagged": rate(flagged), "baseline_stopped_passing": rate(baseline_caught)}
    dump = [{"doc": r["doc"], "row": r["row"], "en": r["en"], "ar": r["ar"], **x.to_dict()}
            for r, x in zip(rows, reports) if not x.ok]
    return out, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["evaluate"])
    ap.add_argument("--sealed", action="store_true")
    ap.add_argument("--dump", type=Path, help="development only: rows that did not pass, including corpus text")
    args = ap.parse_args()
    if args.dump and args.sealed:
        ap.error("--dump is for development rows")
    if not ZIP.exists():
        print(f"{ZIP} is missing, skipping PEACH; the committed results stay as they are. See this file's docstring.")
        return
    rows = [r for r in load() if r["sealed"] == args.sealed]
    result, dump = evaluate(rows)
    print(json.dumps(result, indent=1))
    path = ROOT / "results" / ("peach_sealed.json" if args.sealed else "peach_dev.json")
    path.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    if args.sealed:
        from sealed_log import record
        record("PEACH leaflets, sealed half", path)
    if args.dump:
        with args.dump.open("w", encoding="utf-8", newline="\n") as f:
            for row in dump:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
