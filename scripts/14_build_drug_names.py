"""Build amana/data/drug_names.tsv: English and Arabic drug names from Wikidata (CC0), for items with an
ATC code. The committed file is a snapshot, so the results do not change when Wikidata does.

    python scripts/14_build_drug_names.py
"""
from __future__ import annotations

import csv
import re
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "amana" / "data" / "drug_names.tsv"
QUERY = """SELECT DISTINCT ?en ?ar ?atc WHERE {
  ?d wdt:P267 ?atc; rdfs:label ?en, ?ar . FILTER(LANG(?en)="en" && LANG(?ar)="ar") }"""
URL = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": QUERY})
AGENT = "AMANA-research/0.3 (https://github.com/Momen-Ibrahim-Fawzy/AMANA-Medical-Translation-Safety)"


def main():
    req = urllib.request.Request(URL, headers={"Accept": "text/csv", "User-Agent": AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = list(csv.DictReader(r.read().decode("utf-8").splitlines()))
    table = {}
    for x in rows:
        en = re.sub(r"^\((?:rs|r|s|[-−+])\)-", "", x["en"].strip().lower())
        ar = x["ar"].strip()
        if re.fullmatch(r"[a-z][a-z-]+", en) and re.fullmatch(r"[ء-ْ ]+", ar) and en not in table:
            table[en] = (ar, x["atc"])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("en\tar\tatc\n")
        for en in sorted(table):
            f.write(f"{en}\t{table[en][0]}\t{table[en][1]}\n")
    print(f"{len(table)} of {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
