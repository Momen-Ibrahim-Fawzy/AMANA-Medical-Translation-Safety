"""The PEACH track's plumbing, on made-up rows. The corpus itself is not in this repository."""
import importlib.util
import io
import json
import random
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("peach", ROOT / "scripts" / "17_peach.py")
peach = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peach)


def sheet(rows):
    cells = "".join(f'<row><c r="A{i}" t="inlineStr"><is><t>{a}</t></is></c><c r="B{i}" t="inlineStr"><is><t>{b}</t></is></c></row>'
                    for i, (a, b) in enumerate(rows, 1))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{peach.MAIN[1:-1]}"><sheetData>{cells}</sheetData></worksheet>')
    return buffer.getvalue()


def row(doc, en, sealed):
    return {"doc": doc, "row": 1, "en": en, "ar": "خذ قرصا واحدا", "sealed": sealed}


def test_read_sheet_returns_both_columns():
    assert peach.read_sheet(sheet([("Take 1 tablet.", "خذ قرصا.")])) == [("Take 1 tablet.", "خذ قرصا.")]


def test_a_sentence_in_both_halves_stays_out_of_development():
    en = "Take 1 tablet of this medicine once a day."
    kept = peach.select([row("a", en, False), row("b", en, True), row("c", en.replace("1", "2"), False)])
    assert [(r["doc"], r["sealed"]) for r in kept if r["en"] == en] == [("b", True)]
    assert any(not r["sealed"] for r in kept)


def test_change_unit_needs_mg_on_both_sides():
    rng = random.Random(1)
    assert "ميكروغرام" in peach.change_unit(rng, "Take 5 mg daily.", "تناول 5 ملجم يوميا.")[0]
    assert peach.change_unit(rng, "Take 1 tablet daily.", "تناول 5 ملجم يوميا.") is None
    assert peach.change_unit(rng, "Take 5 mg daily.", "تناول قرصا يوميا.") is None


def test_committed_results_hold_no_corpus_text():
    for path in (ROOT / "results").glob("peach_*.json"):
        assert not re.search("[ء-ي]", path.read_text(encoding="utf-8")), path
        json.loads(path.read_text(encoding="utf-8"))
