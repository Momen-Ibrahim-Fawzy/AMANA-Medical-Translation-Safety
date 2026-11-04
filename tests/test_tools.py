"""The batch report and the benchmark harness, as a user would run them."""
import csv
import json
import subprocess
import sys
from pathlib import Path

from amana import batch, benchmark

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "translations.csv"


def amana(*args):
    return subprocess.run([sys.executable, "-m", "amana", *args], cwd=ROOT, capture_output=True, encoding="utf-8")


def test_batch_writes_report_and_json_and_signals_flags(tmp_path):
    report, out = tmp_path / "r.html", tmp_path / "r.jsonl"
    done = amana("batch", str(EXAMPLE), "--report", str(report), "--json", str(out))
    assert done.returncode == 1, done.stderr
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8 and {r["status"] for r in rows} == {"flag", "needs_review", "ok"}
    page = report.read_text(encoding="utf-8")
    assert "does not mean safe" in page and 'dir="rtl"' in page


def test_batch_fails_closed_when_rows_need_review(tmp_path):
    # an untranslated copy of the English has nothing the checker can confirm
    path = tmp_path / "t.csv"
    path.write_text("source,translation\nShake the bottle well before use.,Shake the bottle well before use.\n", encoding="utf-8")
    assert amana("batch", str(path)).returncode == 2
    assert amana("batch", str(path), "--allow-review").returncode == 0


def test_batch_escapes_html_in_translations():
    rows = batch.check_rows([{"source": "Take 1 tablet.", "translation": "<script>alert(1)</script>"}], Path("x.csv"))
    assert "<script>alert" not in batch.render_html(rows, "t")


def test_batch_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([["a", "b"], ["1", "2"]])
    done = amana("batch", str(path))
    assert done.returncode != 0 and "no column named" in done.stderr


def test_single_check_json_output():
    done = amana("Take 1 tablet of warfarin 3 mg once daily.", "خذ قرصا واحدا من وارفارين 30 ملغ مرة واحدة يوميا.", "--json")
    assert done.returncode == 1 and json.loads(done.stdout)["status"] == "flag"


def test_benchmark_prefers_a_perfect_oracle(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"src_en": "s", "ref_ar": "r", "good_ar": "g", "bad_ar": "b", "bad_matched_ar": "m",
                                 "error_type": "dose"}) + "\n", encoding="utf-8")
    result = benchmark.score(lambda s, h, r: [1.0 if x == "g" else 0.0 for x in h], pairs)
    assert result["reworded"] == 1 and result["same_wording"] == 1


def test_benchmark_runs_on_the_shipped_pairs():
    done = amana("benchmark", "examples.my_metric:char_overlap")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["n"] == 1620


def test_sealed_runs_are_logged(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import sealed_log
    monkeypatch.setattr(sealed_log, "LOG", tmp_path / "sealed_runs.jsonl")
    result = tmp_path / "r.json"
    result.write_text("{}", encoding="utf-8")
    sealed_log.record("a sealed set", result)
    line = json.loads((tmp_path / "sealed_runs.jsonl").read_text(encoding="utf-8"))
    assert line["what"] == "a sealed set" and len(line["result_sha256"]) == 64
