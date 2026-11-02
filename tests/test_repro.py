"""The benchmark is not stored in git, so its rebuild has to match the recorded hash."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_all(*args):
    return subprocess.run([sys.executable, "scripts/run_all.py", "--list", *args], cwd=ROOT, capture_output=True, text=True)


def test_benchmark_rebuild_matches_manifest(tmp_path):
    out = tmp_path / "bench.jsonl"
    subprocess.run([sys.executable, "scripts/01_build_bench.py", "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
    manifest = json.loads((ROOT / "results" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(out.read_bytes()).hexdigest() == manifest["files"]["data/processed/amana_bench.jsonl"]


def test_from_step_skips_earlier_steps():
    lines = run_all("--from-step", "3").stdout.splitlines()
    assert lines[0].split()[0] == "3" and len(lines) == len(run_all().stdout.splitlines()) - 2


def test_from_step_out_of_range_is_an_error():
    assert run_all("--from-step", "999").returncode != 0
