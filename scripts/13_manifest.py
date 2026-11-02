"""Write results/MANIFEST.json: checksums of data and results, the code revision, package versions
and model revisions, so a table in the README can be traced to what produced it.

    python scripts/13_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "unbabel-comet", "sacrebleu", "bert-score", "scikit-learn", "pandas", "numpy", "scipy")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main():
    files = sorted(p for d in ("data/processed", "results") for p in (ROOT / d).glob("*")
                   if p.is_file() and p.name != "MANIFEST.json" and not p.name.startswith("log_"))
    files.append(ROOT / "data" / "hard_cases_holdout.jsonl")
    rows = [json.loads(l) for l in (ROOT / "results" / "mt_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    models = {r["system"]: {"revision": r.get("revision"), **r.get("config", {})} for r in rows}
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {"code_revision_before_this_commit": rev, "python": platform.python_version(),
           "packages": {p: version(p) for p in PACKAGES}, "mt_models": models,
           "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}}
    (ROOT / "results" / "MANIFEST.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out['files'])} files -> results/MANIFEST.json")


if __name__ == "__main__":
    main()
