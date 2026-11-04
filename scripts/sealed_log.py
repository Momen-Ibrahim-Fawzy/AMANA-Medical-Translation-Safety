"""Every scoring of a sealed set appends one line to results/sealed_runs.jsonl: when, which script, the code
revision and a hash of the result. The file is committed, so a re-scoring is visible in the history.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "sealed_runs.jsonl"


def record(what: str, result: Path) -> None:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "amana"], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip())
    except OSError:
        rev, dirty = None, None
    line = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "what": what, "code_revision": rev,
            "uncommitted_checker_changes": dirty,
            "result_sha256": hashlib.sha256(result.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    with LOG.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line) + "\n")
    print(f"sealed run logged -> {LOG.name}")
