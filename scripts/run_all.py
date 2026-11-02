"""Run the experiments in order. Each stage uses files from the stages before it.

    python scripts/run_all.py --list
    python scripts/run_all.py --stages report        # no GPU, uses the committed results
    python scripts/run_all.py                        # everything: needs a GPU and model downloads
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "data": ["01_build_bench.py", "02_build_tico.py"],
    "audit": ["03_metric_audit.py", "11_metric_stats.py"],
    "qe": ["05_train_qe.py --epochs 2", "03_metric_audit.py --metrics AMANA-QE",
           "05_train_qe.py --exclude-types unit laterality negation --tag types",
           "05_train_qe.py --exclude-forms eye_drop inhaler --tag forms"],
    "mt": ["06_mt_audit.py translate --n 300", "06_mt_audit.py evaluate"],
    "report": ["04_guard_eval.py", "08_error_patterns.py", "09_hard_cases.py", "10_guard2_eval.py",
               "07_figures.py", "13_manifest.py"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    for stage in args.stages:
        for cmd in STAGES[stage]:
            line = [sys.executable, str(ROOT / "scripts" / cmd.split()[0]), *cmd.split()[1:]]
            print(f"[{stage}] {cmd}", flush=True)
            if not args.list:
                subprocess.run(line, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
