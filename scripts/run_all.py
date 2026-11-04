"""Run the experiments in order. Each stage uses files from the stages before it.

    python scripts/run_all.py --list
    python scripts/run_all.py --stages report        # no GPU, uses the committed results
    python scripts/run_all.py                        # everything: needs a GPU and model downloads
    python scripts/run_all.py --from-step 8          # resume after a failure, using the numbers from --list
    python scripts/run_all.py --stages sealed        # scores the sealed sets; only once the checker is frozen
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
    "mt": ["06_mt_audit.py translate --n 300", "06_mt_audit.py evaluate", "16_real_inputs.py translate"],
    "peach": ["17_peach.py evaluate"],
    "report": ["01_build_bench.py", "04_guard_eval.py", "08_error_patterns.py", "09_hard_cases.py", "10_guard2_eval.py",
               "10_guard2_eval.py --probes", "18_benchmark_controls.py", "15_drug_names_eval.py", "16_real_inputs.py evaluate",
               "07_figures.py", "13_manifest.py"],
    # never part of a default run; each scoring of a sealed set is logged in results/sealed_runs.jsonl
    "sealed": ["10_guard2_eval.py --holdout", "15_drug_names_eval.py --test", "16_real_inputs.py evaluate --sealed",
               "17_peach.py evaluate --sealed"],
}
DEFAULT = [s for s in STAGES if s != "sealed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=DEFAULT, choices=list(STAGES))
    ap.add_argument("--from-step", type=int, default=1, help="resume at this step number, as shown by --list")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    steps = [(stage, cmd) for stage, cmds in STAGES.items() for cmd in cmds]
    if not 1 <= args.from_step <= len(steps):
        ap.error(f"--from-step must be between 1 and {len(steps)}")
    for i, (stage, cmd) in enumerate(steps, 1):
        if stage not in args.stages or i < args.from_step:
            continue
        line = [sys.executable, str(ROOT / "scripts" / cmd.split()[0]), *cmd.split()[1:]]
        print(f"{i:2d} [{stage}] {cmd}", flush=True)
        if not args.list:
            subprocess.run(line, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
