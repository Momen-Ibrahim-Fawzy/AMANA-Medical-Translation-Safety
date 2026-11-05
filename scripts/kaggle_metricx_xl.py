"""MetricX-24 XL on Kaggle: the XL checkpoint needs about 30 GB of GPU memory in float32, more than the 8 GB GPU used for
everything else. Run this file as a Kaggle script with the "GPU T4 x2" accelerator and internet on, with a private
dataset holding amana_code.tar.gz, made in the repository root by

    tar --exclude=__pycache__ -czf amana_code.tar.gz amana scripts data/processed/amana_bench.jsonl \
        results/metric_audit.json results/benchmark_controls.json results/audit_set.jsonl

Then copy the MetricX-24_XL*.json outputs to results/cache/, merge the two "MetricX-24 XL" entries of the output
benchmark_controls.json into results/, and run `03_metric_audit.py --metrics "MetricX-24 XL" "MetricX-24 XL (QE)"`
and `11_metric_stats.py`, which read the cache and load no model.
"""
import glob, os, shutil, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu==2.6.0"], check=True)
os.makedirs("/kaggle/working/repo", exist_ok=True)
archives = glob.glob("/kaggle/input/**/amana_code.tar.gz", recursive=True)
if archives:
    subprocess.run(["tar", "-xzf", archives[0], "-C", "/kaggle/working/repo"], check=True)
else:  # Kaggle unpacked the archive on upload
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/amana/metrics.py", recursive=True)[0]))
    shutil.copytree(root, "/kaggle/working/repo", dirs_exist_ok=True)
os.chdir("/kaggle/working/repo")
import torch, transformers
import sentencepiece
print("torch", torch.__version__, "transformers", transformers.__version__, "sentencepiece", sentencepiece.__version__, "gpus", torch.cuda.device_count(), flush=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
subprocess.run([sys.executable, "scripts/03_metric_audit.py", "--metrics", "MetricX-24 XL", "MetricX-24 XL (QE)"], check=True, env=env)
subprocess.run([sys.executable, "scripts/18_benchmark_controls.py", "--learned"], check=True, env=env)
for f in glob.glob("results/cache/MetricX-24_XL*.json") + ["results/metric_audit.json", "results/benchmark_controls.json"]:
    shutil.copy(f, "/kaggle/working/" + os.path.basename(f))
shutil.rmtree("/kaggle/working/repo")
print("ALL DONE", flush=True)
