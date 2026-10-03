"""Build the Kaggle uploads for the tau2 experiment.

usage:
    python tau2/kaggle/build_package.py                 # dataset: tau2/kaggle/upload/ (laya-tau2.zip + metadata)
    python tau2/kaggle/build_package.py kernel --smoke                   # smoke run, both phases
    python tau2/kaggle/build_package.py kernel --full --phase select     # full run, phase 1
    python tau2/kaggle/build_package.py kernel --full --phase seeds --setting S1  # full run, phase 2

The dataset holds all tau2 runs (train/calibration/test) and the repo's own code, copied byte for
byte, so Kaggle runs the same logic as the laptop. Push it with
`kaggle datasets version -p tau2/kaggle/upload -m "..."` and the kernel with
`kaggle kernels push -p tau2/kaggle/kernel_finetune`.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PKG = HERE / "package"
UPLOAD = HERE / "upload"
ZIP_PATH = UPLOAD / "laya-tau2.zip"
KERNEL_DIR = HERE / "kernel_finetune"
KAGGLE_USER = "pawnsapprentice"
DATASET = f"{KAGGLE_USER}/laya-tau2"
KERNEL = f"{KAGGLE_USER}/laya-tau2-finetune"

sys.path.insert(0, str(REPO_ROOT / "tau2"))
import experiment as ex

CODE = {
    "common/__init__.py": "common/__init__.py",
    "common/laya_agent.py": "common/laya_agent.py",
    "common/metrics.py": "common/metrics.py",
    "tau2/experiment.py": "tau2/experiment.py",
    "laya_check/typed_decisions.py": "laya_check/typed_decisions.py",
    "laya_check/data/all/test-00000-of-00001.parquet": "laya_check/data/all/test-00000-of-00001.parquet",
    "tau2/kaggle/run_laya_kaggle.py": "run_laya_kaggle.py",
    "tau2/kaggle/train_ddp_laya.py": "train_ddp_laya.py",
    "tau2/kaggle/finetune_kaggle.py": "finetune_kaggle.py",
    # CPU references for the GPU parity checks
    "laya_check/predictions.jsonl": "reference/step0_cpu_predictions.jsonl",
    "tau2/results/cpu_partial/laya_calibration_W1.jsonl": "reference/laya_calibration_W1_cpu.jsonl",
}
RUN_FIELDS = ("run_id", "domain", "group", "failed", "tail_1024")


def build_dataset() -> None:
    if PKG.exists():
        shutil.rmtree(PKG)
    for src, dst in CODE.items():
        (PKG / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / src, PKG / dst)

    assignment = json.loads(ex.SPLIT_PATH.read_text())["groups"]
    rows = [{k: r[k] for k in RUN_FIELDS} | {"split": assignment[r["group"]]} for r in ex.load_jsonl(ex.CORPUS_PATH)]
    ex.write_jsonl(PKG / "tau2_runs.jsonl", rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "calibration", "test")}

    UPLOAD.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PKG.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(PKG))
    (UPLOAD / "dataset-metadata.json").write_text(json.dumps(
        {"id": DATASET, "title": "laya-tau2", "licenses": [{"name": "CC0-1.0"}]}, indent=2) + "\n")
    shutil.rmtree(PKG)
    print(f"dataset: {counts} runs + code -> {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")


def build_kernel(smoke: bool, pipeline_args: list[str]) -> None:
    args = ["pipeline"] + pipeline_args + (["--smoke"] if smoke else [])
    cells = [
        ("markdown", ("# Fine-tune laya-typed on tau2 (Laya's official RLCD trainer, 2xT4)\n\n"
                      f"Run: `finetune_kaggle.py {' '.join(args)}`. All logic is in `finetune_kaggle.py` inside the "
                      "`laya-tau2` dataset; outputs land in `/kaggle/working`.")),
        ("code", ("!pip install -q laya==0.3.21\n"
                  "import torch\n"
                  "print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])")),
        ("code", ("import glob, subprocess, sys, zipfile\n"
                  "scripts = glob.glob('/kaggle/input/**/finetune_kaggle.py', recursive=True)\n"
                  "if not scripts:  # dataset zip not unpacked by Kaggle: unpack it ourselves\n"
                  "    zips = glob.glob('/kaggle/input/**/laya-tau2.zip', recursive=True)\n"
                  "    assert zips, 'laya-tau2 dataset not attached'\n"
                  "    zipfile.ZipFile(zips[0]).extractall('/tmp/laya-tau2')\n"
                  "    scripts = ['/tmp/laya-tau2/finetune_kaggle.py']\n"
                  "print('running', scripts[0])\n"
                  f"subprocess.run([sys.executable, scripts[0]] + {args!r}, check=True)")),
    ]
    nb = {"cells": [{"id": f"cell-{i}", "cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if t == "code" else {})}
                    for i, (t, s) in enumerate(cells)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    if KERNEL_DIR.exists():
        shutil.rmtree(KERNEL_DIR)
    KERNEL_DIR.mkdir(parents=True)
    (KERNEL_DIR / "finetune.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (KERNEL_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL, "title": "laya-tau2-finetune", "code_file": "finetune.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4"}, indent=2) + "\n")
    print(f"kernel -> {KERNEL_DIR}: finetune_kaggle.py {' '.join(args)}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "kernel" and sys.argv[2] in ("--smoke", "--full"):
        build_kernel(smoke=sys.argv[2] == "--smoke", pipeline_args=sys.argv[3:])
    elif len(sys.argv) > 1:
        raise SystemExit("usage: build_package.py | build_package.py kernel --smoke|--full [--phase P] [--setting S]")
    else:
        build_dataset()
