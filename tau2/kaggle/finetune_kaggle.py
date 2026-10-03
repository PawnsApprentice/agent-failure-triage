"""Fine-tune laya-typed-decisions on tau2 with Laya's official RLCD trainer, on Kaggle 2xT4.

Same input as the zero-shot run: wording W3, last tokens of the run cut to laya's room, label = run
failed (reward 0). Each stage runs in its own process so GPU memory is freed before the next one.

usage (on Kaggle; the notebook calls `pipeline`):
    python finetune_kaggle.py pipeline [--smoke]                  # both phases in one session
    python finetune_kaggle.py pipeline --phase select [--smoke]   # phase 1 only
    python finetune_kaggle.py pipeline --phase seeds --setting S1 [--smoke]  # phase 2 only
    python finetune_kaggle.py prepare OUT_DIR [--smoke]
    python finetune_kaggle.py score MODEL_DIR SPLIT OUT_JSONL [--smoke]

Pipeline (fixed in advance):
  phase select: three settings, seed 42, each trained on the train split; temperature fitted (by
    the official script) on the calibration split; pick the lowest calibration log loss; score the
    chosen seed-42 model once on the test split.
  phase seeds: retrain the chosen setting with seeds 43 and 44; score each once on the test split.
  No seed is picked. The phases can run as separate Kaggle sessions so neither nears the 12 h limit.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "tau2"):
    sys.path.insert(0, str(p))

OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else ROOT / "output"
WORK = Path("/tmp/laya_ft")  # models and items stay off /kaggle/working, so they aren't exported

SETTINGS = {  # name: (epochs, lr_encoder, lr_head)
    "S1": (4, 2.5e-5, 1.0e-4),   # official defaults
    "S2": (2, 2.5e-5, 1.0e-4),
    "S3": (4, 1.25e-5, 5.0e-5),
}
SELECTION_SEED = 42  # reproduces the official script's shuffle seeding
EXTRA_SEEDS = (43, 44)
WORDING = "W3"
SMOKE = {"train": 64, "calibration": 16, "test": 16, "epochs": 1}


def log(msg: str) -> None:
    print(msg, flush=True)


def rows_for(split: str, smoke: bool) -> list[dict]:
    import experiment as ex

    rows = [r for r in ex.load_jsonl(ROOT / "tau2_runs.jsonl") if r["split"] == split]
    if smoke:
        failed = [r for r in rows if r["failed"]][: SMOKE[split] // 2]
        rows = failed + [r for r in rows if not r["failed"]][: SMOKE[split] - len(failed)]
    return rows


# ---------------------------------------------------------------------------
# prepare: training/calibration items, built with the official notebook's item builder
# ---------------------------------------------------------------------------


def prepare(out_dir: Path, smoke: bool) -> None:
    import experiment as ex
    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_sequence, render_options
    from transformers import AutoTokenizer

    from common import laya_agent

    # Official notebook, cell 3: tokenizer + config from the checkpoint we fine-tune.
    model_dir = snapshot_download(laya_agent.LAYA_CHECKPOINTS[ex.LAYA_CHECKPOINT])
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    question = ex.laya_question(WORDING)
    room = ex.laya_state_room(question)
    agent = laya_agent.get_agent(ex.LAYA_CHECKPOINT)

    def build_training_item(state, q, failed):
        # Official build_training_item, with the gold distribution replaced by the hard label
        # (noul options are [false, true]; true = the run failed).
        t = q["type"]
        crit = q.get("criteria", {})
        target = [0.0, 1.0] if failed else [1.0, 0.0]
        label = target.index(max(target))
        k = len(render_options({"t": t, "crit": crit}))
        seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                      cfg["max_len"], cfg["head_max_len"])
        if len(markers) != k:
            return None
        return {"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target, "label": label}

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calibration"):
        items, checked = [], 0
        for r in rows_for(split, smoke):
            state, _ = ex.tail_text(agent.tok, r["tail_1024"], room)  # identical to inference
            it = build_training_item(state, question, r["failed"])
            if it is None:
                raise SystemExit(f"{r['run_id']}: markers lost - question does not fit head_max_len")
            if checked < 50:  # training sequence must equal what laya builds at inference
                internal = agent._encode_state(state, ["failed"], {"failed": agent._to_internal(question)})[0]
                assert internal["ids"] == it["ids"] and internal["markers"] == it["markers"], r["run_id"]
                checked += 1
            items.append(it)
        torch.save(items, out_dir / f"{split}_items.pt")
        log(f"prepared {len(items)} {split} items ({sum(i['label'] for i in items)} failed); "
            f"first {checked} match laya's inference encoding")


# ---------------------------------------------------------------------------
# score: one model on one split, through the same run_laya used for zero-shot
# ---------------------------------------------------------------------------


def score(model_dir: str, split: str, out_jsonl: Path, smoke: bool) -> None:
    import experiment as ex

    from common import laya_agent

    laya_agent.LAYA_CHECKPOINTS["finetuned"] = model_dir
    ex.run_laya(rows_for(split, smoke), WORDING, out_jsonl, checkpoint="finetuned")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _run(cmd: list[str], log_path: Path | None = None) -> float:
    t0 = time.time()
    log("$ " + " ".join(cmd))
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
        lines = []
        for line in p.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    if log_path:
        log_path.write_text("".join(lines))
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return time.time() - t0


def _calibration_scores(pred_path: Path, rows: list[dict]) -> dict:
    from common.metrics import auroc

    labels = {r["run_id"]: r["failed"] for r in rows}
    preds = [json.loads(line) for line in pred_path.read_text().splitlines()]
    eps = 1e-6
    nll = -sum(math.log(max(eps, p["p_fail"])) if labels[p["run_id"]] else math.log(max(eps, 1 - p["p_fail"]))
               for p in preds) / len(preds)
    return {"nll": nll, "auroc": auroc([p["p_fail"] for p in preds], [labels[p["run_id"]] for p in preds]),
            "n": len(preds)}


def pipeline(smoke: bool, phase: str = "all", setting_arg: str | None = None) -> None:
    import experiment as ex
    import torch
    from huggingface_hub import snapshot_download

    from common import laya_agent

    OUT.mkdir(parents=True, exist_ok=True)
    flag = ["--smoke"] if smoke else []
    me = [sys.executable, str(Path(__file__).resolve())]
    n_gpu = torch.cuda.device_count()
    info = {"smoke": smoke, "phase": phase, "gpus": [torch.cuda.get_device_name(i) for i in range(n_gpu)],
            "settings": SETTINGS, "selection_seed": SELECTION_SEED, "extra_seeds": EXTRA_SEEDS,
            "fitted_temperatures": {}}
    log(f"GPUs: {info['gpus']}  phase: {phase}")
    if n_gpu < 2:
        raise SystemExit("the official trainer needs 2 GPUs (Accelerator: GPU T4 x2)")

    base_dir = snapshot_download(laya_agent.LAYA_CHECKPOINTS[ex.LAYA_CHECKPOINT])
    timings = {"prepare_s": _run(me + ["prepare", str(WORK)] + flag)}
    cal_rows = rows_for("calibration", smoke)

    def train(setting: str, seed: int) -> Path:
        epochs, lr_enc, lr_head = SETTINGS[setting]
        if smoke:
            epochs = SMOKE["epochs"]
        model_dir = WORK / f"{setting}_seed{seed}"
        secs = _run(["torchrun", "--standalone", "--nproc_per_node=2", str(ROOT / "train_ddp_laya.py"),
                     base_dir, str(model_dir), str(WORK / "train_items.pt"), str(WORK / "calibration_items.pt"),
                     str(epochs), str(lr_enc), str(lr_head), str(seed)], OUT / f"train_{setting}_seed{seed}.log")
        timings[f"train_{setting}_seed{seed}_s"] = secs
        temps = json.loads((model_dir / "rl_agent_config.json").read_text())["temperature"]
        info["fitted_temperatures"][f"{setting}_seed{seed}"] = temps
        log(f"{setting} seed {seed}: trained in {secs / 60:.1f} min, fitted temperatures {temps}")
        return model_dir

    def score_test_once(setting: str, seed: int, model_dir: Path) -> None:
        pred = OUT / f"test_{setting}_seed{seed}.jsonl"
        timings[f"score_test_{setting}_seed{seed}_s"] = _run(me + ["score", str(model_dir), "test", str(pred)] + flag)
        shutil.rmtree(model_dir, ignore_errors=True)

    if phase in ("all", "select"):
        selection = {}
        for setting in SETTINGS:
            model_dir = train(setting, SELECTION_SEED)
            pred = OUT / f"calibration_{setting}_seed{SELECTION_SEED}.jsonl"
            timings[f"score_cal_{setting}_s"] = _run(me + ["score", str(model_dir), "calibration", str(pred)] + flag)
            selection[setting] = _calibration_scores(pred, cal_rows)
            log(f"{setting}: calibration NLL {selection[setting]['nll']:.4f}, AUROC {selection[setting]['auroc']:.3f}")
        chosen = min(selection, key=lambda s: selection[s]["nll"])
        log(f"chosen setting: {chosen} {SETTINGS[chosen]}")
        for setting in SETTINGS:
            if setting != chosen:
                shutil.rmtree(WORK / f"{setting}_seed{SELECTION_SEED}", ignore_errors=True)
        score_test_once(chosen, SELECTION_SEED, WORK / f"{chosen}_seed{SELECTION_SEED}")
        (OUT / "finetune_selection.json").write_text(json.dumps(
            {"rule": "lowest calibration log loss at the temperature fitted on calibration (seed 42)",
             "scores": selection, "chosen": chosen, "info": info, "timings": timings}, indent=2) + "\n")
        setting_arg = chosen

    if phase in ("all", "seeds"):
        if setting_arg not in SETTINGS:
            raise SystemExit(f"phase seeds needs --setting one of {list(SETTINGS)}")
        for seed in EXTRA_SEEDS:
            score_test_once(setting_arg, seed, train(setting_arg, seed))
        (OUT / "finetune_seeds.json").write_text(json.dumps(
            {"setting": setting_arg, "seeds": list(EXTRA_SEEDS), "info": info, "timings": timings}, indent=2) + "\n")
    log(f"done: {sum(timings.values()) / 60:.1f} min of stages")


if __name__ == "__main__":
    argv = sys.argv[1:]
    smoke = "--smoke" in argv

    def opt(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv else None

    args = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--phase", "--setting"))]
    if args and args[0] == "pipeline":
        pipeline(smoke, opt("--phase") or "all", opt("--setting"))
    elif args and args[0] == "prepare":
        prepare(Path(args[1]), smoke)
    elif args and args[0] == "score":
        score(args[1], args[2], Path(args[3]), smoke)
    else:
        print(__doc__.strip())
