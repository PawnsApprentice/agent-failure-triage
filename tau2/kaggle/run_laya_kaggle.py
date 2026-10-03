"""Score laya-typed-decisions on the tau2 calibration + test runs on a Kaggle GPU.

It imports the repo's own code, shipped next to this file (common/laya_agent.py, common/metrics.py,
tau2/experiment.py, laya_check/typed_decisions.py), so prompts, truncation, wording selection and
temperature fitting are the same code that runs locally. On a GPU laya runs in fp16 rather than the
CPU's fp32, so the run checks itself before scoring tau2:

  0. Step 0 again, on this GPU: typed-decisions test split must match 0.766 +/- 0.03 (else stop),
     plus decision-level agreement with the local CPU Step 0 predictions.
  1. Three wordings on the calibration split; pick the lowest NLL at each wording's refit temperature.
  2. Score the test split with the chosen wording.
  3. CPU-vs-GPU parity on calibration runs already scored locally.

Outputs (in /kaggle/working): laya_predictions.csv, laya_wording.json, the per-wording .jsonl
prediction files, step0_gpu.json, parity.json, run_info.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else ROOT / "output"
SMOKE = int(os.environ.get("LAYA_SMOKE", "0"))  # >0: only N items per stage, to test the wiring end to end
for p in (ROOT, ROOT / "tau2", ROOT / "laya_check"):
    sys.path.insert(0, str(p))

import experiment as ex  # imports common.laya_agent first, which sets USE_TF=0 before transformers loads
import typed_decisions as td

from common import laya_agent
from common.metrics import auroc

TYPED_DECISIONS_PARQUET = ROOT / "laya_check" / "data" / "all" / "test-00000-of-00001.parquet"
CPU_STEP0 = ROOT / "reference" / "step0_cpu_predictions.jsonl"
CPU_TAU2 = ROOT / "reference" / "laya_calibration_W1_cpu.jsonl"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_info() -> dict:
    import laya
    import torch

    agent = laya_agent.get_agent(ex.LAYA_CHECKPOINT)
    info = {"laya": laya.__version__, "torch": torch.__version__, "device": str(agent.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": str(agent.dtype), "amp": bool(agent.amp_enabled)}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def step0() -> None:
    cases = td.build_cases(TYPED_DECISIONS_PARQUET)[: SMOKE or None]
    decisions, latency, gpu_preds = [], [], {}
    for c in cases:
        out = laya_agent.predict(td.CHECKPOINT, c["state"], c["questions"])
        latency.append(out["latency_ms"])
        for qid, q in c["questions"].items():
            probs = td.option_probs(q, out["answers"][qid])
            decisions.append({"type": q["type"], "workflow": c["workflow"], "gold": c["gold"][qid], "probs": probs})
            gpu_preds[(c["id"], qid)] = probs
    overall = td.summarize(decisions)
    by_type = {t: td.summarize([d for d in decisions if d["type"] == t])["accuracy"] for t in td.PUBLISHED["by_type"]}
    gap = overall["accuracy"] - td.PUBLISHED["accuracy"]

    agree = diffs = n = 0
    for line in CPU_STEP0.read_text().splitlines():
        row = json.loads(line)
        for d in row.get("decisions", []):
            if (row["case_id"], d["qid"]) not in gpu_preds:  # only in smoke mode
                continue
            g = gpu_preds[(row["case_id"], d["qid"])]
            agree += max(range(len(g)), key=g.__getitem__) == max(range(len(d["probs"])), key=d["probs"].__getitem__)
            diffs = max(diffs, max(abs(a - b) for a, b in zip(g, d["probs"])))
            n += 1
    result = {"accuracy": overall["accuracy"], "published": td.PUBLISHED["accuracy"], "gap": gap,
              "pass": abs(gap) <= td.TOLERANCE, "by_type": by_type,
              "decision_agreement_with_cpu": agree / n, "max_abs_prob_diff_vs_cpu": diffs,
              "latency_ms_per_case_p50": sorted(latency)[len(latency) // 2]}
    (OUT / "step0_gpu.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"Step 0 on this device: accuracy {overall['accuracy']:.4f} (published 0.766, gap {gap:+.4f}) by type {by_type}")
    log(f"  agreement with local CPU decisions {agree}/{n} = {agree / n:.4f}, max |prob diff| {diffs:.4f}")
    if not result["pass"] and not SMOKE:
        raise SystemExit("Step 0 FAILED on this GPU - the harness does not reproduce Laya's published accuracy. Stopping.")


def parity() -> dict:
    cpu = {r["run_id"]: r["p_fail"] for r in ex.load_jsonl(CPU_TAU2)}
    gpu = {r["run_id"]: r["p_fail"] for r in ex.load_jsonl(OUT / "laya_calibration_W1.jsonl") if r["run_id"] in cpu}
    diffs = [abs(cpu[k] - gpu[k]) for k in gpu]
    same_side = sum((cpu[k] >= 0.5) == (gpu[k] >= 0.5) for k in gpu)
    result = {"n": len(diffs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
              "same_side_of_0.5": same_side}
    (OUT / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"CPU vs GPU on {len(diffs)} tau2 calibration runs (W1): max |diff| {result['max_abs_diff']:.4f}, "
        f"mean {result['mean_abs_diff']:.4f}, same side of 0.5 {same_side}/{len(diffs)}")
    return result


def write_csv(rows: list[dict]) -> Path:
    meta = {r["run_id"]: r for r in rows}
    path = OUT / "laya_predictions.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "wording", "run_id", "domain", "label_failed", "p_fail", "latency_ms", "state_tokens"])
        for pred_file in sorted(OUT.glob("laya_calibration_W*.jsonl")) + sorted(OUT.glob("pred_laya-typed_W*.jsonl")):
            split = "calibration" if pred_file.name.startswith("laya_calibration") else "test"
            wid = pred_file.stem.rsplit("_", 1)[1]
            for p in ex.load_jsonl(pred_file):
                r = meta[p["run_id"]]
                w.writerow([split, wid, p["run_id"], r["domain"], int(r["failed"]), repr(p["p_fail"]),
                            f"{p['latency_ms']:.3f}", p["state_tokens"]])
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = run_info()
    log(f"running on {info['device']} ({info['gpu']}), dtype {info['dtype']}, laya {info['laya']}")
    if info["gpu"] is None:
        log("WARNING: no GPU found - check Settings > Accelerator. This will take ~10 hours on CPU.")

    step0()
    rows = ex.load_jsonl(ROOT / "tau2_runs.jsonl")
    cal = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    if SMOKE:
        cpu_ids = {r["run_id"] for r in ex.load_jsonl(CPU_TAU2)}
        cal = [r for r in cal if r["run_id"] in cpu_ids][:SMOKE] + [r for r in cal if r["failed"]][:SMOKE]
        test = test[:SMOKE] + [r for r in test if r["failed"]][:SMOKE]
        log(f"SMOKE MODE: {len(cal)} calibration and {len(test)} test runs only")
    log(f"tau2: {len(cal)} calibration runs, {len(test)} test runs")

    selection = ex.select_wording(cal, OUT)
    log(f"chosen wording {selection['chosen']}: {ex.LAYA_WORDINGS[selection['chosen']]!r}")
    for wid, s in selection["scores"].items():
        log(f"  {wid}: NLL shipped-T {s['nll_shipped']:.4f}, refit-T {s['nll_refit']:.4f} (T {s['t_refit']:.3f}), "
            f"calibration AUROC {s['auroc_calibration']:.3f}")

    test_path = ex.score_test(test, OUT)
    preds = ex.load_jsonl(test_path)
    labels = {r["run_id"]: r["failed"] for r in test}
    log(f"test: {len(preds)} runs scored, AUROC (shipped T) "
        f"{auroc([p['p_fail'] for p in preds], [labels[p['run_id']] for p in preds]):.3f}")

    parity()
    log(f"wrote {write_csv(rows)}; total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
