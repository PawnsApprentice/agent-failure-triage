"""Step 0 harness check: run laya-typed-decisions through our shared Laya wrapper
(common/laya_agent.py) on the LocalLLaMA/typed-decisions test split and compare against the
published 0.766 accuracy. Case building and scoring mirror laya's own
research/scripts/bench_local.py (build_typed_decisions + metrics): gold index from the gold
label, prediction = argmax over each question's option probabilities with shipped temperatures.

usage: python laya_check/typed_decisions.py          # resumes from predictions.jsonl if present
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common import laya_agent  # noqa: E402
from common.metrics import ece_confidence, multiclass_brier, percentile, wilson_ci  # noqa: E402

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
PRED_PATH = HERE / "predictions.jsonl"
RESULTS_PATH = HERE / "results.json"
DATASET = "LocalLLaMA/typed-decisions"
DATASET_REVISION = "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18"
PARQUET = "all/test-00000-of-00001.parquet"
CHECKPOINT = "laya-typed"

# From the laya-typed-decisions model card (self-reported; leaderboard discussion #2).
PUBLISHED = {
    "accuracy": 0.766,
    "by_type": {"noul": 0.857, "choice": 0.733, "score": 0.723},
    "by_workflow": {"invoice_processing": 0.804, "security_incidents": 0.766,
                    "customer_service": 0.764, "agent_trace_observability": 0.730},
}
TOLERANCE = 0.03


def download() -> Path:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=DATASET_REVISION,
                                local_dir=DATA_DIR))


def build_cases(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq

    cases = []
    for r in pq.read_table(parquet).to_pylist():
        questions, gold = json.loads(r["questions"]), json.loads(r["gold"])
        state = r["state"]
        try:
            state = json.loads(state)
        except (TypeError, json.JSONDecodeError):
            pass
        gold_idx = {}
        for qid, q in questions.items():
            label = str(gold[qid]["label"])
            if q["type"] == "choice":
                gold_idx[qid] = list(q["criteria"]).index(label)
            elif q["type"] == "noul":
                gold_idx[qid] = 1 if label.lower() == "true" else 0
            else:
                gold_idx[qid] = int(label)
        cases.append({"id": r["id"], "workflow": r["workflow"], "state": state,
                      "questions": questions, "gold": gold_idx})
    return cases


def option_probs(question: dict, answer: dict) -> list[float]:
    """Option probabilities in the same order as the gold index: criteria order for choice,
    [false, true] for noul, rubric levels for score."""
    if question["type"] == "choice":
        return [float(answer["probabilities"][k]) for k in question["criteria"]]
    if question["type"] == "noul":
        p = float(answer["noul"])
        return [1 - p, p]
    return [float(answer["probabilities"][str(i)]) for i in range(len(question["criteria"]))]


def run_predictions(cases: list[dict]) -> None:
    done = set()
    if PRED_PATH.exists():
        done = {json.loads(line)["case_id"] for line in PRED_PATH.read_text().splitlines() if line}
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(done)} cases already scored, {len(todo)} to go", file=sys.stderr)
    with PRED_PATH.open("a") as f:
        for i, c in enumerate(todo, 1):
            row = {"case_id": c["id"], "workflow": c["workflow"]}
            try:
                out = laya_agent.predict(CHECKPOINT, c["state"], c["questions"])
                row.update(latency_ms=out["latency_ms"], usage=out["usage"], decisions=[
                    {"qid": qid, "type": q["type"], "gold": c["gold"][qid],
                     "probs": option_probs(q, out["answers"][qid])}
                    for qid, q in c["questions"].items()])
            except Exception as e:  # external data: record the failure, score it as dropped
                row.update(error=f"{type(e).__name__}: {e}")
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", file=sys.stderr)


def summarize(decisions: list[dict]) -> dict:
    probs = [d["probs"] for d in decisions]
    gold = [d["gold"] for d in decisions]
    pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
    correct = [p == g for p, g in zip(pred, gold)]
    k, n = sum(correct), len(correct)
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "accuracy": round(k / n, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "brier": round(multiclass_brier(probs, gold), 4),
        "nll": round(sum(-math.log(max(p[g], 1e-12)) for p, g in zip(probs, gold)) / n, 4),
        "ece": round(ece_confidence([max(p) for p in probs], correct), 4),
    }


def report(n_cases: int) -> dict:
    rows = [json.loads(line) for line in PRED_PATH.read_text().splitlines() if line]
    errors = [r for r in rows if "error" in r]
    scored = [r for r in rows if "error" not in r]
    decisions = [dict(d, workflow=r["workflow"]) for r in scored for d in r["decisions"]]
    n_expected = sum(len(r.get("decisions", [])) for r in scored) + 5 * len(errors)

    overall = summarize(decisions)
    overall_all = round(sum(max(range(len(d["probs"])), key=d["probs"].__getitem__) == d["gold"]
                            for d in decisions) / max(n_expected, 1), 4)
    by_type, by_workflow = defaultdict(list), defaultdict(list)
    for d in decisions:
        by_type[d["type"]].append(d)
        by_workflow[d["workflow"]].append(d)
    lat = [r["latency_ms"] for r in scored]
    gap = overall["accuracy"] - PUBLISHED["accuracy"]
    result = {
        "checkpoint": laya_agent.LAYA_CHECKPOINTS[CHECKPOINT],
        "dataset": f"{DATASET}@{DATASET_REVISION}",
        "cases_total": n_cases, "cases_scored": len(scored), "cases_errored": len(errors),
        "errors": [{"case_id": r["case_id"], "error": r["error"]} for r in errors[:10]],
        "overall": overall,
        "accuracy_counting_errored_as_wrong": overall_all,
        "published_accuracy": PUBLISHED["accuracy"],
        "gap": round(gap, 4),
        "pass": abs(gap) <= TOLERANCE and len(scored) == n_cases,
        "by_type": {t: summarize(v) | {"published": PUBLISHED["by_type"].get(t)} for t, v in sorted(by_type.items())},
        "by_workflow": {w: summarize(v) | {"published": PUBLISHED["by_workflow"].get(w)}
                        for w, v in sorted(by_workflow.items())},
        "latency_ms_per_case": {"p50": round(percentile(lat, 0.5)), "p95": round(percentile(lat, 0.95))} if lat else None,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    cases = build_cases(download())
    run_predictions(cases)
    r = report(len(cases))
    o = r["overall"]
    print(f"\n=== Step 0: {r['checkpoint']} on {r['dataset']} ===")
    print(f"cases scored {r['cases_scored']}/{r['cases_total']} (errored {r['cases_errored']}), decisions {o['n']}")
    print(f"accuracy {o['accuracy']:.4f}  95% CI [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]  "
          f"published {r['published_accuracy']}  gap {r['gap']:+.4f}  ->  {'PASS' if r['pass'] else 'FAIL'}"
          f" (tolerance +/-{TOLERANCE})")
    print(f"brier {o['brier']}  nll {o['nll']}  ece {o['ece']}")
    print(f"{'':28s}{'ours':>8s}{'published':>11s}{'n':>6s}")
    for group in ("by_type", "by_workflow"):
        for name, m in r[group].items():
            pub = f"{m['published']:.3f}" if m["published"] is not None else "-"
            print(f"  {name:26s}{m['accuracy']:8.3f}{pub:>11s}{m['n']:6d}")
    if r["latency_ms_per_case"]:
        print(f"latency per case (5 questions): p50 {r['latency_ms_per_case']['p50']}ms  "
              f"p95 {r['latency_ms_per_case']['p95']}ms")
    for e in r["errors"]:
        print(f"  error {e['case_id']}: {e['error']}")


if __name__ == "__main__":
    main()
