"""Format check: laya-typed-decisions was fine-tuned on short structured agent-trace summaries
(LocalLLaMA/typed-decisions agent_trace_observability: agent, constraints, task, trace_summary
counters - no conversation). Re-render 200 calibration runs in that shape and score wording W3 on
CPU, to see whether the transcript format is why Laya is near chance.

Only fields observable in a run are filled; nothing is guessed. Counters whose meaning differs from
the benchmark's (its `irreversible_actions`, `constraint_violations`) are not faked: state-changing
tool calls get their own honest name, and constraint violations are omitted.

usage: python tau2/format_check.py      # resumable; run under nice -n 19
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import experiment as ex  # noqa: E402  (also puts the repo root on sys.path)
from common import laya_agent  # noqa: E402
from common.metrics import auroc, cluster_bootstrap_ci  # noqa: E402

RAW = HERE / "data" / "raw" / "bucket"
OUT_PATH = ex.RESULTS / "format_check_W3.jsonl"
SAMPLE_PATH = ex.RESULTS / "format_check_sample.jsonl"
N_SAMPLE = 200
WORDING = "W3"

WRITE_TOOLS = {
    "airline": {"book_reservation", "cancel_reservation", "update_reservation_baggages",
                "update_reservation_flights", "update_reservation_passengers"},
    "retail": {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
               "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
               "return_delivered_order_items"},
    "telecom": {"enable_roaming", "refuel_data", "resume_line", "send_payment_request"},
}


def sample_calibration() -> list[dict]:
    """All calibration failures plus successes filling the rest, stratified by domain."""
    cal = ex.split_rows("calibration")
    failed = [r for r in cal if r["failed"]]
    ok_by_domain = defaultdict(list)
    for r in cal:
        if not r["failed"]:
            ok_by_domain[r["domain"]].append(r)
    n_ok = N_SAMPLE - len(failed)
    total_ok = sum(len(v) for v in ok_by_domain.values())
    rng = random.Random(ex.SEED)
    picked = list(failed)
    for dom, rs in sorted(ok_by_domain.items()):
        picked += rng.sample(rs, round(n_ok * len(rs) / total_ok))
    return picked


def structured_state(sim: dict, domain: str, agent_model: str) -> dict:
    msgs = sim["messages"]
    agent_calls = [c["name"] for m in msgs if m["role"] == "assistant" for c in m.get("tool_calls") or []]
    first_user = next((m["content"] for m in msgs if m["role"] == "user" and m.get("content")), "")
    return {
        "agent": {"autonomy": "unsupervised", "model": agent_model},
        "constraints": [f"Follow the {domain} customer-service policy"],
        "task": first_user[:300],
        "trace_summary": {
            "steps": sum(m["role"] == "assistant" for m in msgs),
            "tool_calls": len(agent_calls),
            "tool_errors": sum(bool(m.get("error")) for m in msgs if m["role"] == "tool"),
            "state_changing_actions": sum(c in WRITE_TOOLS[domain] for c in agent_calls),
            "handoffs_to_human": sum(c == "transfer_to_human_agents" for c in agent_calls),
            "customer_device_actions": sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "user"),
            "duration_s": round(float(sim.get("duration") or 0), 1),
            "ended_by": sim.get("termination_reason"),
        },
    }


def render_sample(rows: list[dict]) -> list[dict]:
    by_file = defaultdict(list)
    for r in rows:
        by_file[RAW / r["submission"] / r["file"]].append(r)
    out = []
    for path, rs in by_file.items():
        sims = {(str(s["task_id"]), s["trial"]): s for s in json.loads(path.read_text())["simulations"]}
        for r in rs:
            sim = sims[(r["task_id"], r["trial"])]
            out.append({"run_id": r["run_id"], "domain": r["domain"], "group": r["group"], "failed": r["failed"],
                        "state": structured_state(sim, r["domain"], r["agent_model"])})
    return out


def main() -> None:
    if not SAMPLE_PATH.exists():
        ex.write_jsonl(SAMPLE_PATH, render_sample(sample_calibration()))
    sample = ex.load_jsonl(SAMPLE_PATH)
    question = ex.laya_question(WORDING)
    done = {r["run_id"] for r in ex.load_jsonl(OUT_PATH)} if OUT_PATH.exists() else set()
    with OUT_PATH.open("a") as f:
        for i, r in enumerate([r for r in sample if r["run_id"] not in done], 1):
            out = laya_agent.predict(ex.LAYA_CHECKPOINT, r["state"], {"failed": question})
            f.write(json.dumps({"run_id": r["run_id"], "p_fail": float(out["answers"]["failed"]["noul"]),
                                "latency_ms": out["latency_ms"], "input_tokens": out["usage"]["input_tokens"]}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i} scored", file=sys.stderr)

    preds = {p["run_id"]: p for p in ex.load_jsonl(OUT_PATH)}
    by_group = defaultdict(list)
    for r in sample:
        by_group[r["group"]].append((preds[r["run_id"]]["p_fail"], r["failed"]))
    items = [x for g in by_group.values() for x in g]

    def stat(xs):
        return auroc([p for p, _ in xs], [y for _, y in xs])

    lo, hi = cluster_bootstrap_ci(list(by_group.values()), stat, seed=ex.SEED)
    n_fail = sum(y for _, y in items)
    print(f"\nformat check: {len(items)} calibration runs ({n_fail} failed), structured agent-trace state, wording {WORDING}")
    print(f"AUROC {stat(items):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (task-cluster bootstrap)")
    for dom in ex.DOMAINS:
        xs = [(preds[r["run_id"]]["p_fail"], r["failed"]) for r in sample if r["domain"] == dom]
        print(f"  {dom:8s} n={len(xs):3d} failed={sum(y for _, y in xs):3d}  AUROC {stat(xs):.3f}")
    ps = sorted(p for p, _ in items)
    print(f"p_fail spread: min {ps[0]:.3f}  median {ps[len(ps) // 2]:.3f}  max {ps[-1]:.3f}; "
          f"input tokens median {sorted(p['input_tokens'] for p in preds.values())[len(preds) // 2]}")
    print("example state:", json.dumps(sample[0]["state"])[:600])


if __name__ == "__main__":
    main()
