"""LLM-judge baseline for the tau2 experiment: Claude Haiku 4.5 reads the full rendered run and
gives the probability (0-100) that the agent failed the customer's task. Blind judge: transcript
only, no task specification or policy. Every raw response is stored.

usage:
    python tau2/haiku_judge.py estimate   # sample 500 test runs, count tokens (free), print the cost
    python tau2/haiku_judge.py run        # call Haiku on the sample; resumable
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import experiment as ex  # also puts the repo root on sys.path

from common.anthropic_client import get_client

MODEL = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1.00, 5.00  # $ per million tokens (claude-api skill, cached 2026-06-24)
CONTEXT_LIMIT = 200_000
N_SAMPLE = 500
MAX_TOKENS = 64  # the answer is a one-field JSON object
SAMPLE_PATH = ex.RESULTS / "haiku_sample.json"
COUNTS_PATH = ex.RESULTS / "haiku_token_counts.json"
PRED_PATH = ex.RESULTS / "pred_haiku.jsonl"

SYSTEM = ("You audit transcripts of an AI customer-service agent. The transcript shows the agent's "
          "messages, the customer's messages, the agent's tool calls, and the tool results.")
INSTRUCTION = ("Estimate the probability, from 0 to 100, that the agent failed to complete the customer's "
               "task: that after this conversation the customer's request was not fully and correctly carried "
               "out in the company's systems.")
SCHEMA = {"type": "json_schema", "schema": {
    "type": "object",
    "properties": {"p_fail": {"type": "integer", "description": "probability 0-100 that the agent failed the customer's task"}},
    "required": ["p_fail"], "additionalProperties": False}}


def messages_for(text: str) -> list[dict]:
    return [{"role": "user", "content": f"<transcript>\n{text}\n</transcript>\n\n{INSTRUCTION}"}]


def sample() -> list[dict]:
    """500 test runs, stratified by domain x outcome so the sample keeps the test split's mix."""
    if SAMPLE_PATH.exists():
        ids = set(json.loads(SAMPLE_PATH.read_text())["run_ids"])
        return [r for r in ex.split_rows("test") if r["run_id"] in ids]
    test = ex.split_rows("test")
    strata = defaultdict(list)
    for r in test:
        strata[(r["domain"], r["failed"])].append(r)
    rng = random.Random(ex.SEED)
    picked = []
    for key, rs in sorted(strata.items()):
        picked += rng.sample(rs, round(N_SAMPLE * len(rs) / len(test)))
    SAMPLE_PATH.write_text(json.dumps({"seed": ex.SEED, "run_ids": [r["run_id"] for r in picked]}) + "\n")
    return picked


def estimate() -> None:
    rows = sample()
    counts = json.loads(COUNTS_PATH.read_text()) if COUNTS_PATH.exists() else {}
    client = get_client(max_retries=8)
    for i, r in enumerate([r for r in rows if r["run_id"] not in counts], 1):
        resp = client.messages.count_tokens(model=MODEL, system=SYSTEM, messages=messages_for(r["text"]),
                                            output_config={"format": SCHEMA})
        counts[r["run_id"]] = resp.input_tokens
        if i % 50 == 0:
            COUNTS_PATH.write_text(json.dumps(counts))
            print(f"  counted {len(counts)}/{len(rows)}", file=sys.stderr)
    COUNTS_PATH.write_text(json.dumps(counts))

    toks = [counts[r["run_id"]] for r in rows]
    over = [r for r in rows if counts[r["run_id"]] > CONTEXT_LIMIT - MAX_TOKENS]
    n_in, n_out = sum(toks), len(rows) * MAX_TOKENS
    cost = n_in / 1e6 * PRICE_IN + n_out / 1e6 * PRICE_OUT
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[(r["domain"], r["failed"])][0] += 1
    print(f"\nsample: {len(rows)} test runs, {sum(r['failed'] for r in rows)} failed "
          f"({sum(r['failed'] for r in rows) / len(rows):.1%}); by domain: "
          + ", ".join(f"{d} {sum(n for (dd, _), (n, _) in by.items() if dd == d)}" for d in ex.DOMAINS))
    s = sorted(toks)
    print(f"input tokens per call: median {s[len(s) // 2]:,}  p90 {s[int(0.9 * len(s))]:,}  max {s[-1]:,}  "
          f"total {n_in:,}")
    print(f"runs over Haiku's {CONTEXT_LIMIT:,}-token context: {len(over)}")
    print(f"estimated cost, {MODEL} at ${PRICE_IN}/${PRICE_OUT} per MTok: "
          f"${cost:.2f} (input ${n_in / 1e6 * PRICE_IN:.2f} + output at most ${n_out / 1e6 * PRICE_OUT:.2f}); "
          f"Batches API would be about ${cost / 2:.2f} but gives no per-call latency")


def run() -> None:
    rows = sample()
    done = {p["run_id"] for p in ex.load_jsonl(PRED_PATH)} if PRED_PATH.exists() else set()
    client = get_client(max_retries=8)
    todo = [r for r in rows if r["run_id"] not in done]
    print(f"Haiku judge: {len(done)} done, {len(todo)} to go", file=sys.stderr)
    with PRED_PATH.open("a") as f:
        for i, r in enumerate(todo, 1):
            t0 = time.perf_counter()
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                                          messages=messages_for(r["text"]), output_config={"format": SCHEMA})
            latency_ms = (time.perf_counter() - t0) * 1000
            text = next((b.text for b in resp.content if b.type == "text"), "")
            try:
                p = max(0, min(100, int(json.loads(text)["p_fail"]))) / 100
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                p = None  # kept with the raw response; reported as a parse failure, never imputed
            f.write(json.dumps({
                "run_id": r["run_id"], "p_fail": p, "latency_ms": latency_ms,
                "raw": {"response_text": text, "stop_reason": resp.stop_reason, "model": resp.model,
                        "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens},
            }) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    commands = {"estimate": estimate, "run": run}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--help"
    if cmd in commands:
        commands[cmd]()
    else:
        print(__doc__.strip())
