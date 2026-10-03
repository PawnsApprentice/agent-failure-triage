"""Shared Laya wiring: load a checkpoint explicitly (never the auto-selecting Router, so we
always know which checkpoint answered) and run typed questions through it."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("USE_TF", "0")  # avoids the transformers TF-probe hang noted in the laya model card

LAYA_CHECKPOINTS = {
    "laya-base": "convaiinnovations/laya",
    "laya-typed": "convaiinnovations/laya-typed-decisions",
}
NOUL_TEMP_BUCKET = "noul:2"  # laya.common.temp_bucket(QTYPES["noul"], k=2): the yes/no question bucket

_agents: dict[str, object] = {}


def get_agent(checkpoint_key: str):
    if checkpoint_key not in _agents:
        import laya

        repo_id = LAYA_CHECKPOINTS[checkpoint_key]
        print(f"loading Laya checkpoint {repo_id!r}...", file=sys.stderr)
        _agents[checkpoint_key] = laya.load(repo_id)
    return _agents[checkpoint_key]


def predict(checkpoint_key: str, state, questions: dict) -> dict:
    """One `agent.predict` call; returns laya's raw answers plus wall-clock latency."""
    agent = get_agent(checkpoint_key)
    t0 = time.perf_counter()
    result = agent.predict(state, questions)
    return {
        "checkpoint": LAYA_CHECKPOINTS[checkpoint_key],
        "answers": result["answers"],
        "usage": result.get("usage"),
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }


def noul_temperature(checkpoint_key: str) -> float:
    agent = get_agent(checkpoint_key)
    return agent.temperature_by_options.get(NOUL_TEMP_BUCKET, agent.temperature[2])


def set_noul_temperature(checkpoint_key: str, t: float) -> None:
    get_agent(checkpoint_key).temperature_by_options[NOUL_TEMP_BUCKET] = t
