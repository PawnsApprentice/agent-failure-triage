"""Binary-decision metrics shared by every experiment: accuracy, confusion counts, brier, ECE,
Wilson CI, reliability bins, latency percentiles, and temperature fitting on logits."""
from __future__ import annotations

import math
from itertools import pairwise

TEMP_MIN, TEMP_MAX = 0.5, 5.0  # laya.common's clamp range for temperatures


def decide(p: float, stated: bool | None = None, threshold: float = 0.5) -> bool:
    """The scored decision: the model's own boolean where it states one, else p >= threshold."""
    return bool(stated) if stated is not None else p >= threshold


def accuracy_counts(decisions: list[bool], labels: list[bool]) -> tuple[int, int]:
    return sum(d == bool(y) for d, y in zip(decisions, labels)), len(labels)


def confusion_counts(decisions: list[bool], labels: list[bool]) -> dict[str, int]:
    c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for d, y in zip(decisions, labels):
        c[("T" if d == bool(y) else "F") + ("P" if d else "N")] += 1
    return c


def brier(preds: list[float], labels: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(preds)


def ece(preds: list[float], labels: list[float], bins: int = 10) -> float:
    """Expected Calibration Error binned on the model's own P(positive) (not max(p): with two
    classes that carries the same information without folding both sides onto one axis)."""
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    for lo, hi in pairwise(edges):
        sel = [(p, y) for p, y in zip(preds, labels) if (lo == 0 and p <= hi) or (lo < p <= hi)]
        if not sel:
            continue
        conf = sum(p for p, _ in sel) / len(sel)
        acc = sum(y for _, y in sel) / len(sel)
        total += len(sel) / len(preds) * abs(conf - acc)
    return total


def ece_confidence(conf: list[float], correct: list[bool], bins: int = 15) -> float:
    """Multiclass ECE on top-class confidence, same definition as laya's research/scripts/bench_local.py."""
    edges = [i / bins for i in range(bins + 1)]
    total = 0.0
    for lo, hi in pairwise(edges):
        sel = [(c, k) for c, k in zip(conf, correct) if lo < c <= hi]
        if sel:
            total += len(sel) / len(conf) * abs(sum(c for c, _ in sel) / len(sel) - sum(k for _, k in sel) / len(sel))
    return total


def multiclass_brier(probs: list[list[float]], gold_idx: list[int]) -> float:
    """Mean over decisions of sum_k (p_k - onehot_k)^2."""
    return sum(sum((p - (1.0 if k == g else 0.0)) ** 2 for k, p in enumerate(ps))
               for ps, g in zip(probs, gold_idx)) / len(probs)


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    return s[min(int(len(s) * pct), len(s) - 1)]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion - safer than the normal
    approximation near 0 or 1."""
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def reliability_bins(preds: list[float], labels: list[float], bins: int = 10, min_count: int = 5):
    """Equal-width bins on P(positive); adjacent bins with < min_count samples are merged, and a
    leftover tail is carried into the last bin. Returns (mean prediction, observed rate, n)."""
    edges = [i / bins for i in range(bins + 1)]
    raw = [[(p, y) for p, y in zip(preds, labels) if (lo == 0 and p <= hi) or (lo < p <= hi)]
           for lo, hi in pairwise(edges)]
    merged, carry = [], []
    for sel in raw:
        carry = carry + sel
        if len(carry) >= min_count:
            merged.append(carry)
            carry = []
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)
    return [(sum(p for p, _ in sel) / len(sel), sum(y for _, y in sel) / len(sel), len(sel)) for sel in merged]


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) statistic, ties averaged."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(bool(y) for y in labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stratified_auroc(scores: list[float], labels: list[bool], strata: list) -> float:
    """AUROC counting only (positive, negative) pairs from the same stratum, e.g. the same domain.
    Equals the per-stratum AUROCs averaged with weights n_pos * n_neg, so differences in base rate
    between strata can't raise it."""
    by = {}
    for p, y, s in zip(scores, labels, strata):
        by.setdefault(s, ([], []))
        by[s][0].append(p)
        by[s][1].append(y)
    num = den = 0.0
    for ps, ys in by.values():
        n_pos = sum(bool(y) for y in ys)
        pairs = n_pos * (len(ys) - n_pos)
        if pairs:
            num += auroc(ps, ys) * pairs
            den += pairs
    return num / den if den else float("nan")


def recall_at_budget(scores: list[float], labels: list[bool], budget: float = 0.10) -> float:
    """Share of all positives caught when the top `budget` fraction of items by score is reviewed.
    Items tied at the cut-off share the remaining review slots evenly (the expected recall under
    random tie-breaking), so the result doesn't depend on input order."""
    n_pos = sum(bool(y) for y in labels)
    if not n_pos:
        return float("nan")
    k = math.ceil(budget * len(scores))
    cut = sorted(scores, reverse=True)[k - 1]
    above = [bool(y) for p, y in zip(scores, labels) if p > cut]
    tied = [bool(y) for p, y in zip(scores, labels) if p == cut]
    return (sum(above) + (k - len(above)) * sum(tied) / len(tied)) / n_pos


def cluster_bootstrap_ci(clusters: list[list], stat, n_boot: int = 1000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for stat(items), resampling whole clusters (e.g. tasks) with replacement, so
    correlated items within a cluster don't shrink the interval. Undefined resamples are dropped."""
    import random

    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        items = [x for _ in clusters for x in clusters[rng.randrange(len(clusters))]]
        v = stat(items)
        if not math.isnan(v):
            values.append(v)
    values.sort()
    if not values:
        return float("nan"), float("nan")
    return values[int(alpha / 2 * (len(values) - 1))], values[int((1 - alpha / 2) * (len(values) - 1))]


def paired_cluster_bootstrap(clusters: list[list[tuple]], stat, n_boot: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> dict:
    """Paired bootstrap of stat(model A) - stat(model B). Items are (score_a, score_b, label, *extra),
    and stat sees (score, label, *extra), e.g. extra = domain for a within-domain statistic. Each
    resample draws whole clusters once and scores both models on the same items."""
    import random

    def diff(items):
        return stat([(a, *rest) for a, _, *rest in items]) - stat([(b, *rest) for _, b, *rest in items])

    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        v = diff([x for _ in clusters for x in clusters[rng.randrange(len(clusters))]])
        if not math.isnan(v):
            values.append(v)
    values.sort()
    return {"diff": diff([x for c in clusters for x in c]),
            "ci95": [values[int(alpha / 2 * (len(values) - 1))], values[int((1 - alpha / 2) * (len(values) - 1))]],
            "p_a_not_better": sum(v <= 0 for v in values) / len(values), "n_boot": len(values)}


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def binary_nll(z_raw: list[float], labels: list[float], t: float) -> float:
    total = 0.0
    for z, y in zip(z_raw, labels):
        p = min(max(_sigmoid(z / t), 1e-9), 1 - 1e-9)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(labels)


def binary_brier(z_raw: list[float], labels: list[float], t: float) -> float:
    return sum((_sigmoid(z / t) - y) ** 2 for z, y in zip(z_raw, labels)) / len(labels)


def fit_temperature(z_raw: list[float], labels: list[float], steps: int = 400) -> float:
    """Grid-search the temperature in laya's allowed range that minimizes NLL of
    sigmoid(z_raw / T) against binary labels."""
    grid = [TEMP_MIN + i * (TEMP_MAX - TEMP_MIN) / steps for i in range(steps + 1)]
    return min(grid, key=lambda t: binary_nll(z_raw, labels, t))
