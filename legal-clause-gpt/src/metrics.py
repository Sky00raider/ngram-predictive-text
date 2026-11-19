"""Evaluation maths, in plain numpy: everything works on saved logits.

Notation: for N clauses and k classes,
  probs   (N, k)  softmax over the k label-token logits
  pred    (N,)    argmax class
  conf    (N,)    the probability of that argmax class ("confidence")
  target  (N,)    true class
  correct (N,)    pred == target

Abstention: the tagger answers only when conf >= threshold and otherwise says
"needs human review". Raising the threshold lowers *coverage* (the share of
clauses it answers) and, if confidence means anything, raises the accuracy on
the clauses it does answer. The threshold is chosen on validation and then
applied unchanged to test.

Calibration: a model is calibrated if, among predictions made with confidence
about p, a share of about p is correct. Expected calibration error (ECE) is the
gap |accuracy - confidence| averaged over confidence bins, weighted by how
many predictions fall in each bin.
"""
import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---- accuracy and F1 -------------------------------------------------------

def per_class(pred: np.ndarray, target: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Support, precision, recall (= per-class accuracy) and F1 for each class 0..k-1."""
    tp = np.bincount(target[pred == target], minlength=k).astype(float)
    n_true = np.bincount(target, minlength=k).astype(float)
    n_pred = np.bincount(pred, minlength=k).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(n_pred > 0, tp / n_pred, 0.0)
        recall = np.where(n_true > 0, tp / n_true, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"support": n_true.astype(int), "precision": precision, "recall": recall, "f1": f1}


def macro_f1(pred: np.ndarray, target: np.ndarray, k: int) -> float:
    """Unweighted mean F1 over the classes that occur in target or pred."""
    stats = per_class(pred, target, k)
    present = (stats["support"] > 0) | (np.bincount(pred, minlength=k) > 0)
    return float(stats["f1"][present].mean())


def confused_pairs(pred: np.ndarray, target: np.ndarray, top: int = 15) -> list[tuple[int, int, int, int]]:
    """Most confused class pairs, both directions counted together.

    Returns [(a, b, a_as_b, b_as_a)] sorted by a_as_b + b_as_a, where a_as_b
    is how many clauses of class a were predicted as b.
    """
    counts: dict[tuple[int, int], int] = {}
    for t, p in zip(target[pred != target].tolist(), pred[pred != target].tolist()):
        counts[(t, p)] = counts.get((t, p), 0) + 1
    pairs = {}
    for (t, p), n in counts.items():
        a, b = min(t, p), max(t, p)
        ab, ba = pairs.get((a, b), (0, 0))
        pairs[(a, b)] = (ab + n, ba) if t == a else (ab, ba + n)
    ranked = sorted(pairs.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
    return [(a, b, ab, ba) for (a, b), (ab, ba) in ranked[:top]]


# ---- abstention ------------------------------------------------------------

def coverage_curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Answer the n most confident clauses, for n = 1..N.

    Returns coverage (n/N), accuracy on those n, and the confidence of the
    n-th (the threshold that would keep exactly them, ties aside).
    """
    order = np.argsort(-conf, kind="stable")
    hits = np.cumsum(correct[order])
    n = np.arange(1, len(conf) + 1)
    return n / len(conf), hits / n, conf[order]


def threshold_for_accuracy(conf: np.ndarray, correct: np.ndarray, target_acc: float) -> float:
    """Lowest confidence threshold whose answered clauses reach target_acc (max coverage).

    If even the single most confident clause misses the target, returns a
    threshold above every confidence (answer nothing).
    """
    _, acc, thresholds = coverage_curve(conf, correct)
    ok = np.nonzero(acc >= target_acc)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(np.nextafter(conf.max(), 2.0))


def at_threshold(conf: np.ndarray, correct: np.ndarray, threshold: float) -> dict[str, float]:
    """Coverage and accuracy when answering only conf >= threshold."""
    keep = conf >= threshold
    return {"threshold": float(threshold),
            "coverage": float(keep.mean()),
            "accuracy": float(correct[keep].mean()) if keep.any() else float("nan"),
            "abstained": int((~keep).sum())}


# ---- calibration -----------------------------------------------------------

def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> dict:
    """Equal-width confidence bins: count, mean confidence and accuracy per bin, and ECE."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    count = np.bincount(which, minlength=bins)
    with np.errstate(invalid="ignore"):
        mean_conf = np.bincount(which, weights=conf, minlength=bins) / count
        acc = np.bincount(which, weights=correct.astype(float), minlength=bins) / count
    gaps = np.where(count > 0, np.abs(acc - mean_conf), 0.0)
    ece = float((gaps * count).sum() / count.sum())
    return {"edges": edges, "count": count, "confidence": mean_conf, "accuracy": acc, "ece": ece}


def nll(logits: np.ndarray, target: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the true class."""
    p = softmax(logits, temperature)[np.arange(len(target)), target]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    """Temperature scaling: the single T > 0 minimising NLL (fit on validation only).

    Dividing logits by T leaves every argmax, and so accuracy, unchanged; it
    only stretches or flattens the confidences. Coarse log-grid, then refine.
    """
    grid = np.exp(np.linspace(np.log(0.2), np.log(10.0), 60))
    best = grid[np.argmin([nll(logits, target, t) for t in grid])]
    fine = np.linspace(best / 1.1, best * 1.1, 41)
    return float(fine[np.argmin([nll(logits, target, t) for t in fine])])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):           # average the ranks of tied values
            tied = v == value
            r[tied] = r[tied].mean()
        return r
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])
