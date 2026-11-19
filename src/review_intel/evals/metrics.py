"""Classification metrics, written out by hand so every number is easy to explain.

All functions take parallel lists of true and predicted labels. A class with no
true and no predicted examples gets 0 for precision/recall/F1 and is left out of
macro-F1, so absent categories don't drag the average down.
"""

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # how many examples truly have this label
    predicted: int  # how many times the model chose it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return _safe_div(sum(t == p for t, p in zip(y_true, y_pred, strict=True)), len(y_true))


def per_class(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, ClassMetrics]:
    """One-vs-rest precision, recall and F1 for each label."""
    pairs = list(zip(y_true, y_pred, strict=True))
    true_counts, pred_counts = Counter(y_true), Counter(y_pred)
    result: dict[str, ClassMetrics] = {}
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        precision = _safe_div(tp, pred_counts[label])
        recall = _safe_div(tp, true_counts[label])
        f1 = _safe_div(2 * precision * recall, precision + recall)
        result[label] = ClassMetrics(precision, recall, f1, true_counts[label], pred_counts[label])
    return result


def macro_f1(metrics: dict[str, ClassMetrics]) -> float:
    """Unweighted mean F1 over labels that occur, so rare categories count as much as praise."""
    present = [m.f1 for m in metrics.values() if m.support or m.predicted]
    return _safe_div(sum(present), len(present))


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """matrix[true][predicted] = count. Predictions outside `labels` (e.g. failures)
    get their own column so they are visible rather than silently dropped."""
    columns = list(labels) + sorted(set(y_pred) - set(labels))
    matrix = {t: dict.fromkeys(columns, 0) for t in labels}
    for t, p in zip(y_true, y_pred, strict=True):
        if t in matrix:
            matrix[t][p] += 1
    return matrix


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    statistic: Callable[[Sequence[str], Sequence[str]], float],
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric over paired labels.

    With 35 test reviews one review moves accuracy by ~3 points; the interval makes that
    uncertainty part of the result instead of a footnote. The seed keeps it reproducible.
    """
    pairs = list(zip(y_true, y_pred, strict=True))
    if not pairs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        stats.append(statistic([t for t, _ in sample], [p for _, p in sample]))
    stats.sort()
    tail = (1 - level) / 2
    low = stats[int(tail * (resamples - 1))]
    high = stats[round((1 - tail) * (resamples - 1))]
    return (round(low, 4), round(high, 4))
