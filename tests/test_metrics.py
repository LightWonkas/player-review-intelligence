import pytest

from review_intel.evals.metrics import (
    accuracy,
    bootstrap_ci,
    confusion_matrix,
    macro_f1,
    per_class,
)

LABELS = ["bug", "ads", "praise"]
TRUE = ["bug", "bug", "ads", "praise", "praise", "praise"]
PRED = ["bug", "ads", "ads", "praise", "praise", "bug"]


def test_accuracy() -> None:
    assert accuracy(TRUE, PRED) == pytest.approx(4 / 6)


def test_accuracy_requires_equal_lengths() -> None:
    with pytest.raises(ValueError):
        accuracy(["a"], [])


def test_per_class_precision_recall_f1() -> None:
    metrics = per_class(TRUE, PRED, LABELS)

    # bug: predicted twice, right once; truly bug twice, found once
    assert metrics["bug"].precision == pytest.approx(0.5)
    assert metrics["bug"].recall == pytest.approx(0.5)
    assert metrics["bug"].f1 == pytest.approx(0.5)
    # ads: predicted twice, right once; the one true ads was found
    assert metrics["ads"].precision == pytest.approx(0.5)
    assert metrics["ads"].recall == pytest.approx(1.0)
    assert metrics["ads"].f1 == pytest.approx(2 / 3)
    # praise: never wrongly predicted, found 2 of 3
    assert metrics["praise"].precision == pytest.approx(1.0)
    assert metrics["praise"].recall == pytest.approx(2 / 3)
    assert (metrics["praise"].support, metrics["praise"].predicted) == (3, 2)


def test_macro_f1_ignores_absent_labels() -> None:
    metrics = per_class(TRUE, PRED, [*LABELS, "never_seen"])

    expected = (0.5 + 2 / 3 + 0.8) / 3
    assert macro_f1(metrics) == pytest.approx(expected)


def test_zero_division_gives_zero() -> None:
    metrics = per_class(["a"], ["b"], ["a", "b"])

    assert metrics["a"].precision == 0.0
    assert metrics["b"].recall == 0.0
    assert metrics["a"].f1 == 0.0


def test_confusion_matrix_keeps_unknown_predictions_visible() -> None:
    matrix = confusion_matrix(["bug", "ads"], ["bug", "<failed>"], ["bug", "ads"])

    assert matrix["bug"] == {"bug": 1, "ads": 0, "<failed>": 0}
    assert matrix["ads"] == {"bug": 0, "ads": 0, "<failed>": 1}


def test_bootstrap_interval_brackets_the_estimate_and_narrows_with_more_data() -> None:
    small = bootstrap_ci(TRUE, PRED, accuracy)
    big = bootstrap_ci(TRUE * 20, PRED * 20, accuracy)

    assert small[0] <= 4 / 6 <= small[1]
    assert big[0] <= 4 / 6 <= big[1]
    assert (big[1] - big[0]) < (small[1] - small[0])
    assert bootstrap_ci(TRUE, PRED, accuracy) == small  # seeded: reproducible
    assert bootstrap_ci([], [], accuracy) == (0.0, 0.0)


def test_bootstrap_interval_for_a_perfect_score_is_a_point() -> None:
    assert bootstrap_ci(TRUE, TRUE, accuracy) == (1.0, 1.0)
