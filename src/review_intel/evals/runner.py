"""Run the classifier on the labeled set, score it, save and compare results.

Predictions are cached per (prompt_version, model): re-running the eval without
changing the prompt costs nothing, and changing the prompt automatically triggers
a fresh, comparable run.
"""

import json
import random
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from review_intel.classify import Classifier, ClassifyOutcome
from review_intel.evals.dataset import LabeledReview
from review_intel.evals.judge import judge_summary
from review_intel.evals.metrics import accuracy, confusion_matrix, macro_f1, per_class
from review_intel.llm.pricing import UsageTotals

PREDICTIONS_DIR = Path("evals/predictions")
RESULTS_DIR = Path("evals/results")
FAILED = "<failed>"
README_START = "<!-- eval:start -->"
README_END = "<!-- eval:end -->"


@dataclass(frozen=True)
class Prediction:
    key: str
    primary_category: str | None
    categories: list[str]
    sentiment: str | None
    short_summary: str | None
    error: str | None = None

    @classmethod
    def from_outcome(cls, key: str, outcome: ClassifyOutcome) -> "Prediction":
        result = outcome.result
        if result is None:
            return cls(key, None, [], None, None, outcome.error or "unknown error")
        return cls(
            key,
            result.primary_category,
            result.categories,
            result.sentiment.value,
            result.short_summary,
        )


def predictions_path(cache_dir: Path, version: str, model: str) -> Path:
    return cache_dir / f"{version}__{model}.jsonl"


def load_predictions(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {item["key"]: Prediction(**item) for item in items}  # later lines win


def predict(
    labels: Sequence[LabeledReview],
    classifier: Classifier,
    cache_dir: Path = PREDICTIONS_DIR,
    *,
    concurrency: int = 4,
) -> tuple[dict[str, Prediction], UsageTotals]:
    """Classify labeled reviews that have no cached prediction (failed ones are retried)."""
    path = predictions_path(cache_dir, classifier.prompt_version, classifier.model)
    cached = load_predictions(path)
    todo = [
        (i, label)
        for i, label in enumerate(labels)
        if cached.get(label.key) is None or cached[label.key].error is not None
    ]
    usage = UsageTotals()
    if not todo:
        return cached, usage

    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool,
        path.open("a", encoding="utf-8") as handle,
    ):
        futures = [
            (label, pool.submit(classifier.classify, label.to_input(i))) for i, label in todo
        ]
        for label, future in futures:
            outcome = future.result()
            for item in outcome.usages:
                usage.add(item)
            prediction = Prediction.from_outcome(label.key, outcome)
            cached[label.key] = prediction
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
    return cached, usage


@dataclass
class SplitScores:
    n: int
    failed: int
    primary_accuracy: float
    macro_f1: float
    sentiment_accuracy: float
    per_class: dict[str, dict[str, Any]]
    confusion: dict[str, dict[str, int]]


def score(
    labels: Sequence[LabeledReview], predictions: dict[str, Prediction], names: Sequence[str]
) -> SplitScores:
    true_primary = [label.primary_category for label in labels]
    pred_primary = [
        (predictions.get(label.key) and predictions[label.key].primary_category) or FAILED
        for label in labels
    ]
    true_sentiment = [label.sentiment.value for label in labels]
    pred_sentiment = [
        (predictions.get(label.key) and predictions[label.key].sentiment) or FAILED
        for label in labels
    ]
    classes = per_class(true_primary, pred_primary, names)
    return SplitScores(
        n=len(labels),
        failed=pred_primary.count(FAILED),
        primary_accuracy=accuracy(true_primary, pred_primary),
        macro_f1=macro_f1(classes),
        sentiment_accuracy=accuracy(true_sentiment, pred_sentiment),
        per_class={name: m.to_dict() for name, m in classes.items()},
        confusion=confusion_matrix(true_primary, pred_primary, names),
    )


@dataclass
class JudgeScores:
    model: str
    n: int
    errors: int
    mean_score: float
    share_at_least_4: float
    low_scores: list[dict[str, Any]] = field(default_factory=list)


def run_judge(
    client: anthropic.Anthropic,
    judge_model: str,
    labels: Sequence[LabeledReview],
    predictions: dict[str, Prediction],
    n: int,
    *,
    seed: int = 11,
    concurrency: int = 4,
) -> tuple[JudgeScores, UsageTotals]:
    """Judge `n` per-review summaries (a fixed sample) for faithfulness to the review."""
    eligible = sorted(
        (label for label in labels if (p := predictions.get(label.key)) and p.short_summary),
        key=lambda label: label.key,
    )
    sample = random.Random(seed).sample(eligible, min(n, len(eligible)))
    usage = UsageTotals()

    def judge(label: LabeledReview) -> tuple[LabeledReview, Any]:
        summary = predictions[label.key].short_summary or ""
        source = f"{label.title}\n{label.body}" if label.title else label.body
        return label, judge_summary(client, judge_model, summary, [source])

    scores: list[int] = []
    low: list[dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for label, result in pool.map(judge, sample):
            if result.usage is not None:
                usage.add(result.usage)
            if result.verdict is None:
                errors += 1
                continue
            scores.append(result.verdict.score)
            if result.verdict.score <= 3:
                low.append(
                    {
                        "key": label.key,
                        "score": result.verdict.score,
                        "summary": predictions[label.key].short_summary,
                        "unsupported_claims": result.verdict.unsupported_claims,
                    }
                )
    mean = sum(scores) / len(scores) if scores else 0.0
    share = sum(s >= 4 for s in scores) / len(scores) if scores else 0.0
    return JudgeScores(judge_model, len(sample), errors, mean, share, low), usage


@dataclass
class EvalReport:
    created_at: str
    prompt_version: str
    model: str
    n_labels: int
    splits: dict[str, SplitScores]
    judge: JudgeScores | None
    cost_usd: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    labels: Sequence[LabeledReview],
    predictions: dict[str, Prediction],
    names: Sequence[str],
    *,
    prompt_version: str,
    model: str,
    judge: JudgeScores | None = None,
    cost_usd: dict[str, float | None] | None = None,
) -> EvalReport:
    splits = {
        "test": score([lb for lb in labels if lb.split == "test"], predictions, names),
        "dev": score([lb for lb in labels if lb.split == "dev"], predictions, names),
        "all": score(labels, predictions, names),
    }
    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prompt_version=prompt_version,
        model=model,
        n_labels=len(labels),
        splits=splits,
        judge=judge,
        cost_usd=cost_usd or {},
    )


def save_report(report: EvalReport, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    day = report.created_at[:10]
    path = results_dir / f"{day}_{report.prompt_version}_{report.model}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_reports(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in results_dir.glob("*.json")]
    return sorted(reports, key=lambda r: r["created_at"])


def previous_report(
    reports: list[dict[str, Any]], prompt_version: str, model: str
) -> dict[str, Any] | None:
    """The latest run of a *different* prompt or model: what this run should be compared to."""
    others = [r for r in reports if (r["prompt_version"], r["model"]) != (prompt_version, model)]
    return others[-1] if others else None


def _delta(current: float, previous: float | None) -> str:
    if previous is None:
        return ""
    diff = current - previous
    return f"  ({'+' if diff >= 0 else ''}{diff:.3f} vs previous)"


def format_report(report: EvalReport, previous: dict[str, Any] | None) -> str:
    lines = [f"Eval: {report.n_labels} labels | {report.model} | prompt {report.prompt_version}"]
    if previous:
        lines.append(f"Compared with: {previous['model']} | prompt {previous['prompt_version']}")
    for name in ("test", "dev", "all"):
        s = report.splits[name]
        prev = previous["splits"][name] if previous else None
        lines.append(
            f"  {name:<4} n={s.n:<3} primary acc {s.primary_accuracy:.3f}"
            f"{_delta(s.primary_accuracy, prev and prev['primary_accuracy'])} | "
            f"macro-F1 {s.macro_f1:.3f}{_delta(s.macro_f1, prev and prev['macro_f1'])} | "
            f"sentiment {s.sentiment_accuracy:.3f}"
            f"{_delta(s.sentiment_accuracy, prev and prev['sentiment_accuracy'])}"
            + (f" | {s.failed} failed" if s.failed else "")
        )

    overall = report.splits["all"]
    lines.append("\nPer category (all labels):  precision  recall  f1    support")
    for name, m in overall.per_class.items():
        if m["support"] or m["predicted"]:
            lines.append(
                f"  {name:<20} {m['precision']:>9.2f} {m['recall']:>7.2f} {m['f1']:>5.2f}"
                f" {m['support']:>8}"
            )

    mistakes = sorted(
        (
            (count, true, pred)
            for true, row in overall.confusion.items()
            for pred, count in row.items()
            if pred != true and count
        ),
        reverse=True,
    )[:5]
    if mistakes:
        lines.append("\nMost common confusions (true -> predicted):")
        lines += [f"  {true} -> {pred}: {count}" for count, true, pred in mistakes]

    if report.judge:
        j = report.judge
        lines.append(
            f"\nSummary faithfulness ({j.model}, n={j.n}): mean {j.mean_score:.2f}/5, "
            f"{j.share_at_least_4:.0%} scored 4+" + (f", {j.errors} errors" if j.errors else "")
        )
        for item in j.low_scores[:3]:
            lines.append(f"  {item['score']}/5 {item['summary']!r}: {item['unsupported_claims']}")
    costs = {k: v for k, v in report.cost_usd.items() if v}
    if costs:
        lines.append("\nCost this run: " + ", ".join(f"{k} ${v:.4f}" for k, v in costs.items()))
    return "\n".join(lines)


def render_readme_block(reports: list[dict[str, Any]]) -> str:
    rows = [
        "| Date | Prompt | Model | Test acc | Test macro-F1 | Sentiment acc (all) | Judge |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in reports:
        test, everything = r["splits"]["test"], r["splits"]["all"]
        judge = f"{r['judge']['mean_score']:.2f}/5" if r.get("judge") else "-"
        rows.append(
            f"| {r['created_at'][:10]} | `{r['prompt_version']}` | {r['model']} "
            f"| {test['primary_accuracy']:.2f} (n={test['n']}) | {test['macro_f1']:.2f} "
            f"| {everything['sentiment_accuracy']:.2f} | {judge} |"
        )
    return "\n".join([README_START, *rows, README_END])


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _: block, text)
    else:
        text = text.rstrip() + "\n\n## Eval results\n\n" + block + "\n"
    readme.write_text(text, encoding="utf-8")
