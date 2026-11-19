"""Interactive, blind labeling: you never see the model's answer while labeling.

Showing the prediction would pull your labels toward it (anchoring) and make the
model look better than it is. Input and output are injected so tests can drive it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from review_intel.domain import Sentiment
from review_intel.evals.dataset import Candidate, LabeledReview, append_label, make_label
from review_intel.llm.taxonomy import Taxonomy

Ask = Callable[[str], str]
Say = Callable[[str], None]

SENTIMENT_KEYS = {"n": Sentiment.NEGATIVE, "u": Sentiment.NEUTRAL, "p": Sentiment.POSITIVE}


class Quit(Exception):
    pass


class Skip(Exception):
    pass


@dataclass
class LabelingResult:
    labeled: int
    skipped: int
    quit: bool


def _menu(taxonomy: Taxonomy) -> str:
    items = [f"{i:>2} {name}" for i, name in enumerate(taxonomy.names, start=1)]
    half = (len(items) + 1) // 2
    rows = [
        f"  {left:<24}{right}"
        for left, right in zip(items[:half], [*items[half:], ""], strict=False)
    ]
    return "\n".join(rows)


def _parse_category(answer: str, names: list[str]) -> str | None:
    answer = answer.strip().lower()
    if answer.isdigit() and 1 <= int(answer) <= len(names):
        return names[int(answer) - 1]
    return answer if answer in names else None


def _ask_until[T](ask: Ask, say: Say, prompt: str, parse: Callable[[str], T | None]) -> T:
    while True:
        answer = ask(prompt).strip()
        if answer.lower() == "q":
            raise Quit
        if answer.lower() == "s":
            raise Skip
        parsed = parse(answer)
        if parsed is not None:
            return parsed
        say("  Not understood, try again.")


def label_one(candidate: Candidate, taxonomy: Taxonomy, ask: Ask, say: Say) -> LabeledReview:
    names = taxonomy.names
    stars = "*" * candidate.rating + "." * (5 - candidate.rating)
    say(f"{candidate.store.value}  {stars}  v{candidate.app_version or '?'}")
    if candidate.title:
        say(f"Title: {candidate.title}")
    say(f"Body:  {candidate.body or '(no text)'}")
    say(_menu(taxonomy))

    primary = _ask_until(
        ask, say, "Primary category (number, s=skip, q=quit): ", lambda a: _parse_category(a, names)
    )

    def parse_extra(answer: str) -> list[str] | None:
        if not answer:
            return []
        parts = [_parse_category(part, names) for part in answer.split(",")]
        return None if any(p is None for p in parts) else [p for p in parts if p]

    extra = _ask_until(ask, say, "Other categories (e.g. 3,5; Enter for none): ", parse_extra)
    sentiment = _ask_until(
        ask,
        say,
        "Sentiment [n]egative / ne[u]tral / [p]ositive: ",
        lambda a: SENTIMENT_KEYS.get(a.lower()),
    )
    return make_label(candidate, primary, extra, sentiment)


def run_labeling(
    candidates: list[Candidate],
    taxonomy: Taxonomy,
    path: Path,
    *,
    already_labeled: int,
    target: int,
    ask: Ask,
    say: Say,
) -> LabelingResult:
    result = LabelingResult(labeled=0, skipped=0, quit=False)
    for candidate in candidates:
        done = already_labeled + result.labeled
        if done >= target:
            break
        say(f"\n[{done + 1}/{target}] " + "-" * 60)
        try:
            label = label_one(candidate, taxonomy, ask, say)
        except Skip:
            result.skipped += 1
            continue
        except Quit:
            result.quit = True
            break
        append_label(label, path)
        result.labeled += 1
    return result
