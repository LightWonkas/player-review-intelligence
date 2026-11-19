"""Server-side SVG charts, so the dashboard needs no JavaScript.

Specs follow the dataviz rules used across the project: bars at most 24px thick
with a 4px rounded data end and a square baseline, a 2px surface gap between
stacked segments, hairline recessive axes, selective labels, a native tooltip
(<title>) on every mark, and colors by role from CSS custom properties (so dark
mode is a stylesheet change). Every chart has a table view in the template.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from review_intel.insights import CategoryCount, TrendPoint

BAR = 16  # bar thickness (<= 24)
RADIUS = 4
GAP = 2  # surface gap between stacked segments


def _num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _bar_right(x: float, y: float, width: float, thickness: float) -> str:
    """Horizontal bar: square at the baseline (left), 4px rounded data end (right)."""
    if width <= RADIUS:
        return f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(width)}" height="{thickness}"/>'
    r = RADIUS
    return (
        f'<path d="M{_num(x)},{_num(y)} h{_num(width - r)} a{r},{r} 0 0 1 {r},{r} '
        f'v{_num(thickness - 2 * r)} a{r},{r} 0 0 1 -{r},{r} h-{_num(width - r)} z"/>'
    )


def _column_top(x: float, y: float, thickness: float, height: float, rounded: bool) -> str:
    """Vertical segment; the top one gets the 4px rounded data end."""
    if not rounded or height <= RADIUS:
        return f'<rect x="{_num(x)}" y="{_num(y)}" width="{thickness}" height="{_num(height)}"/>'
    r = RADIUS
    return (
        f'<path d="M{_num(x)},{_num(y + height)} v-{_num(height - r)} a{r},{r} 0 0 1 {r},-{r} '
        f'h{_num(thickness - 2 * r)} a{r},{r} 0 0 1 {r},{r} v{_num(height - r)} z"/>'
    )


def nice_max(value: int) -> tuple[int, int]:
    """A clean axis maximum and tick step (1, 2 or 5 times a power of ten), ~4 ticks."""
    if value <= 0:
        return 1, 1
    raw = value / 4
    power = 10 ** math.floor(math.log10(raw)) if raw >= 1 else 1
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    step = max(1, int(step))
    return step * math.ceil(value / step), step


def category_bars(
    categories: Sequence[CategoryCount], labels: dict[str, str], width: int = 460
) -> str:
    """One series (reviews per primary category), so no legend: the heading names it."""
    if not categories:
        return ""
    label_w, value_w, row = 160, 64, 26
    plot_w = width - label_w - value_w
    height = row * len(categories) + 8
    top = max(c.reviews for c in categories)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Reviews by category">',
        f'<line class="axis" x1="{label_w}" y1="0" x2="{label_w}" y2="{height - 4}"/>',
    ]
    for i, item in enumerate(categories):
        y = i * row + 4
        bar_y = y + (row - BAR) / 2
        bar_w = plot_w * item.reviews / top
        name = labels.get(item.category, item.category)
        share = f"{item.share * 100:.0f}%"
        parts += [
            f'<g class="mark"><title>{escape(name)}: {item.reviews} reviews ({share})</title>',
            f'<rect class="hit" x="0" y="{y}" width="{width}" height="{row}"/>',
            f'<text class="label" x="{label_w - 10}" y="{y + row / 2}" '
            f'text-anchor="end" dominant-baseline="middle">{escape(name)}</text>',
            f'<g class="series-1">{_bar_right(label_w, bar_y, max(bar_w, 1), BAR)}</g>',
            f'<text class="value" x="{_num(label_w + bar_w + 8)}" y="{y + row / 2}" '
            f'dominant-baseline="middle">{item.reviews} · {share}</text>',
            "</g>",
        ]
    parts.append("</svg>")
    return "".join(parts)


@dataclass(frozen=True)
class Segment:
    key: str  # css class: negative / neutral / positive
    label: str


SENTIMENT_SEGMENTS = (  # bottom to top: the negative arm sits on the baseline
    Segment("negative", "Negative"),
    Segment("neutral", "Neutral"),
    Segment("positive", "Positive"),
)


def sentiment_columns(points: Sequence[TrendPoint], width: int = 460, height: int = 220) -> str:
    """Stacked columns per bucket. Three series, so the template shows a legend."""
    if not points:
        return ""
    left, bottom, top_pad = 36, 28, 20
    plot_h = height - bottom - top_pad
    plot_w = width - left - 8
    axis_max, step = nice_max(max(p.reviews for p in points))
    slot = plot_w / len(points)
    thickness = min(24.0, slot * 0.6)
    scale = plot_h / axis_max
    base_y = top_pad + plot_h
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Review sentiment per period">'
    ]
    for tick in range(0, axis_max + 1, step):
        y = base_y - tick * scale
        css = "axis" if tick == 0 else "grid"
        parts += [
            f'<line class="{css}" x1="{left}" y1="{_num(y)}" x2="{width - 8}" y2="{_num(y)}"/>',
            f'<text class="tick" x="{left - 8}" y="{_num(y)}" text-anchor="end" '
            f'dominant-baseline="middle">{tick:,}</text>',
        ]
    for i, point in enumerate(points):
        x = left + slot * i + (slot - thickness) / 2
        counts = {"negative": point.negative, "neutral": point.neutral, "positive": point.positive}
        present = [s for s in SENTIMENT_SEGMENTS if counts[s.key] > 0]
        y = base_y
        for j, segment in enumerate(present):
            h = counts[segment.key] * scale
            is_top = j == len(present) - 1
            drawn = (
                h if is_top else max(h - GAP, 0.5)
            )  # the gap is surface, taken from the lower one
            y -= h
            tip = (
                f"Week of {point.start}: {counts[segment.key]} {segment.label.lower()} "
                f"of {point.reviews}"
            )
            parts.append(
                f'<g class="mark {segment.key}"><title>{escape(tip)}</title>'
                f"{_column_top(x, y + (h - drawn), thickness, drawn, rounded=is_top)}</g>"
            )
        # "Nov 16" needs ~40px; when slots are narrower, label every other one,
        # keeping the latest so the current week is always named.
        every = 1 if slot >= 40 else 2
        if (len(points) - 1 - i) % every == 0:
            parts.append(
                f'<text class="tick" x="{_num(x + thickness / 2)}" y="{base_y + 18}" '
                f'text-anchor="middle">{point.start:%b %d}</text>'
            )
    last = points[-1]
    if last.reviews:
        x = left + slot * (len(points) - 1) + slot / 2
        parts.append(
            f'<text class="value" x="{_num(x)}" y="{_num(base_y - last.reviews * scale - 6)}" '
            f'text-anchor="middle">{last.reviews}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
