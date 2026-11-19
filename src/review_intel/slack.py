"""Optional: post a short version of the report to a Slack incoming webhook.

Slack's mrkdwn has no tables, so the message carries the summary, the issue
headlines and the action items; the full report stays in the file and the dashboard.
The webhook URL is a secret and only comes from `SLACK_WEBHOOK_URL`.
"""

import httpx

from review_intel.report import STORE_LABELS, ReportRun, _label, _pct


def slack_text(run: ReportRun) -> str:
    facts, prose = run.facts, run.prose
    start = facts.period_start.date()
    end = facts.period_end.date()
    lines = [
        f"*Player review report: {facts.game_name}* ({start} to {end})",
        prose.summary,
        "",
        "*Top issues*",
    ]
    for issue in facts.issues.issues:
        lines.append(
            f"- {_label(issue.category)}: {issue.reviews} reviews ({_pct(issue.share)}), "
            f"previous period {issue.previous_reviews}"
        )
    lines += ["", "*Action items*"]
    for item in sorted(prose.action_items, key=lambda a: a.priority):
        lines.append(f"- [{item.priority}, {item.team}] {item.action}")
    if facts.spikes and facts.spikes_significant:
        spike = facts.spikes[0]
        lines += [
            "",
            f"*Version watch*: {STORE_LABELS[spike.store]} v{spike.version} "
            f"{_label(spike.category)} {_pct(spike.before_share)} -> {_pct(spike.after_share)}",
        ]
    return "\n".join(lines)


def post_to_slack(webhook_url: str, text: str, *, client: httpx.Client | None = None) -> None:
    """Raises httpx.HTTPStatusError if Slack rejects the message."""
    http = client or httpx.Client(timeout=15.0)
    try:
        response = http.post(webhook_url, json={"text": text})
        response.raise_for_status()
    finally:
        if client is None:
            http.close()
