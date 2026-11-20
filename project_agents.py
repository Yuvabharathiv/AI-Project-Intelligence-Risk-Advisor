"""Evidence-based project scope, risk, and action extraction."""

import re
from typing import Any


_DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)
_OWNER_PATTERN = re.compile(
    r"\b(?:owner|assignee|assigned to|responsible(?: person)?)\s*[:=-]\s*"
    r"([^,;|]+)",
    re.IGNORECASE,
)
_OWNER_TAIL_PATTERN = re.compile(
    r"\s+\b(?:due|by|target|deadline|status|for)\b.*$", re.IGNORECASE
)

_SCOPE_CATEGORIES = {
    "goal": (
        "project goal",
        "objective",
        "success criteria",
        "goal:",
        "aim:",
    ),
    "deliverable": ("deliverable", "output:", "will deliver", "will provide"),
    "milestone": ("milestone", "phase ", "release", "launch", "go-live"),
    "responsibility": (
        "owner:",
        "assignee:",
        "assigned to",
        "responsible:",
    ),
    "timeline": ("deadline", "due:", "due date", "target date", "by "),
}

_RISK_SIGNALS = (
    ("high", ("overdue", "delayed", "behind schedule",
              "at risk", "slipping", "missed deadline")),
    ("medium", ("dependency gap", "depends on", "dependent on",
                "awaiting approval", "unresolved",
                "no owner", "owner not assigned")),
)
_BLOCKER_LABEL_PATTERN = re.compile(r"\bblockers?\s*[:=-]\s*\S", re.IGNORECASE)
_BLOCKED_STATUS_PATTERN = re.compile(
    r"\b(?:is|are|was|were|remains?|remain|still|currently)\s+blocked\b|"
    r"\bblocked\s+(?:by|until|because)\b",
    re.IGNORECASE,
)
_RISK_LABEL_PATTERN = re.compile(r"^\s*risk\s*[:=-]\s*\S", re.IGNORECASE)
_PENDING_DECISION_LABEL_PATTERN = re.compile(
    r"^\s*pending decisions?\s*[:=-]\s*\S", re.IGNORECASE
)
_BLOCKER_SIGNALS = (
    "blocked",
    "blocker",
    "unresolved",
    "dependency gap",
    "stuck",
)
_DECISION_SIGNALS = (
    "pending decision",
    "decision needed",
    "needs a decision",
    "awaiting approval",
    "approval pending",
)
_ACTION_SIGNALS = (
    "action item",
    "next step",
    "follow up",
    "follow-up",
    "todo",
    "to-do",
    "assigned to",
    "must ",
    "needs to ",
    "need to ",
)


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line)
        if line and not re.fullmatch(r"[-|=\s]+", line):
            lines.append(line)
    return lines


def _owner_in(line: str) -> str | None:
    match = _OWNER_PATTERN.search(line)
    if not match:
        return None
    owner = _OWNER_TAIL_PATTERN.sub("", match.group(1)).strip(" .:")
    return owner or None


def _scope_category(line: str) -> list[str]:
    normalized = line.casefold()
    categories = [
        category
        for category, indicators in _SCOPE_CATEGORIES.items()
        if any(indicator in normalized for indicator in indicators)
    ]
    if _DATE_PATTERN.search(line) and "timeline" not in categories:
        categories.append("timeline")
    if _owner_in(line) and "responsibility" not in categories:
        categories.append("responsibility")
    return categories


def extract_scope_and_deliverables(text: str, source: str) -> dict[str, Any]:
    """Extract project goals, deliverables, milestones, timelines, and owners."""
    findings = []
    for line in _clean_lines(text):
        categories = _scope_category(line)
        if categories:
            date_match = _DATE_PATTERN.search(line)
            findings.append(
                {
                    "categories": categories,
                    "text": line,
                    "owner": _owner_in(line),
                    "timeline": date_match.group(0) if date_match else None,
                    "source": source,
                }
            )
    return {"findings": findings}


def detect_delivery_risks(text: str, source: str) -> dict[str, Any]:
    """Identify explicit schedule and dependency risk signals in source text."""
    risks = []
    for line in _clean_lines(text):
        normalized = line.casefold()
        for severity, indicators in _RISK_SIGNALS:
            matched = [indicator for indicator in indicators if indicator in normalized]
            if severity == "high":
                if _BLOCKER_LABEL_PATTERN.search(line):
                    matched.append("blocker")
                if _BLOCKED_STATUS_PATTERN.search(line):
                    matched.append("blocked")
            elif _RISK_LABEL_PATTERN.search(line):
                matched.append("risk")
            elif _PENDING_DECISION_LABEL_PATTERN.search(line):
                matched.append("pending decision")
            if matched:
                risks.append(
                    {
                        "severity": severity,
                        "signals": matched,
                        "evidence": line,
                        "source": source,
                    }
                )
                break

    if any(risk["severity"] == "high" for risk in risks):
        forecast = "High delivery risk"
    elif risks:
        forecast = "Watch: potential delivery risk"
    else:
        forecast = "No explicit delivery risk found"

    return {
        "forecast": forecast,
        "risks": risks,
        "limitation": (
            "This signal is based only on explicit wording in the documents; "
            "it is not a date estimate or a guarantee of delivery."
        ),
    }


def identify_blockers_and_actions(text: str, source: str) -> dict[str, Any]:
    """Extract pending decisions, unresolved issues, and action items."""
    blockers = []
    decisions = []
    actions = []

    for line in _clean_lines(text):
        normalized = line.casefold()
        if (
            _BLOCKER_LABEL_PATTERN.search(line)
            or _BLOCKED_STATUS_PATTERN.search(line)
            or any(
                signal in normalized
                for signal in _BLOCKER_SIGNALS
                if signal not in {"blocked", "blocker"}
            )
        ):
            blockers.append({"text": line, "source": source})
        if any(signal in normalized for signal in _DECISION_SIGNALS):
            decisions.append({"text": line, "source": source})
        if any(signal in normalized for signal in _ACTION_SIGNALS):
            actions.append(
                {
                    "text": line,
                    "owner": _owner_in(line),
                    "due_date": (
                        _DATE_PATTERN.search(line).group(0)
                        if _DATE_PATTERN.search(line)
                        else None
                    ),
                    "source": source,
                }
            )

    return {"blockers": blockers, "pending_decisions": decisions, "actions": actions}


def analyze_project_document(text: str, source: str) -> dict[str, Any]:
    """Run all project analysis agents for one extracted document."""
    return {
        "scope": extract_scope_and_deliverables(text, source),
        "delivery_risk": detect_delivery_risks(text, source),
        "blockers_and_actions": identify_blockers_and_actions(text, source),
    }


def generate_project_documentation(
    analyses: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Create evidence-linked user stories, risks, and actions from analyses."""
    user_stories = []
    risks = []
    actions = []

    for analysis in analyses:
        for finding in analysis["scope"]["findings"]:
            if {"goal", "deliverable"} & set(finding["categories"]):
                user_stories.append(
                    {
                        "story": (
                            "As a project stakeholder, I want the documented "
                            f"outcome “{finding['text']}” to be delivered."
                        ),
                        "source": finding["source"],
                        "evidence": finding["text"],
                    }
                )

        for risk in analysis["delivery_risk"]["risks"]:
            risks.append(
                {
                    "severity": risk["severity"],
                    "description": risk["evidence"],
                    "source": risk["source"],
                    "evidence": risk["evidence"],
                }
            )

        actions.extend(analysis["blockers_and_actions"]["actions"])

    return {"user_stories": user_stories, "risk_register": risks, "actions": actions}


def calculate_project_health(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Score scope clarity, delivery risk, and blockers using transparent rules."""
    if not analyses:
        raise ValueError("At least one document analysis is required.")

    scope_categories = {
        category
        for analysis in analyses
        for finding in analysis["scope"]["findings"]
        for category in finding["categories"]
    }
    scope_dimensions = ("goal", "deliverable", "milestone", "timeline")
    scope_score = round(
        100
        * sum(category in scope_categories for category in scope_dimensions)
        / len(scope_dimensions)
    )

    risks = [
        risk
        for analysis in analyses
        for risk in analysis["delivery_risk"]["risks"]
    ]
    high_risks = sum(risk["severity"] == "high" for risk in risks)
    medium_risks = sum(risk["severity"] == "medium" for risk in risks)
    timeline_score = max(0, 100 - 30 * high_risks - 15 * medium_risks)

    blockers = [
        blocker
        for analysis in analyses
        for blocker in analysis["blockers_and_actions"]["blockers"]
    ]
    blocker_score = max(0, 100 - 25 * len(blockers))
    dimensions = {
        "scope_clarity": {
            "score": scope_score,
            "rationale": (
                f"{len(scope_categories.intersection(scope_dimensions))} of 4 "
                "scope and schedule signals are documented."
            ),
        },
        "timeline_risk": {
            "score": timeline_score,
            "rationale": (
                f"{high_risks} high and {medium_risks} medium explicit risk "
                "signals; each subtracts 30 and 15 points respectively."
            ),
        },
        "blockers": {
            "score": blocker_score,
            "rationale": (
                f"{len(blockers)} explicit blocker signal(s); each subtracts "
                "25 points."
            ),
        },
    }
    return {
        "overall_score": round(
            sum(dimension["score"] for dimension in dimensions.values())
            / len(dimensions)
        ),
        "dimensions": dimensions,
        "blocker_count": len(blockers),
        "risk_count": len(risks),
        "scoring_note": (
            "Scores are rule-based signals from uploaded text, not a prediction "
            "of project success. Scope starts at zero and earns 25 points for "
            "each documented goal, deliverable, milestone, or timeline."
        ),
    }
