"""Render reason codes into a readable investigator case note.

The note is deliberately short — a few sentences an analyst can scan in seconds
and drop straight into a case file. It's built from the reason codes by simple
templating, so every sentence is traceable to concrete evidence and the output
is fully reproducible. No model writes this prose; that keeps it auditable.
"""

from __future__ import annotations

from fingraph.explain.reasons import Reason

_RISK_BAND = [
    (0.85, "critical"),
    (0.70, "high"),
    (0.50, "elevated"),
    (0.0, "low"),
]


def risk_band(score: float) -> str:
    for floor, label in _RISK_BAND:
        if score >= floor:
            return label
    return "low"


def write_case_note(account_id: str, risk_score: float, reasons: list[Reason]) -> str:
    """Compose a concise case note for one flagged account."""
    band = risk_band(risk_score)

    if not reasons:
        return (
            f"Account {account_id} carries a {band} risk score of {risk_score:.2f}, "
            "but no single structural pattern stands out. Recommend routine monitoring."
        )

    lead = f"Account {account_id} is flagged at {band} risk ({risk_score:.2f}). "

    # The headline reason gets a full sentence; the rest fold into a supporting clause.
    headline = reasons[0].detail
    body = headline if headline.endswith(".") else headline + "."
    sentences = [lead + body]

    if len(reasons) > 1:
        supporting = " ".join(r.detail for r in reasons[1:])
        sentences.append("Additional indicators: " + supporting)

    high_count = sum(1 for r in reasons if r.severity == "high")
    if high_count >= 2:
        sentences.append(
            "Multiple high-severity patterns coincide on this account, which warrants "
            "priority review."
        )
    else:
        sentences.append("Recommend analyst review and a check of linked accounts.")

    return " ".join(sentences)
