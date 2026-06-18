"""Reason codes: the evidence behind a flag.

A risk score on its own tells an investigator nothing they can act on. This
module inspects a flagged account against the wider population and emits typed
pieces of evidence — extreme fan-in, cycle membership, suspicious counterparties,
bursty timing — each with a severity and a plain phrase. Thresholds are relative
to the population (percentiles), so "extreme" means extreme for this dataset
rather than against an arbitrary hard-coded number.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class Reason:
    """One piece of evidence supporting a flag."""

    code: str
    severity: str  # "high" | "medium" | "low"
    detail: str


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Where a value sits within a distribution, in [0, 1]."""
    return float((series <= value).mean())


def explain_account(
    account_id: str,
    features: pd.DataFrame,
    graph: nx.MultiDiGraph,
    risk_scores: pd.Series,
    high_risk_threshold: float = 0.7,
) -> list[Reason]:
    """Collect the reasons an account looks suspicious.

    Returns an ordered list of Reason objects, most severe first. An empty list
    means nothing notable stood out — the account looks ordinary.
    """
    row = features.loc[account_id]
    reasons: list[Reason] = []

    # Sits on a money cycle — among the strongest structural laundering signals.
    if row.get("in_cycle", 0) == 1:
        reasons.append(
            Reason(
                code="ON_CYCLE",
                severity="high",
                detail="Account sits on a closed transaction cycle, where funds "
                "return to their origin.",
            )
        )

    # Extreme fan-in relative to the population — the shape of a structuring collector.
    fan_in_pct = _percentile_rank(features["unique_senders"], row["unique_senders"])
    if fan_in_pct >= 0.97 and row["unique_senders"] >= 5:
        reasons.append(
            Reason(
                code="HIGH_FAN_IN",
                severity="high",
                detail=f"Receives from {int(row['unique_senders'])} distinct senders, "
                f"in the top {round((1 - fan_in_pct) * 100, 1)}% of the population — "
                "consistent with funnelling many small deposits into one account.",
            )
        )

    # Suspicious company: how many of its counterparties are themselves high-risk.
    dirty_neighbours = _high_risk_neighbours(account_id, graph, risk_scores, high_risk_threshold)
    if dirty_neighbours:
        severity = "high" if len(dirty_neighbours) >= 3 else "medium"
        reasons.append(
            Reason(
                code="RISKY_COUNTERPARTIES",
                severity=severity,
                detail=f"Transacts with {len(dirty_neighbours)} already high-risk "
                "account(s), so risk propagates through these relationships.",
            )
        )

    # Bursty timing — activity packed into a short window suggests automation.
    if row.get("in_burstiness", 0) >= 0.8 and row["in_degree"] >= 5:
        reasons.append(
            Reason(
                code="BURST_ACTIVITY",
                severity="medium",
                detail="Incoming transactions are tightly clustered in time, a pattern "
                "typical of coordinated or automated movement.",
            )
        )

    # Large net flow relative to the population — a major accumulation point.
    net_pct = _percentile_rank(features["net_flow"], row["net_flow"])
    if net_pct >= 0.98 and row["net_flow"] > 0:
        reasons.append(
            Reason(
                code="LARGE_INFLOW",
                severity="low",
                detail=f"Net inflow of {row['net_flow']:,.0f} is among the largest in "
                "the population.",
            )
        )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    reasons.sort(key=lambda r: severity_order[r.severity])
    return reasons


def _high_risk_neighbours(
    account_id: str,
    graph: nx.MultiDiGraph,
    risk_scores: pd.Series,
    threshold: float,
) -> list[str]:
    """Neighbours (either direction) whose own risk score clears the threshold."""
    neighbours = set(graph.predecessors(account_id)) | set(graph.successors(account_id))
    neighbours.discard(account_id)
    return [n for n in neighbours if risk_scores.get(n, 0.0) >= threshold]
