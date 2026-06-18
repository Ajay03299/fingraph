"""The investigation entry point.

Bundles everything the rest of the system needs to present a flagged account:
its risk score, the typed reason codes, and a ready-to-read case note. This is
the function the API and dashboard call, so it's the stable public surface of
the explainability layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx
import pandas as pd

from fingraph.explain.reasons import Reason, explain_account
from fingraph.explain.summary import risk_band, write_case_note


@dataclass
class Investigation:
    account_id: str
    risk_score: float
    risk_band: str
    reasons: list[Reason]
    case_note: str

    def to_dict(self) -> dict:
        """JSON-friendly form for the API."""
        return {
            "account_id": self.account_id,
            "risk_score": round(self.risk_score, 4),
            "risk_band": self.risk_band,
            "reasons": [asdict(r) for r in self.reasons],
            "case_note": self.case_note,
        }


def investigate(
    account_id: str,
    features: pd.DataFrame,
    graph: nx.MultiDiGraph,
    risk_scores: pd.Series,
) -> Investigation:
    """Produce a complete investigation record for a single account."""
    score = float(risk_scores.get(account_id, 0.0))
    reasons = explain_account(account_id, features, graph, risk_scores)
    note = write_case_note(account_id, score, reasons)
    return Investigation(
        account_id=account_id,
        risk_score=score,
        risk_band=risk_band(score),
        reasons=reasons,
        case_note=note,
    )
