"""Evaluation utilities for the anomaly detector.

Ground-truth account labels are derived from the edges: an account is dirty if
it took part in any planted laundering transaction. These labels are used only
to score the model, never to train it.

We avoid plain accuracy on purpose. With laundering accounts being rare, a model
that flags nobody scores high on accuracy and catches nothing. ROC-AUC, average
precision, and precision@k describe a rare-event detector honestly.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def account_labels(graph: nx.MultiDiGraph) -> pd.Series:
    """One label per account: 1 if it touched any laundering edge, else 0."""
    dirty: set[str] = set()
    for u, v, data in graph.edges(data=True):
        if data["is_laundering"]:
            dirty.add(u)
            dirty.add(v)
    labels = {node: int(node in dirty) for node in graph.nodes}
    return pd.Series(labels, name="is_dirty")


def precision_at_k(scores: pd.Series, labels: pd.Series, k: int) -> float:
    """Share of truly dirty accounts among the top-k highest-risk accounts.

    This is the metric an investigator feels directly: if I work the top k
    alerts, how many are real?
    """
    top_k = scores.sort_values(ascending=False).head(k).index
    aligned = labels.reindex(top_k).fillna(0)
    return round(float(aligned.mean()), 3)


def evaluate(scores: pd.Series, labels: pd.Series, k: int = 20) -> dict[str, float]:
    """Compute the headline detection metrics, aligning scores to labels first."""
    aligned_labels = labels.reindex(scores.index).fillna(0)
    return {
        "roc_auc": round(float(roc_auc_score(aligned_labels, scores)), 3),
        "avg_precision": round(float(average_precision_score(aligned_labels, scores)), 3),
        f"precision_at_{k}": precision_at_k(scores, labels, k),
        "num_dirty_accounts": int(aligned_labels.sum()),
        "num_accounts": int(len(aligned_labels)),
    }
