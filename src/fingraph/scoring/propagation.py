"""Graph-aware risk scoring by propagation.

The anomaly detector scores each account on its own merits. But laundering is
a network phenomenon: risk should flow along the money. An account that
repeatedly transacts with suspicious accounts is itself more suspicious, even
when its own features look ordinary.

We model this as a damped diffusion, in the spirit of personalised PageRank.
Each account starts at its anomaly score (the seed). On every step it keeps a
fraction of its own seed and absorbs a share of its neighbours' current risk,
weighted by how much money flows between them. Damping anchors each account to
its own evidence so risk doesn't wash out into a uniform grey.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def propagate_risk(
    graph: nx.MultiDiGraph,
    seed_scores: pd.Series,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Diffuse seed risk scores across the transaction graph.

    Args:
        graph: the transaction multigraph.
        seed_scores: per-account anomaly scores in [0, 1] to seed from.
        damping: weight kept on an account's own seed each step. Higher means
            more self-reliant; lower lets neighbours pull harder. 0.85 is the
            classic PageRank value and behaves well here.
        iterations: hard cap on diffusion steps.
        tolerance: stop early once scores stop moving meaningfully.

    Returns:
        Settled risk scores per account, normalised to [0, 1].
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    seed = np.array([seed_scores.get(node, 0.0) for node in nodes], dtype=float)

    # Build a row-normalised neighbour-influence matrix from money flow. We
    # treat the graph as undirected for influence: laundering risk is guilt by
    # association in both directions, not just downstream.
    weights = np.zeros((n, n), dtype=float)
    for u, v, data in graph.edges(data=True):
        i, j = index[u], index[v]
        weights[i, j] += data["amount"]
        weights[j, i] += data["amount"]

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # isolated accounts keep only their own seed
    transition = weights / row_sums

    scores = seed.copy()
    for _ in range(iterations):
        updated = damping * seed + (1 - damping) * (transition @ scores)
        if np.abs(updated - scores).max() < tolerance:
            scores = updated
            break
        scores = updated

    spread = scores.max() - scores.min()
    normalised = (scores - scores.min()) / spread if spread else np.zeros_like(scores)
    return pd.Series(normalised, index=nodes, name="propagated_risk")


def blend_scores(
    anomaly: pd.Series,
    propagated: pd.Series,
    weight: float = 0.5,
) -> pd.Series:
    """Combine raw anomaly and propagated risk into one final score.

    Keeping both terms matters: the anomaly score preserves an account's own
    evidence, while propagation adds the context of who it deals with. A 50/50
    blend is a sensible, defensible default.
    """
    aligned = propagated.reindex(anomaly.index).fillna(0.0)
    blended = weight * anomaly + (1 - weight) * aligned
    return pd.Series(blended, index=anomaly.index, name="final_risk")
