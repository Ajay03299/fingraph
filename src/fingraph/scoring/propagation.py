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
from scipy import sparse


def propagate_risk(
    graph: nx.MultiDiGraph,
    seed_scores: pd.Series,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Diffuse seed risk scores across the transaction graph.

    Risk flows along money: each step, an account keeps a fraction of its own
    seed and absorbs a share of its neighbours' current risk, weighted by flow.
    Built on a sparse transition matrix so it scales to graphs with hundreds of
    thousands of accounts — a dense matrix would need O(n^2) memory and fall
    over on real-world data.

    Args:
        graph: the transaction multigraph.
        seed_scores: per-account anomaly scores in [0, 1] to seed from.
        damping: weight kept on an account's own seed each step (0.85 is the
            classic PageRank value and behaves well here).
        iterations: hard cap on diffusion steps.
        tolerance: stop early once scores stop moving meaningfully.

    Returns:
        Settled risk scores per account, normalised to [0, 1].
    """
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    seed = np.array([seed_scores.get(node, 0.0) for node in nodes], dtype=float)

    # Accumulate money flow as sparse coordinates. We treat influence as
    # undirected — laundering risk is guilt by association in both directions.
    rows, cols, data = [], [], []
    for u, v, attrs in graph.edges(data=True):
        i, j = index[u], index[v]
        amount = attrs["amount"]
        rows += [i, j]
        cols += [j, i]
        data += [amount, amount]

    weights = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Row-normalise so each account's incoming influence sums to 1.
    row_sums = np.asarray(weights.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # isolated accounts keep only their own seed
    inv = sparse.diags(1.0 / row_sums)
    transition = inv @ weights

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
