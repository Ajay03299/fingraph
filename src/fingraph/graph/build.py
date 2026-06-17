"""Build a transaction graph from raw transaction data.

Accounts become nodes, transactions become directed edges. We use a
MultiDiGraph because direction matters (A paying B is not the same as the
reverse) and two accounts can transact more than once, so parallel edges
need to be allowed rather than collapsed.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_transaction_graph(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """Construct a directed multigraph from transactions.

    Every account that appears as a source or destination becomes a node.
    If an accounts frame is supplied, its metadata is attached to the nodes;
    otherwise nodes are created bare from the transactions alone.
    """
    graph = nx.MultiDiGraph()

    if accounts is not None:
        for row in accounts.itertuples(index=False):
            graph.add_node(
                row.account_id,
                account_type=row.account_type,
                country=row.country,
                opened_at=row.opened_at,
            )

    for tx in transactions.itertuples(index=False):
        # add_edge silently creates either endpoint if it doesn't exist yet,
        # so accounts missing from the metadata frame still end up as nodes.
        graph.add_edge(
            tx.source,
            tx.destination,
            key=tx.tx_id,
            amount=tx.amount,
            timestamp=tx.timestamp,
            channel=tx.channel,
            pattern=tx.pattern,
            is_laundering=tx.is_laundering,
        )

    return graph
