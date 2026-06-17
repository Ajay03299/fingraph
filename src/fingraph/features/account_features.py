"""Per-account feature engineering.

Each account is summarised as a single feature row that blends three views:

  volume     - how much money moves through the account
  velocity   - how fast and how concentrated that movement is in time
  structure  - the account's shape in the graph (fan-in/out, cycles, paths)

The structural features are the point of the whole project: structuring shows
up as extreme fan-in, layering as long directed paths, and cycles as literal
cycles. None of these are visible one transaction at a time.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def compute_account_features(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Compute one feature row per account node.

    Returns a DataFrame indexed by account_id. Cycle membership is computed
    once for the whole graph rather than per-node, since finding cycles is the
    expensive part and we only need a membership lookup afterwards.
    """
    cycle_members = _accounts_in_cycles(graph)
    rows = []

    for node in graph.nodes:
        in_amounts = [d["amount"] for _, _, d in graph.in_edges(node, data=True)]
        out_amounts = [d["amount"] for _, _, d in graph.out_edges(node, data=True)]
        in_times = [d["timestamp"] for _, _, d in graph.in_edges(node, data=True)]
        out_times = [d["timestamp"] for _, _, d in graph.out_edges(node, data=True)]

        in_degree = len(in_amounts)
        out_degree = len(out_amounts)
        total_in = float(np.sum(in_amounts)) if in_amounts else 0.0
        total_out = float(np.sum(out_amounts)) if out_amounts else 0.0

        # Distinct counterparties — a structuring collector receives from many
        # different senders, which is more telling than raw edge count.
        unique_senders = len({u for u, _, _ in graph.in_edges(node, data=True)})
        unique_receivers = len({v for _, v, _ in graph.out_edges(node, data=True)})

        rows.append(
            {
                "account_id": node,
                # volume
                "total_in": round(total_in, 2),
                "total_out": round(total_out, 2),
                "net_flow": round(total_in - total_out, 2),
                "avg_in_amount": round(float(np.mean(in_amounts)), 2) if in_amounts else 0.0,
                "avg_out_amount": round(float(np.mean(out_amounts)), 2) if out_amounts else 0.0,
                # structure
                "in_degree": in_degree,
                "out_degree": out_degree,
                "unique_senders": unique_senders,
                "unique_receivers": unique_receivers,
                # fan-in ratio: high when many distinct senders feed one account
                "fan_in_ratio": (
                    round(unique_senders / out_degree, 3) if out_degree else float(unique_senders)
                ),
                # velocity: transactions packed into a short window look automated
                "in_burstiness": _burstiness(in_times),
                "out_burstiness": _burstiness(out_times),
                # cycle membership: a strong, direct laundering signal
                "in_cycle": int(node in cycle_members),
            }
        )

    return pd.DataFrame(rows).set_index("account_id")


def _burstiness(timestamps: list) -> float:
    """How concentrated a set of timestamps is in time, in [0, 1].

    Returns the share of activity packed into the busiest 24-hour window.
    A value near 1 means almost everything happened in a single day, which is
    the temporal fingerprint of automated, coordinated movement.
    """
    if len(timestamps) < 2:
        return 0.0
    times = pd.to_datetime(pd.Series(timestamps)).sort_values()
    span_hours = (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600
    if span_hours <= 24:
        return 1.0
    # Slide a 24h window and find the densest stretch.
    counts = [(times >= t) & (times < t + pd.Timedelta(hours=24)) for t in times]
    busiest = max(int(c.sum()) for c in counts)
    return round(busiest / len(times), 3)


def _accounts_in_cycles(graph: nx.MultiDiGraph) -> set[str]:
    """Return the set of accounts that sit on at least one directed cycle.

    We collapse the multigraph to a simple DiGraph first: cycle membership
    doesn't care how many times two accounts transacted, only whether the
    directed loop exists. simple_cycles can be expensive on dense graphs, so
    we cap the cycle length to the short loops that actual laundering uses.
    """
    simple = nx.DiGraph(graph)
    members: set[str] = set()
    for cycle in nx.simple_cycles(simple, length_bound=6):
        members.update(cycle)
    return members
