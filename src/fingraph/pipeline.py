"""The FinGraph analysis pipeline.

Wires the stages together — data, graph, features, anomaly scoring, graph-aware
propagation, blended risk — and holds the result in memory so the API and
dashboard can serve many requests without recomputing everything each time.

Build it once, then query it: the ranked alert queue, a single account's
investigation, or the underlying graph for visualisation.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from fingraph.data.generator import GeneratorConfig
from fingraph.data.sources import DataSource, SyntheticSource
from fingraph.detection.evaluate import account_labels, evaluate
from fingraph.detection.model import AnomalyDetector
from fingraph.explain.investigate import Investigation, investigate
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph
from fingraph.scoring.propagation import blend_scores, propagate_risk


class Pipeline:
    """Runs the full analysis and serves results.

    A single build() call does all the heavy lifting; after that, lookups are
    cheap reads against the cached scores and graph.
    """

    def __init__(
        self,
        source: DataSource | GeneratorConfig | None = None,
        config: GeneratorConfig | None = None,
    ):
        # Backwards compatibility: older callers do Pipeline(config) positionally.
        # If the first argument is a GeneratorConfig, treat it as the config.
        if isinstance(source, GeneratorConfig):
            config = source
            source = None
        self.source = source or SyntheticSource(config)
        self.config = config
        self.source = source or SyntheticSource(config)
        self.config = config
        self.accounts: pd.DataFrame | None = None
        self.transactions: pd.DataFrame | None = None
        self.graph: nx.MultiDiGraph | None = None
        self.features: pd.DataFrame | None = None
        self.risk_scores: pd.Series | None = None
        self.labels: pd.Series | None = None

    def build(self) -> Pipeline:
        self.accounts, self.transactions = self.source.load()
        self.graph = build_transaction_graph(self.transactions, self.accounts)
        self.features = compute_account_features(self.graph)

        anomaly = AnomalyDetector().fit_score(self.features)
        propagated = propagate_risk(self.graph, anomaly)
        self.risk_scores = blend_scores(anomaly, propagated)
        self.labels = account_labels(self.graph)
        return self

    # --- queries -----------------------------------------------------------

    def _require_built(self) -> None:
        if self.risk_scores is None:
            raise RuntimeError("Pipeline not built yet — call build() first.")

    def alerts(self, top_n: int = 25) -> pd.DataFrame:
        """The ranked alert queue: highest-risk accounts first."""
        self._require_built()
        ranked = self.risk_scores.sort_values(ascending=False).head(top_n)
        return pd.DataFrame(
            {
                "account_id": ranked.index,
                "risk_score": ranked.values.round(4),
                "is_dirty": self.labels.reindex(ranked.index).fillna(0).astype(int).values,
            }
        ).reset_index(drop=True)

    def investigation(self, account_id: str) -> Investigation:
        """Full investigation record for one account."""
        self._require_built()
        return investigate(account_id, self.features, self.graph, self.risk_scores)

    def metrics(self, k: int = 20) -> dict[str, float]:
        """Detection metrics against the planted ground truth."""
        self._require_built()
        return evaluate(self.risk_scores, self.labels, k=k)

    def ego_graph(self, account_id: str, radius: int = 1) -> nx.MultiDiGraph:
        """The local neighbourhood around an account, for visualisation."""
        self._require_built()
        undirected = self.graph.to_undirected(as_view=True)
        nodes = nx.ego_graph(undirected, account_id, radius=radius).nodes
        return self.graph.subgraph(nodes).copy()
