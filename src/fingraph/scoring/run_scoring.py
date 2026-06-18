"""End-to-end graph-aware scoring run, compared against the baseline.

Runs the full pipeline, then scores accounts two ways — raw anomaly score
versus the propagation-blended score — and prints both metric sets side by
side so the lift from going graph-aware is visible at a glance.

    python -m fingraph.scoring.run_scoring
"""

from __future__ import annotations

from fingraph.data.generator import generate_dataset
from fingraph.detection.evaluate import account_labels, evaluate
from fingraph.detection.model import AnomalyDetector
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph
from fingraph.scoring.propagation import blend_scores, propagate_risk


def main() -> None:
    accounts, transactions = generate_dataset()
    graph = build_transaction_graph(transactions, accounts)
    features = compute_account_features(graph)
    labels = account_labels(graph)

    anomaly = AnomalyDetector().fit_score(features)
    propagated = propagate_risk(graph, anomaly)
    final = blend_scores(anomaly, propagated)

    baseline_metrics = evaluate(anomaly, labels, k=20)
    graph_metrics = evaluate(final, labels, k=20)

    print("FinGraph: baseline vs graph-aware scoring")
    print("-" * 52)
    print(f"{'metric':>22} | {'baseline':>10} | {'graph-aware':>12}")
    print("-" * 52)
    for key in ("roc_auc", "avg_precision", "precision_at_20"):
        print(f"{key:>22} | {baseline_metrics[key]:>10} | {graph_metrics[key]:>12}")


if __name__ == "__main__":
    main()
