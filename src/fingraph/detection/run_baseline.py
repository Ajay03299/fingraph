"""End-to-end baseline run: data -> graph -> features -> scores -> metrics.

A single command that exercises the whole pipeline built so far and prints how
well the unsupervised detector recovers the planted laundering accounts.

    python -m fingraph.detection.run_baseline
"""

from __future__ import annotations

from fingraph.data.generator import generate_dataset
from fingraph.detection.evaluate import account_labels, evaluate
from fingraph.detection.model import AnomalyDetector
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph


def main() -> None:
    accounts, transactions = generate_dataset()
    graph = build_transaction_graph(transactions, accounts)
    features = compute_account_features(graph)

    detector = AnomalyDetector()
    scores = detector.fit_score(features)
    labels = account_labels(graph)

    metrics = evaluate(scores, labels, k=20)

    print("FinGraph baseline anomaly detection")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name:>22}: {value}")

    print("\nTop 5 highest-risk accounts:")
    ranked = scores.sort_values(ascending=False).head(5)
    for account_id, score in ranked.items():
        flag = "DIRTY" if labels.get(account_id, 0) else "clean"
        print(f"  {account_id}  risk={score:.3f}  [{flag}]")


if __name__ == "__main__":
    main()
