"""Investigate the top-ranked accounts end to end and print their case notes.

python -m fingraph.explain.run_investigation
"""

from __future__ import annotations

from fingraph.data.generator import generate_dataset
from fingraph.detection.model import AnomalyDetector
from fingraph.explain.investigate import investigate
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph
from fingraph.scoring.propagation import blend_scores, propagate_risk


def main(top_n: int = 5) -> None:
    accounts, transactions = generate_dataset()
    graph = build_transaction_graph(transactions, accounts)
    features = compute_account_features(graph)

    anomaly = AnomalyDetector().fit_score(features)
    propagated = propagate_risk(graph, anomaly)
    final = blend_scores(anomaly, propagated)

    print("FinGraph investigation — top flagged accounts\n")
    for account_id in final.sort_values(ascending=False).head(top_n).index:
        case = investigate(account_id, features, graph, final)
        print(f"=== {case.account_id}  [{case.risk_band.upper()}  {case.risk_score:.2f}] ===")
        print(case.case_note)
        print()


if __name__ == "__main__":
    main()
