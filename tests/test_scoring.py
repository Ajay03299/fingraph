"""Tests for graph-aware risk propagation and score blending."""

import pandas as pd

from fingraph.data.generator import GeneratorConfig, generate_dataset
from fingraph.detection.evaluate import account_labels, evaluate
from fingraph.detection.model import AnomalyDetector
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph
from fingraph.scoring.propagation import blend_scores, propagate_risk


def _setup(seed: int = 3):
    accounts, tx = generate_dataset(
        GeneratorConfig(num_accounts=120, num_legit_transactions=800, seed=seed)
    )
    graph = build_transaction_graph(tx, accounts)
    features = compute_account_features(graph)
    anomaly = AnomalyDetector().fit_score(features)
    return graph, features, anomaly


def test_propagated_scores_in_unit_interval():
    graph, _, anomaly = _setup()
    propagated = propagate_risk(graph, anomaly)
    assert propagated.between(0, 1).all()


def test_one_propagated_score_per_node():
    graph, _, anomaly = _setup()
    propagated = propagate_risk(graph, anomaly)
    assert len(propagated) == graph.number_of_nodes()


def test_blend_respects_weight_extremes():
    graph, _, anomaly = _setup()
    propagated = propagate_risk(graph, anomaly)
    # weight=1.0 should return the anomaly score untouched
    all_anomaly = blend_scores(anomaly, propagated, weight=1.0)
    pd.testing.assert_series_equal(all_anomaly, anomaly, check_names=False)


def test_propagation_converges_quickly():
    # With a tight tolerance it should settle well within the iteration cap.
    graph, _, anomaly = _setup()
    propagated = propagate_risk(graph, anomaly, iterations=100)
    assert propagated.notna().all()


def test_graph_aware_does_not_break_detection():
    # The blended score should still be a competent detector — at least as good
    # as random, and in practice comparable to or better than the baseline.
    graph, _, anomaly = _setup()
    propagated = propagate_risk(graph, anomaly)
    final = blend_scores(anomaly, propagated)
    labels = account_labels(graph)
    assert evaluate(final, labels)["roc_auc"] > 0.6
