"""Tests for the anomaly detector and its evaluation."""

from fingraph.data.generator import GeneratorConfig, generate_dataset
from fingraph.detection.evaluate import account_labels, evaluate, precision_at_k
from fingraph.detection.model import AnomalyDetector
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph


def _setup(seed: int = 3):
    accounts, tx = generate_dataset(
        GeneratorConfig(num_accounts=120, num_legit_transactions=800, seed=seed)
    )
    graph = build_transaction_graph(tx, accounts)
    features = compute_account_features(graph)
    return graph, features


def test_scores_are_in_unit_interval():
    graph, features = _setup()
    scores = AnomalyDetector().fit_score(features)
    assert scores.between(0, 1).all()


def test_one_score_per_account():
    graph, features = _setup()
    scores = AnomalyDetector().fit_score(features)
    assert len(scores) == len(features)
    assert scores.index.equals(features.index)


def test_account_labels_are_binary():
    graph, _ = _setup()
    labels = account_labels(graph)
    assert set(labels.unique()).issubset({0, 1})


def test_detector_beats_random():
    # A real signal should clear 0.5 AUC comfortably. Loose threshold to stay
    # robust across seeds while still proving the model learns something.
    graph, features = _setup()
    scores = AnomalyDetector().fit_score(features)
    labels = account_labels(graph)
    metrics = evaluate(scores, labels)
    assert metrics["roc_auc"] > 0.6


def test_precision_at_k_bounds():
    graph, features = _setup()
    scores = AnomalyDetector().fit_score(features)
    labels = account_labels(graph)
    assert 0.0 <= precision_at_k(scores, labels, k=10) <= 1.0
