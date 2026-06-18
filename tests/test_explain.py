"""Tests for the explainability layer."""

from fingraph.data.generator import GeneratorConfig, generate_dataset
from fingraph.detection.model import AnomalyDetector
from fingraph.explain.investigate import investigate
from fingraph.explain.reasons import Reason, explain_account
from fingraph.explain.summary import risk_band, write_case_note
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
    final = blend_scores(anomaly, propagate_risk(graph, anomaly))
    return graph, features, final


def test_top_account_has_reasons():
    graph, features, scores = _setup()
    top = scores.idxmax()
    reasons = explain_account(top, features, graph, scores)
    assert len(reasons) > 0
    assert all(isinstance(r, Reason) for r in reasons)


def test_reasons_sorted_by_severity():
    graph, features, scores = _setup()
    top = scores.idxmax()
    reasons = explain_account(top, features, graph, scores)
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [order[r.severity] for r in reasons]
    assert ranks == sorted(ranks)


def test_risk_band_thresholds():
    assert risk_band(0.95) == "critical"
    assert risk_band(0.75) == "high"
    assert risk_band(0.55) == "elevated"
    assert risk_band(0.10) == "low"


def test_case_note_is_nonempty_string():
    graph, features, scores = _setup()
    top = scores.idxmax()
    reasons = explain_account(top, features, graph, scores)
    note = write_case_note(top, float(scores[top]), reasons)
    assert isinstance(note, str) and len(note) > 0


def test_investigation_serialises_to_dict():
    graph, features, scores = _setup()
    top = scores.idxmax()
    case = investigate(top, features, graph, scores)
    payload = case.to_dict()
    assert payload["account_id"] == top
    assert "case_note" in payload and "reasons" in payload
    assert isinstance(payload["reasons"], list)
