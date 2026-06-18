"""Tests for the analysis pipeline."""

import pytest
from fingraph.data.generator import GeneratorConfig
from fingraph.pipeline import Pipeline


def _pipeline() -> Pipeline:
    config = GeneratorConfig(num_accounts=100, num_legit_transactions=600, seed=3)
    return Pipeline(config).build()


def test_pipeline_builds_all_artifacts():
    p = _pipeline()
    assert p.graph is not None
    assert p.features is not None
    assert p.risk_scores is not None


def test_alerts_are_ranked_descending():
    alerts = _pipeline().alerts(top_n=10)
    scores = alerts["risk_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_investigation_returns_record_for_known_account():
    p = _pipeline()
    account_id = p.alerts(top_n=1)["account_id"].iloc[0]
    case = p.investigation(account_id)
    assert case.account_id == account_id


def test_query_before_build_raises():
    with pytest.raises(RuntimeError):
        Pipeline(GeneratorConfig(num_accounts=20, num_legit_transactions=50)).alerts()


def test_ego_graph_includes_focus_account():
    p = _pipeline()
    account_id = p.alerts(top_n=1)["account_id"].iloc[0]
    ego = p.ego_graph(account_id)
    assert account_id in ego
