"""Tests for the analysis pipeline."""

import pytest
from fingraph.data.generator import GeneratorConfig
from fingraph.pipeline import Pipeline


def test_pipeline_builds_all_artifacts(built_pipeline):
    assert built_pipeline.graph is not None
    assert built_pipeline.features is not None
    assert built_pipeline.risk_scores is not None


def test_alerts_are_ranked_descending(built_pipeline):
    alerts = built_pipeline.alerts(top_n=10)
    scores = alerts["risk_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_investigation_returns_record_for_known_account(built_pipeline):
    account_id = built_pipeline.alerts(top_n=1)["account_id"].iloc[0]
    case = built_pipeline.investigation(account_id)
    assert case.account_id == account_id


def test_query_before_build_raises():
    # This one must NOT use the fixture — it tests the unbuilt state.
    with pytest.raises(RuntimeError):
        Pipeline(GeneratorConfig(num_accounts=20, num_legit_transactions=50)).alerts()


def test_ego_graph_includes_focus_account(built_pipeline):
    account_id = built_pipeline.alerts(top_n=1)["account_id"].iloc[0]
    ego = built_pipeline.ego_graph(account_id)
    assert account_id in ego
