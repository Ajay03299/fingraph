"""Tests for account feature engineering."""

from fingraph.data.generator import GeneratorConfig, generate_dataset
from fingraph.features.account_features import compute_account_features
from fingraph.graph.build import build_transaction_graph


def _features(seed: int = 3):
    accounts, tx = generate_dataset(
        GeneratorConfig(num_accounts=80, num_legit_transactions=400, seed=seed)
    )
    graph = build_transaction_graph(tx, accounts)
    return compute_account_features(graph)


def test_one_row_per_account():
    feats = _features()
    assert feats.index.is_unique


def test_expected_feature_columns_present():
    feats = _features()
    expected = {"total_in", "total_out", "fan_in_ratio", "in_cycle", "in_burstiness"}
    assert expected.issubset(feats.columns)


def test_some_accounts_sit_in_cycles():
    # The generator plants cycles, so at least a few accounts should be flagged.
    feats = _features()
    assert feats["in_cycle"].sum() > 0


def test_features_have_no_missing_values():
    feats = _features()
    assert not feats.isnull().any().any()
