"""Tests for the synthetic data generator."""

import pandas as pd

from fingraph.data.generator import GeneratorConfig, generate_dataset


def _small_config(seed: int = 7) -> GeneratorConfig:
    return GeneratorConfig(
        num_accounts=60,
        num_legit_transactions=300,
        num_structuring_rings=2,
        num_layering_chains=2,
        num_cycles=2,
        seed=seed,
    )


def test_dataset_is_reproducible():
    accounts_a, tx_a = generate_dataset(_small_config())
    accounts_b, tx_b = generate_dataset(_small_config())
    pd.testing.assert_frame_equal(accounts_a, accounts_b)
    pd.testing.assert_frame_equal(tx_a, tx_b)


def test_no_self_transfers():
    _, tx = generate_dataset(_small_config())
    assert (tx["source"] != tx["destination"]).all()


def test_contains_planted_laundering():
    _, tx = generate_dataset(_small_config())
    assert tx["is_laundering"].sum() > 0


def test_legitimate_transactions_are_never_flagged():
    _, tx = generate_dataset(_small_config())
    legit = tx[tx["pattern"] == "legitimate"]
    assert not legit["is_laundering"].any()


def test_transaction_ids_are_unique():
    _, tx = generate_dataset(_small_config())
    assert tx["tx_id"].is_unique
