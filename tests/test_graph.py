"""Tests for transaction graph construction."""

import networkx as nx

from fingraph.data.generator import GeneratorConfig, generate_dataset
from fingraph.graph.build import build_transaction_graph


def _dataset(seed: int = 3):
    return generate_dataset(GeneratorConfig(num_accounts=50, num_legit_transactions=200, seed=seed))


def test_graph_has_expected_nodes_and_edges():
    accounts, tx = _dataset()
    graph = build_transaction_graph(tx, accounts)
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_edges() == len(tx)


def test_edges_carry_laundering_label():
    accounts, tx = _dataset()
    graph = build_transaction_graph(tx, accounts)
    laundering_edges = [d for _, _, d in graph.edges(data=True) if d["is_laundering"]]
    assert len(laundering_edges) == int(tx["is_laundering"].sum())


def test_nodes_without_metadata_still_created():
    _, tx = _dataset()
    graph = build_transaction_graph(tx)  # no accounts frame
    assert graph.number_of_nodes() > 0
