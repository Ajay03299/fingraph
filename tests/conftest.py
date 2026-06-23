"""Shared pytest fixtures.

A built synthetic pipeline is expensive, so we build one per test session and
share it. Tests that only read from a pipeline (querying alerts, investigations)
reuse this instead of each building their own.
"""

import pytest

from fingraph.data.generator import GeneratorConfig
from fingraph.pipeline import Pipeline


@pytest.fixture(scope="session")
def built_pipeline() -> Pipeline:
    config = GeneratorConfig(num_accounts=100, num_legit_transactions=600, seed=3)
    return Pipeline(config).build()
