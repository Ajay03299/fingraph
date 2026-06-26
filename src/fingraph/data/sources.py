"""Data source abstraction.

The pipeline shouldn't care whether transactions come from the synthetic
generator or a real benchmark file — it just needs accounts and transactions in
the internal schema. A DataSource is anything that can hand back that pair, which
lets the synthetic generator and the IBM loader sit behind one interface.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from fingraph.data.generator import GeneratorConfig, generate_dataset


class DataSource(Protocol):
    """Anything that yields (accounts, transactions) in FinGraph's schema."""

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class SyntheticSource:
    """Wraps the seeded synthetic generator as a DataSource."""

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return generate_dataset(self.config)