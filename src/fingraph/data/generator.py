"""Synthetic transaction generator for FinGraph.

Produces a retail-bank style dataset: a population of accounts, a stream of
ordinary day-to-day transactions, and a handful of planted money-laundering
patterns hidden inside the noise. Because the laundering is planted, we know
the ground truth and can measure detection quality later.

Everything is seeded, so a given config always produces the same dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fingraph.data.schema import Account, AccountType, Transaction, TransactionPattern

CHANNELS = ["wire", "ach", "card", "p2p"]
COUNTRIES = ["US", "GB", "DE", "SG", "AE", "NG"]
COUNTRY_WEIGHTS = [0.55, 0.15, 0.10, 0.08, 0.07, 0.05]


@dataclass
class GeneratorConfig:
    """Knobs for the synthetic dataset. The defaults give a small, fast,
    realistic set that's comfortable to work with on a laptop."""

    num_accounts: int = 500
    num_legit_transactions: int = 6000
    num_structuring_rings: int = 5
    num_layering_chains: int = 8
    num_cycles: int = 4
    start_date: datetime = field(default_factory=lambda: datetime(2025, 1, 1))
    days: int = 90
    seed: int = 42


class SyntheticDataGenerator:
    """Builds accounts and transactions for one dataset.

    The flow is: lay down a population of accounts, fill the period with
    ordinary transfers, then thread the laundering typologies through the
    same accounts so the fraud looks like it's hiding among real customers.
    """

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.accounts: list[Account] = []
        self.transactions: list[Transaction] = []

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._build_accounts()
        self._build_legit_traffic()
        for _ in range(self.config.num_structuring_rings):
            self._plant_structuring()
        for _ in range(self.config.num_layering_chains):
            self._plant_layering()
        for _ in range(self.config.num_cycles):
            self._plant_cycle()
        return self._to_frames()

    # --- helpers -----------------------------------------------------------

    def _account_ids(self) -> list[str]:
        return [a.account_id for a in self.accounts]

    def _random_timestamp(self) -> datetime:
        offset = int(self.rng.integers(0, self.config.days * 86400))
        return self.config.start_date + timedelta(seconds=offset)

    def _add(self, source, destination, amount, timestamp, pattern) -> None:
        # tx_id is left blank here and assigned once everything is sorted by time.
        self.transactions.append(
            Transaction(
                tx_id="",
                timestamp=timestamp,
                source=str(source),
                destination=str(destination),
                amount=round(float(amount), 2),
                channel=str(self.rng.choice(CHANNELS)),
                pattern=pattern,
            )
        )

    # --- population --------------------------------------------------------

    def _build_accounts(self) -> None:
        for i in range(self.config.num_accounts):
            # Roughly a fifth of accounts are businesses; the rest personal.
            is_business = self.rng.random() < 0.2
            age_days = int(self.rng.integers(30, 1500))
            self.accounts.append(
                Account(
                    account_id=f"ACC{i:06d}",
                    account_type=AccountType.BUSINESS if is_business else AccountType.PERSONAL,
                    country=str(self.rng.choice(COUNTRIES, p=COUNTRY_WEIGHTS)),
                    opened_at=self.config.start_date - timedelta(days=age_days),
                )
            )

    def _build_legit_traffic(self) -> None:
        ids = self._account_ids()
        for _ in range(self.config.num_legit_transactions):
            # choice with replace=False guarantees we never transfer to ourselves.
            source, destination = self.rng.choice(ids, size=2, replace=False)
            amount = self.rng.lognormal(mean=4.5, sigma=1.1)  # long-tailed, like real spend
            self._add(source, destination, amount, self._random_timestamp(),
                      TransactionPattern.LEGITIMATE)

    # --- planted laundering typologies ------------------------------------

    def _plant_structuring(self) -> None:
        # Fan-in: a crowd of mule accounts each drip a small amount into one
        # collector, every deposit deliberately under a $10k reporting line.
        ids = self._account_ids()
        collector = self.rng.choice(ids)
        others = [i for i in ids if i != collector]
        num_mules = int(self.rng.integers(8, 16))
        mules = self.rng.choice(others, size=num_mules, replace=False)
        window_start = self._random_timestamp()
        for mule in mules:
            amount = self.rng.uniform(1000, 9500)
            ts = window_start + timedelta(hours=float(self.rng.uniform(0, 72)))
            self._add(mule, collector, amount, ts, TransactionPattern.STRUCTURING)

    def _plant_layering(self) -> None:
        # A chain A -> B -> C -> ... with a slice skimmed at each hop and the
        # money moving fast. Hard to spot per-transaction, glaring as a path.
        ids = self._account_ids()
        chain_len = int(self.rng.integers(4, 7))
        chain = self.rng.choice(ids, size=chain_len, replace=False)
        amount = self.rng.uniform(20000, 80000)
        ts = self._random_timestamp()
        for src, dst in zip(chain[:-1], chain[1:]):
            self._add(src, dst, amount, ts, TransactionPattern.LAYERING)
            amount *= self.rng.uniform(0.90, 0.98)
            ts += timedelta(hours=float(self.rng.uniform(1, 12)))

    def _plant_cycle(self) -> None:
        # A closed loop that returns the money to where it began.
        ids = self._account_ids()
        ring_len = int(self.rng.integers(3, 6))
        ring = list(self.rng.choice(ids, size=ring_len, replace=False))
        closed = ring + [ring[0]]
        amount = self.rng.uniform(15000, 50000)
        ts = self._random_timestamp()
        for src, dst in zip(closed[:-1], closed[1:]):
            self._add(src, dst, amount, ts, TransactionPattern.CYCLE)
            amount *= self.rng.uniform(0.95, 0.99)
            ts += timedelta(hours=float(self.rng.uniform(1, 8)))

    # --- output ------------------------------------------------------------

    def _to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Sort chronologically, then hand out stable, readable transaction ids.
        self.transactions.sort(key=lambda t: t.timestamp)
        for i, tx in enumerate(self.transactions):
            tx.tx_id = f"TX{i:07d}"

        accounts_df = pd.DataFrame([a.model_dump(mode="json") for a in self.accounts])
        tx_df = pd.DataFrame([t.model_dump(mode="json") for t in self.transactions])
        accounts_df["opened_at"] = pd.to_datetime(accounts_df["opened_at"], format="ISO8601")
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"], format="ISO8601")
        return accounts_df, tx_df


def generate_dataset(config: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: build a dataset from a config (or sensible defaults)."""
    return SyntheticDataGenerator(config or GeneratorConfig()).generate()