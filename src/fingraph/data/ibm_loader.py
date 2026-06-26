"""Loader for the IBM Anti-Money-Laundering benchmark dataset.

The IBM file is large (~5M rows) and its laundering labels are extremely rare
(~0.1%), so two things matter: mapping its real-world schema onto FinGraph's
internal one, and sampling sensibly. We keep every laundering transaction and
sample legitimate ones around them — losing the rare positive signal to a naive
random sample would defeat the purpose.

IBM columns of interest:
    Timestamp, From Bank, Account (sender), To Bank, Account.1 (receiver),
    Amount Paid, Payment Currency, Payment Format, Is Laundering
Account identity is Bank + Account, since the same account number can recur
across different banks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fingraph.data.schema import TransactionPattern

# pandas renames the second duplicate "Account" column to "Account.1" on load.
_SENDER_ACCT = "Account"
_RECEIVER_ACCT = "Account.1"


class IBMSource:
    """Loads and adapts the IBM AML benchmark into FinGraph's schema.

    Keeps all laundering rows and samples legitimate rows so the resulting graph
    is laptop-sized while preserving the (rare) positive signal.
    """

    def __init__(self, path: str | Path, max_rows: int = 300_000, seed: int = 42):
        self.path = Path(path)
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = pd.read_csv(self.path)
        raw = self._sample(raw)
        transactions = self._to_transactions(raw)
        accounts = self._derive_accounts(transactions)
        return accounts, transactions

    # --- sampling ----------------------------------------------------------

    def _sample(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Keep every laundering row, fill the rest with sampled legitimate rows."""
        if len(raw) <= self.max_rows:
            return raw

        dirty = raw[raw["Is Laundering"] == 1]
        legit = raw[raw["Is Laundering"] == 0]

        # Reserve room for all dirty rows; sample legit rows for the remainder.
        legit_budget = max(self.max_rows - len(dirty), 0)
        legit_budget = min(legit_budget, len(legit))
        sampled_idx = self.rng.choice(legit.index, size=legit_budget, replace=False)
        sampled_legit = legit.loc[sampled_idx]

        combined = pd.concat([dirty, sampled_legit])
        return combined.sort_values("Timestamp").reset_index(drop=True)

    # --- schema mapping ----------------------------------------------------

    def _to_transactions(self, raw: pd.DataFrame) -> pd.DataFrame:
        # An account is uniquely identified by its bank plus its account number.
        source = raw["From Bank"].astype(str) + "-" + raw[_SENDER_ACCT].astype(str)
        destination = raw["To Bank"].astype(str) + "-" + raw[_RECEIVER_ACCT].astype(str)

        tx = pd.DataFrame(
            {
                "tx_id": [f"TX{i:08d}" for i in range(len(raw))],
                "timestamp": pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M"),
                "source": source,
                "destination": destination,
                "amount": raw["Amount Paid"].astype(float).round(2),
                "channel": raw["Payment Format"].astype(str),
                "pattern": np.where(
                    raw["Is Laundering"] == 1,
                    TransactionPattern.LAYERING.value,  # generic laundering tag
                    TransactionPattern.LEGITIMATE.value,
                ),
                "currency": raw["Payment Currency"].astype(str),
                "is_laundering": raw["Is Laundering"] == 1,
            }
        )

        # Real data contains self-loops (e.g. reinvestments); drop them, since a
        # node transacting with itself adds no relational signal to the graph.
        tx = tx[tx["source"] != tx["destination"]].reset_index(drop=True)
        return tx

    def _derive_accounts(self, transactions: pd.DataFrame) -> pd.DataFrame:
        ids = pd.unique(
            pd.concat([transactions["source"], transactions["destination"]])
        )
        # The benchmark has no rich account metadata we map, so accounts carry
        # just an id; downstream code already tolerates missing node attributes.
        return pd.DataFrame({"account_id": ids})