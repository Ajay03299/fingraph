"""Core data types for FinGraph.

Two entities sit at the heart of the system: accounts (the nodes of the
transaction graph) and transactions (the edges). Both are validated with
pydantic so bad data fails loudly at the boundary instead of quietly
corrupting everything downstream.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, computed_field


class AccountType(str, Enum):
    PERSONAL = "personal"
    BUSINESS = "business"


class TransactionPattern(str, Enum):
    """How a transaction came to exist. Anything other than LEGITIMATE is part
    of a planted laundering typology, and counts as ground-truth fraud."""

    LEGITIMATE = "legitimate"
    STRUCTURING = "structuring"  # many small deposits funnelled into one account
    LAYERING = "layering"        # funds pushed through a chain to hide their origin
    CYCLE = "cycle"              # money routed in a loop back to its source


class Account(BaseModel):
    account_id: str
    account_type: AccountType
    country: str
    opened_at: datetime


class Transaction(BaseModel):
    tx_id: str
    timestamp: datetime
    source: str
    destination: str
    amount: float
    channel: str
    pattern: TransactionPattern
    currency: str = "USD"

    @computed_field
    @property
    def is_laundering(self) -> bool:
        # One source of truth: a transaction is laundering iff it belongs to a
        # planted typology. Keeps the label and the pattern from ever drifting apart.
        return self.pattern != TransactionPattern.LEGITIMATE