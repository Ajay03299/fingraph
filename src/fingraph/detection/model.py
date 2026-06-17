"""Anomaly detection model for FinGraph.

An IsolationForest scores each account by how easily it can be isolated from
the rest of the population — anomalies separate with few splits, ordinary
accounts need many. The model is unsupervised: it never sees the laundering
labels. Those are kept aside purely to evaluate the scores afterwards, which
mirrors how real AML teams work with little or no labelled fraud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """Fits an IsolationForest over account features and produces risk scores.

    Scores are normalised to [0, 1] where higher means more suspicious, so the
    rest of the system has a single consistent scale to reason about.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        # contamination is the model's prior on what fraction of accounts are
        # anomalous. We keep it small and deliberately independent of the true
        # rate — in production nobody knows the real number.
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
        )
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> AnomalyDetector:
        self.feature_names = list(features.columns)
        scaled = self.scaler.fit_transform(features.values)
        self.model.fit(scaled)
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Return a risk score in [0, 1] per account, higher = more suspicious.

        IsolationForest's raw decision_function is higher for normal points, so
        we negate it, then min-max normalise across the population so scores are
        comparable and easy to threshold downstream.
        """
        scaled = self.scaler.transform(features.values)
        raw = -self.model.decision_function(scaled)  # flip: high = anomalous
        spread = raw.max() - raw.min()
        normalised = (raw - raw.min()) / spread if spread else np.zeros_like(raw)
        return pd.Series(normalised, index=features.index, name="risk_score")

    def fit_score(self, features: pd.DataFrame) -> pd.Series:
        return self.fit(features).score(features)
