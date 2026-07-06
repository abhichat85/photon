"""The routing policy: predicts P(the cheap model's output is acceptable) from
request features. Calibrated logistic regression — fast, interpretable, and 3+
orders of magnitude cheaper than the inference it gates (parent spec anti-goal:
never an LLM-as-router). Trained offline on replay data (learning.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from photon.core.features import RequestFeatures


class PolicyModel:
    def __init__(self):
        self._model = None  # sklearn estimator, set by fit()/load()

    def fit(self, features: list[RequestFeatures], labels: list[int]) -> None:
        from sklearn.linear_model import LogisticRegression

        X = np.array([f.to_vector() for f in features], dtype=float)
        y = np.array(labels, dtype=int)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self._model = model

    def predict_acceptable(self, features: RequestFeatures) -> float:
        """Return P(cheap acceptable) in [0, 1]. Fails closed (0.0) when
        untrained, so an unfit policy never routes anything to the cheap model."""
        if self._model is None:
            return 0.0
        X = np.array([features.to_vector()], dtype=float)
        return float(self._model.predict_proba(X)[0][1])

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str | Path) -> "PolicyModel":
        import joblib

        inst = cls()
        inst._model = joblib.load(path)
        return inst
