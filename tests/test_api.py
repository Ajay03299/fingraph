"""Tests for the FastAPI service, exercised through FastAPI's TestClient."""

from fastapi.testclient import TestClient

from fingraph.api.app import app


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_metrics_endpoint():
    with TestClient(app) as client:
        body = client.get("/metrics").json()
        assert "roc_auc" in body


def test_alerts_endpoint():
    with TestClient(app) as client:
        alerts = client.get("/alerts?top_n=5").json()
        assert len(alerts) == 5
        assert "risk_score" in alerts[0]


def test_unknown_account_returns_404():
    with TestClient(app) as client:
        assert client.get("/accounts/NOPE").status_code == 404
