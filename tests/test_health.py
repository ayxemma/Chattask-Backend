"""Tests for health-check endpoints."""
from starlette.testclient import TestClient


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_ok(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_content_type_is_json(client: TestClient):
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]
