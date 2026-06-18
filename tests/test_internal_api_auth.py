from fastapi.testclient import TestClient

from app.main import app


def test_health_is_public(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_API_KEY", raising=False)

    response = TestClient(app).get("/health")

    assert response.status_code == 200


def test_api_is_closed_when_key_is_not_configured(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_API_KEY", raising=False)

    response = TestClient(app).get("/api/notices/missing")

    assert response.status_code == 503
    assert response.json()["error"] == "AI_AUTH_NOT_CONFIGURED"


def test_api_rejects_invalid_key(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_API_KEY", "test-internal-key")

    response = TestClient(app).get(
        "/api/notices/missing",
        headers={"X-Internal-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "UNAUTHORIZED"


def test_api_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_API_KEY", "test-internal-key")

    response = TestClient(app).get(
        "/api/notices/missing",
        headers={"X-Internal-API-Key": "test-internal-key"},
    )

    assert response.status_code == 404
    assert response.json()["error"] == "NOTICE_NOT_FOUND"
