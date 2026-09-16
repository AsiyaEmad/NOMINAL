from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_application_starts_and_health_returns_service_status() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "NOMINAL",
        "environment": "development",
        "enabled_models": 3,
    }


def test_cors_allows_the_local_dashboard_origin() -> None:
    with TestClient(create_app()) as client:
        response = client.options(
            "/api/metrics/summary",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
