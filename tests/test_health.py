"""
Tests for the health endpoint.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """
    Test client fixture.
    """
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """
    Test the health check endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"} 