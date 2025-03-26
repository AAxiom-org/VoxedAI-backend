"""
Test script for the agent implementation using PocketFlow.
"""
import asyncio
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.agent import AgentRequest, AgentResponse
from app.agents.base.flow import run_agent_flow


@pytest.mark.asyncio
async def test_agent_flow():
    """Test the agent flow directly."""
    response = await run_agent_flow(
        space_id="test-space",
        query="What is the weather today?",
        view=None,
        stream=False
    )
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_agent_endpoint():
    """Test the agent endpoint via the API."""
    client = TestClient(app)
    response = client.post(
        "/api/v1/agent/run",
        json={
            "space_id": "test-space",
            "query": "What is the weather today?",
            "view": None,
            "stream": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "response" in data
    assert "metadata" in data


if __name__ == "__main__":
    asyncio.run(test_agent_flow())
    print("Agent flow test completed successfully") 