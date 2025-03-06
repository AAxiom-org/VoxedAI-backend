"""
Tests for the query endpoints.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def mock_rag_service():
    """
    Mock RAG service fixture.
    """
    with patch("app.api.v1.endpoints.query.rag_service") as mock:
        # Set up mock methods
        mock.query = AsyncMock()
        mock.retrieve_context = AsyncMock()
        mock.generate_answer = AsyncMock()
        
        yield mock


def test_query_non_streaming(client, mock_rag_service):
    """
    Test the query endpoint with non-streaming response.
    """
    # Set up test data
    query_text = "Test query"
    
    # Set up mock returns
    context = [
        {
            "text": "Test context",
            "score": 0.9,
            "file_id": "test_file_id",
            "file_path": "test_path/test_file.txt",
            "source": "test_file.txt",
            "metadata": {"chunk_index": 0}
        }
    ]
    mock_rag_service.query.return_value = ("Test answer", context, 123.45)
    
    # Make request
    response = client.post(
        "/api/v1/query",
        json={
            "query": query_text,
            "top_k": 3,
            "model_name": "gemini",
            "use_rag": True,
            "stream": False
        }
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"
    assert len(response.json()["sources"]) == 1
    assert response.json()["sources"][0]["text"] == "Test context"
    assert response.json()["query_time_ms"] == 123.45
    
    # Verify mock calls
    mock_rag_service.query.assert_called_once_with(
        query=query_text,
        top_k=3,
        model_name="gemini",
        use_rag=True,
        stream=False,
        namespace=None,
        filter=None
    )


def test_query_streaming(client, mock_rag_service):
    """
    Test the query endpoint with streaming response.
    """
    # Set up test data
    query_text = "Test query"
    
    # Set up mock returns
    context = [
        {
            "text": "Test context",
            "score": 0.9,
            "file_id": "test_file_id",
            "file_path": "test_path/test_file.txt",
            "source": "test_file.txt",
            "metadata": {"chunk_index": 0}
        }
    ]
    mock_rag_service.retrieve_context.return_value = context
    
    # Create a mock async generator for the answer stream
    async def mock_stream():
        yield "Test "
        yield "answer"
    
    mock_rag_service.generate_answer.return_value = mock_stream()
    
    # Make request
    with client.stream(
        "POST",
        "/api/v1/query",
        json={
            "query": query_text,
            "top_k": 3,
            "model_name": "gemini",
            "use_rag": True,
            "stream": True
        }
    ) as response:
        # Check response
        assert response.status_code == 200
        
        # Collect streamed data
        chunks = []
        for chunk in response.iter_lines():
            if chunk:
                chunks.append(chunk)
        
        # Parse and check the chunks
        assert len(chunks) >= 3  # sources + tokens + done
        
        # Check sources chunk
        sources_chunk = json.loads(chunks[0].replace("data: ", ""))
        assert sources_chunk["type"] == "sources"
        assert len(sources_chunk["data"]) == 1
        assert sources_chunk["data"][0]["text"] == "Test context"
        
        # Check token chunks
        token_chunks = [json.loads(c.replace("data: ", "")) for c in chunks[1:-1]]
        for chunk in token_chunks:
            assert chunk["type"] == "token"
        
        # Check done chunk
        done_chunk = json.loads(chunks[-1].replace("data: ", ""))
        assert done_chunk["type"] == "done"
        assert "query_time_ms" in done_chunk
    
    # Verify mock calls
    mock_rag_service.retrieve_context.assert_called_once_with(
        query=query_text,
        top_k=3,
        namespace=None,
        filter=None
    )
    mock_rag_service.generate_answer.assert_called_once_with(
        query=query_text,
        context=context,
        model_name="gemini",
        stream=True
    ) 