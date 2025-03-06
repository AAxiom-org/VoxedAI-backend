"""
Tests for the file endpoints.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.file_metadata import FileMetadata
from app.models.notebook_file import NotebookFile


@pytest.fixture
def client():
    """
    Test client fixture.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_file_service():
    """
    Mock file service fixture.
    """
    with patch("app.api.v1.endpoints.files.file_service") as mock:
        # Set up mock methods
        mock.get_notebook_file = AsyncMock()
        mock.get_file_metadata = AsyncMock()
        mock.create_file_metadata = AsyncMock()
        mock.update_file_metadata = AsyncMock()
        mock.get_file_text_content = AsyncMock()
        
        yield mock


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service fixture.
    """
    with patch("app.api.v1.endpoints.files.llm_service") as mock:
        # Set up mock methods
        mock.generate_file_description = AsyncMock()
        
        yield mock


@pytest.fixture
def mock_embedding_service():
    """
    Mock embedding service fixture.
    """
    with patch("app.api.v1.endpoints.files.embedding_service") as mock:
        # Set up mock methods
        mock.delete_file_vectors = AsyncMock()
        
        yield mock


def test_ingest_file(client, mock_file_service, mock_llm_service):
    """
    Test the file ingestion endpoint.
    """
    # Set up test data
    file_id = str(uuid.uuid4())
    
    # Set up mock returns
    notebook_file = NotebookFile(
        id=uuid.UUID(file_id),
        notebook_id=uuid.uuid4(),
        user_id="test_user",
        file_name="test_file.txt",
        file_path="test_path/test_file.txt",
        file_type="text/plain",
        file_size=100
    )
    mock_file_service.get_notebook_file.return_value = notebook_file
    mock_file_service.get_file_metadata.return_value = None
    
    mock_file_service.get_file_text_content.return_value = "Test file content"
    
    mock_llm_result = {
        "description": "Test description",
        "metadata": {
            "topics": ["test"],
            "entities": ["test"],
            "key_points": ["test"]
        }
    }
    mock_llm_service.generate_file_description.return_value = mock_llm_result
    
    metadata = FileMetadata(
        id=uuid.uuid4(),
        file_id=uuid.UUID(file_id),
        file_path="test_path/test_file.txt",
        description="Test description",
        metadata=mock_llm_result["metadata"]
    )
    mock_file_service.create_file_metadata.return_value = metadata
    
    # Make request
    response = client.post(
        "/api/v1/files/ingest",
        json={"file_id": file_id}
    )
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == "File ingestion started"
    assert response.json()["metadata"]["file_id"] == file_id
    
    # Verify mock calls
    mock_file_service.get_notebook_file.assert_called_once_with(file_id)
    mock_file_service.get_file_metadata.assert_called_once_with(file_id)
    mock_file_service.get_file_text_content.assert_called_once_with(
        notebook_file.file_path, notebook_file.file_type
    )
    mock_llm_service.generate_file_description.assert_called_once_with(
        file_content="Test file content",
        file_name=notebook_file.file_name,
        file_type=notebook_file.file_type
    )
    mock_file_service.create_file_metadata.assert_called_once_with(
        file_id=uuid.UUID(file_id),
        file_path=notebook_file.file_path,
        description=mock_llm_result["description"],
        metadata=mock_llm_result["metadata"]
    )


def test_get_file_metadata(client, mock_file_service):
    """
    Test the get file metadata endpoint.
    """
    # Set up test data
    file_id = str(uuid.uuid4())
    
    # Set up mock returns
    metadata = FileMetadata(
        id=uuid.uuid4(),
        file_id=uuid.UUID(file_id),
        file_path="test_path/test_file.txt",
        description="Test description",
        metadata={"test": "test"}
    )
    mock_file_service.get_file_metadata.return_value = metadata
    
    # Make request
    response = client.get(f"/api/v1/files/{file_id}/metadata")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["file_id"] == file_id
    assert response.json()["description"] == "Test description"
    
    # Verify mock calls
    mock_file_service.get_file_metadata.assert_called_once_with(file_id)


def test_delete_file_vectors(client, mock_file_service, mock_embedding_service):
    """
    Test the delete file vectors endpoint.
    """
    # Set up test data
    file_id = str(uuid.uuid4())
    
    # Set up mock returns
    metadata = FileMetadata(
        id=uuid.uuid4(),
        file_id=uuid.UUID(file_id),
        file_path="test_path/test_file.txt",
        description="Test description",
        metadata={"test": "test"},
        pinecone_id="test_pinecone_id"
    )
    mock_file_service.get_file_metadata.return_value = metadata
    mock_embedding_service.delete_file_vectors.return_value = {"deleted": 10}
    
    # Make request
    response = client.delete(f"/api/v1/files/{file_id}")
    
    # Check response
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["message"] == f"Vectors for file {file_id} deleted"
    assert response.json()["details"] == {"deleted": 10}
    
    # Verify mock calls
    mock_embedding_service.delete_file_vectors.assert_called_once_with(file_id, "")
    mock_file_service.get_file_metadata.assert_called_once_with(file_id)
    mock_file_service.update_file_metadata.assert_called_once_with(
        id=metadata.id,
        pinecone_id=None
    ) 