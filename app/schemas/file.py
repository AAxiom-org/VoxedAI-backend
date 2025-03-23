"""
Pydantic schemas for file-related operations.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class FileMetadataBase(BaseModel):
    """Base schema for file metadata."""
    file_path: str = Field(..., description="Path of the file in storage")
    description: Optional[str] = Field(None, description="Description of the file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FileMetadataCreate(FileMetadataBase):
    """Schema for creating file metadata."""
    file_id: UUID = Field(..., description="ID of the space file")


class FileMetadataUpdate(BaseModel):
    """Schema for updating file metadata."""
    description: Optional[str] = Field(None, description="Description of the file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    pinecone_id: Optional[str] = Field(None, description="ID of the vector in Pinecone")


class FileMetadataInDB(FileMetadataBase):
    """Schema for file metadata in the database."""
    id: UUID = Field(..., description="ID of the file metadata record")
    file_id: UUID = Field(..., description="ID of the space file")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    pinecone_id: Optional[str] = Field(None, description="ID of the vector in Pinecone")

    class Config:
        from_attributes = True


class FileMetadataResponse(FileMetadataInDB):
    """Response schema for file metadata."""
    pass


class FileIngestRequest(BaseModel):
    """Request schema for file ingestion."""
    file_id: UUID = Field(..., description="ID of the file to ingest")


class FileIngestResponse(BaseModel):
    """Response schema for file ingestion."""
    success: bool = Field(..., description="Whether the ingestion was successful")
    metadata: FileMetadataResponse = Field(..., description="The created file metadata")
    message: str = Field(..., description="Status message")


class VectorMetadata(BaseModel):
    """Schema for vector metadata in Pinecone."""
    file_id: str = Field(..., description="ID of the file")
    file_path: str = Field(..., description="Path of the file in storage")
    text_chunk: str = Field(..., description="Text chunk represented by the vector")
    chunk_index: int = Field(..., description="Index of the chunk in the document")
    source: str = Field(..., description="Source of the text (file name, document title, etc.)")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryResult(BaseModel):
    """Schema for query results."""
    text: str = Field(..., description="Text chunk")
    score: float = Field(..., description="Similarity score")
    file_id: str = Field(..., description="ID of the file")
    file_path: str = Field(..., description="Path of the file in storage")
    source: str = Field(..., description="Source of the text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryRequest(BaseModel):
    """Request schema for querying."""
    query: str = Field(..., description="Query text")
    top_k: int = Field(5, description="Number of results to return")
    model_name: str = Field("gemini", description="LLM model to use (gemini, anthropic, openai)")
    use_rag: bool = Field(True, description="Whether to use RAG")
    stream: bool = Field(True, description="Whether to stream the response")
    namespace: Optional[str] = Field(None, description="Namespace to query in Pinecone")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter to apply to the query")
    user_id: Optional[str] = Field(None, description="User ID to fetch toggled files for RAG")
    is_coding_question: bool = Field(False, description="Whether the question is about coding")


class QueryResponse(BaseModel):
    """Response schema for querying."""
    answer: str = Field(..., description="LLM-generated answer")
    sources: List[QueryResult] = Field(default_factory=list, description="Sources used for the answer")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class DeleteByPineconeIdRequest(BaseModel):
    """Request schema for deleting vectors by Pinecone ID."""
    file_id: str = Field(..., description="The Pinecone ID of the file to delete") 