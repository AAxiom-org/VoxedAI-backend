"""
Database models for file metadata.
"""
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


class FileMetadata:
    """
    Represents a file metadata record in the database.
    """

    id: UUID
    file_id: UUID
    file_path: str
    description: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    pinecone_id: Optional[str]

    def __init__(
        self,
        id: UUID,
        file_id: UUID,
        file_path: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        pinecone_id: Optional[str] = None,
    ):
        self.id = id
        self.file_id = file_id
        self.file_path = file_path
        self.description = description
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.pinecone_id = pinecone_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileMetadata":
        """
        Creates a FileMetadata instance from a dictionary.
        
        Args:
            data: The dictionary containing file metadata.
            
        Returns:
            FileMetadata: A new FileMetadata instance.
        """
        return cls(
            id=data.get("id"),
            file_id=data.get("file_id"),
            file_path=data.get("file_path"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            pinecone_id=data.get("pinecone_id"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the FileMetadata instance to a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the file metadata.
        """
        return {
            "id": str(self.id) if self.id else None,
            "file_id": str(self.file_id) if self.file_id else None,
            "file_path": self.file_path,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else self.created_at,
            "updated_at": self.updated_at.isoformat() if hasattr(self.updated_at, 'isoformat') else self.updated_at,
            "pinecone_id": self.pinecone_id,
        } 