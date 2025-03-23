"""
Database models for space files.
"""
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


class NotebookFile:
    """
    Represents a space file record in the database.
    """

    id: UUID
    space_id: UUID
    user_id: str
    file_name: str
    file_path: str
    file_type: str
    file_size: int
    created_at: datetime

    def __init__(
        self,
        id: UUID,
        space_id: UUID,
        user_id: str,
        file_name: str,
        file_path: str,
        file_type: str,
        file_size: int,
        created_at: Optional[datetime] = None,
    ):
        self.id = id
        self.space_id = space_id
        self.user_id = user_id
        self.file_name = file_name
        self.file_path = file_path
        self.file_type = file_type
        self.file_size = file_size
        self.created_at = created_at or datetime.now()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotebookFile":
        """
        Creates a NotebookFile instance from a dictionary.
        
        Args:
            data: The dictionary containing space file data.
            
        Returns:
            NotebookFile: A new NotebookFile instance.
        """
        return cls(
            id=data.get("id"),
            space_id=data.get("space_id"),
            user_id=data.get("user_id"),
            file_name=data.get("file_name"),
            file_path=data.get("file_path"),
            file_type=data.get("file_type"),
            file_size=data.get("file_size"),
            created_at=data.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the NotebookFile instance to a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary representation of the space file.
        """
        return {
            "id": str(self.id) if self.id else None,
            "space_id": str(self.space_id) if self.space_id else None,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @property
    def is_text_file(self) -> bool:
        """
        Checks if the file is a text file.
        
        Returns:
            bool: True if the file is a text file, False otherwise.
        """
        return self.file_type in [
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "application/pdf",
        ]
    
    @property
    def is_audio_file(self) -> bool:
        """
        Checks if the file is an audio file.
        
        Returns:
            bool: True if the file is an audio file, False otherwise.
        """
        return self.file_type in [
            "audio/mpeg",
            "audio/wav",
            "audio/mp4",
            "audio/webm",
        ]
    
    @property
    def is_image_file(self) -> bool:
        """
        Checks if the file is an image file.
        
        Returns:
            bool: True if the file is an image file, False otherwise.
        """
        return self.file_type in [
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        ]
    
    @property
    def is_video_file(self) -> bool:
        """
        Checks if the file is a video file.
        
        Returns:
            bool: True if the file is a video file, False otherwise.
        """
        return self.file_type in [
            "video/mp4",
            "video/webm",
        ]
    
    @property
    def is_document_file(self) -> bool:
        """
        Checks if the file is a document file.
        
        Returns:
            bool: True if the file is a document file, False otherwise.
        """
        return self.file_type in [
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ] 