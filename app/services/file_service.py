"""
File service module for handling file operations.
"""
import io
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.core.logging import logger
from app.db.supabase import supabase_client
from app.models.file_metadata import FileMetadata
from app.models.notebook_file import NotebookFile
from app.services.file_processors import FileProcessorFactory


class FileService:
    """
    Service for handling file operations.
    """

    @staticmethod
    async def get_file_metadata(file_id: str) -> Optional[FileMetadata]:
        """
        Fetches file metadata from the database.
        
        Args:
            file_id: The ID of the file.
            
        Returns:
            Optional[FileMetadata]: The file metadata, or None if not found.
        """
        try:
            metadata_dict = await supabase_client.get_file_metadata(file_id)
            if not metadata_dict:
                return None
            return FileMetadata.from_dict(metadata_dict)
        except Exception as e:
            logger.error(f"Error fetching file metadata: {e}")
            raise

    @staticmethod
    async def get_notebook_file(file_id: str) -> Optional[NotebookFile]:
        """
        Fetches notebook file information from the database.
        
        Args:
            file_id: The ID of the file.
            
        Returns:
            Optional[NotebookFile]: The notebook file, or None if not found.
        """
        try:
            file_dict = await supabase_client.get_notebook_file(file_id)
            if not file_dict:
                return None
            return NotebookFile.from_dict(file_dict)
        except Exception as e:
            logger.error(f"Error fetching notebook file: {e}")
            raise

    @staticmethod
    async def create_file_metadata(
        file_id: UUID, file_path: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> FileMetadata:
        """
        Creates a new file metadata record.
        
        Args:
            file_id: The ID of the file.
            file_path: The path of the file in storage.
            description: Optional description of the file.
            metadata: Optional additional metadata.
            
        Returns:
            FileMetadata: The created file metadata.
        """
        try:
            metadata_dict = {
                "file_id": str(file_id),
                "file_path": file_path,
                "description": description,
                "metadata": metadata or {},
            }
            
            result = await supabase_client.create_file_metadata(metadata_dict)
            return FileMetadata.from_dict(result)
        except Exception as e:
            logger.error(f"Error creating file metadata: {e}")
            raise

    @staticmethod
    async def update_file_metadata(
        id: UUID, 
        description: Optional[str] = None, 
        metadata: Optional[Dict[str, Any]] = None,
        pinecone_id: Optional[str] = None
    ) -> FileMetadata:
        """
        Updates an existing file metadata record.
        
        Args:
            id: The ID of the metadata record.
            description: Optional updated description.
            metadata: Optional updated metadata.
            pinecone_id: Optional Pinecone vector ID.
            
        Returns:
            FileMetadata: The updated file metadata.
        """
        try:
            update_dict = {}
            if description is not None:
                update_dict["description"] = description
            if metadata is not None:
                update_dict["metadata"] = metadata
            if pinecone_id is not None:
                update_dict["pinecone_id"] = pinecone_id
                
            result = await supabase_client.update_file_metadata(str(id), update_dict)
            return FileMetadata.from_dict(result)
        except Exception as e:
            logger.error(f"Error updating file metadata: {e}")
            raise

    @staticmethod
    async def fetch_file_content(file_path: str) -> bytes:
        """
        Fetches a file from Supabase storage.
        
        Args:
            file_path: The path of the file in storage.
            
        Returns:
            bytes: The file contents.
        """
        try:
            return await supabase_client.fetch_file_from_storage(file_path)
        except Exception as e:
            logger.error(f"Error fetching file from storage: {e}")
            raise

    @staticmethod
    async def get_file_text_content(file_path: str, file_type: str) -> str:
        """
        Fetches and processes a file to extract its text content.
        
        Args:
            file_path: The path of the file in storage.
            file_type: The MIME type of the file.
            
        Returns:
            str: The extracted text content.
        """
        try:
            # Fetch file content from storage
            file_content = await FileService.fetch_file_content(file_path)
            
            # Use the appropriate file processor based on file type
            try:
                processor = FileProcessorFactory.get_processor(file_type)
                return await processor.process(file_content, file_path)
            except ValueError as e:
                logger.warning(f"Unsupported file type for text extraction: {file_type}")
                return f"Unsupported file type: {file_type}"
            except Exception as e:
                logger.error(f"Error processing file with processor: {e}")
                # Fallback to basic text extraction for supported types
                if file_type in ["text/plain", "text/markdown", "text/csv"]:
                    return file_content.decode("utf-8")
                else:
                    return f"Error extracting content from {file_type} file: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise
    
    @staticmethod
    async def get_file_metadata_content(file_path: str, file_type: str) -> Dict[str, Any]:
        """
        Fetches and extracts metadata from a file.
        
        Args:
            file_path: The path of the file in storage.
            file_type: The MIME type of the file.
            
        Returns:
            Dict[str, Any]: The extracted metadata.
        """
        try:
            # Fetch file content from storage
            file_content = await FileService.fetch_file_content(file_path)
            
            # Use the appropriate file processor based on file type
            try:
                processor = FileProcessorFactory.get_processor(file_type)
                return await processor.get_metadata(file_content, file_path)
            except ValueError:
                logger.warning(f"Unsupported file type for metadata extraction: {file_type}")
                return {"error": f"Unsupported file type: {file_type}"}
            except Exception as e:
                logger.error(f"Error extracting metadata with processor: {e}")
                return {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Error extracting metadata from file: {e}")
            raise


file_service = FileService() 