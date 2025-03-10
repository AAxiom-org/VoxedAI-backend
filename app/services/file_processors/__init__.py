"""
File processor modules for extracting content from different file types.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Define base class first to avoid circular imports
class FileProcessor(ABC):
    """
    Base abstract class for file processors.
    """
    
    @abstractmethod
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process file content and extract text.
        
        Args:
            file_content: Raw file content bytes
            file_path: Path to the file
            
        Returns:
            str: Extracted text content
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the file.
        
        Args:
            file_content: Raw file content bytes
            file_path: Path to the file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        pass


# Now import specific processors
from app.services.file_processors.audio_processor import AudioProcessor
from app.services.file_processors.document_processor import DocumentProcessor
from app.services.file_processors.spreadsheet_processor import SpreadsheetProcessor
from app.services.file_processors.presentation_processor import PresentationProcessor
from app.services.file_processors.image_processor import ImageProcessor
from app.services.file_processors.video_processor import VideoProcessor
from app.services.file_processors.text_processor import TextProcessor


class FileProcessorFactory:
    """
    Factory for creating file processors based on file type.
    """
    
    @staticmethod
    def get_processor(file_type: str) -> FileProcessor:
        """
        Get the appropriate processor for a file type.
        
        Args:
            file_type: MIME type of the file
            
        Returns:
            FileProcessor: An instance of the appropriate processor
        """
        # Audio processors
        if file_type in ["audio/mpeg", "audio/wav", "audio/mp4", "audio/webm"]:
            from app.services.file_processors.audio_processor import AudioProcessor
            return AudioProcessor()
            
        # Document processors
        elif file_type in ["application/pdf"]:
            from app.services.file_processors.document_processor import PDFProcessor
            return PDFProcessor()
        elif file_type in ["text/plain"]:
            from app.services.file_processors.text_processor import PlainTextProcessor
            return PlainTextProcessor()
        elif file_type in ["text/markdown"]:
            from app.services.file_processors.text_processor import MarkdownProcessor
            return MarkdownProcessor()
        elif file_type in ["application/json"]:
            from app.services.file_processors.text_processor import JSONProcessor
            return JSONProcessor()
        elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            from app.services.file_processors.document_processor import WordProcessor
            return WordProcessor()
            
        # Spreadsheet processors
        elif file_type in ["text/csv"]:
            from app.services.file_processors.spreadsheet_processor import CSVProcessor
            return CSVProcessor()
        elif file_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            from app.services.file_processors.spreadsheet_processor import ExcelProcessor
            return ExcelProcessor()
            
        # Presentation processors
        elif file_type in ["application/vnd.openxmlformats-officedocument.presentationml.presentation"]:
            from app.services.file_processors.presentation_processor import PowerPointProcessor
            return PowerPointProcessor()
            
        # Image processors
        elif file_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            from app.services.file_processors.image_processor import ImageProcessor
            return ImageProcessor()
            
        # Video processors
        elif file_type in ["video/mp4", "video/webm"]:
            from app.services.file_processors.video_processor import VideoProcessor
            return VideoProcessor()
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 