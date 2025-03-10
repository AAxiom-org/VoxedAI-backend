"""
Text processor module for extracting text from plain text and markdown files.
"""

import os
import re
import io
import base64
import json
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse

import markdown
import requests
from PIL import Image

from app.core.logging import logger
from app.services.file_processors import FileProcessor


class TextProcessor(FileProcessor):
    """
    Base processor for text files.
    """
    
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process text content.
        
        Args:
            file_content: Raw text file content bytes
            file_path: Path to the text file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Decode bytes to string
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            try:
                return file_content.decode('latin-1')
            except Exception as e:
                logger.error(f"Error decoding text file: {e}")
                raise
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from text file.
        
        Args:
            file_content: Raw text file content bytes
            file_path: Path to the text file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Basic metadata
        metadata = {
            'file_size': len(file_content),
            'file_extension': os.path.splitext(file_path)[1],
            'line_count': 0,
            'word_count': 0,
            'char_count': 0
        }
        
        try:
            # Decode content
            content = await self.process(file_content, file_path)
            
            # Count lines, words, and characters
            lines = content.split('\n')
            metadata['line_count'] = len(lines)
            
            words = content.split()
            metadata['word_count'] = len(words)
            
            metadata['char_count'] = len(content)
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting text metadata: {e}")
            return metadata


class PlainTextProcessor(TextProcessor):
    """
    Processor for plain text files.
    Handles: text/plain
    """
    pass


class MarkdownProcessor(TextProcessor):
    """
    Processor for Markdown files.
    Handles: text/markdown
    """
    
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process Markdown content.
        
        Args:
            file_content: Raw Markdown file content bytes
            file_path: Path to the Markdown file
            
        Returns:
            str: Extracted text content with Markdown formatting
        """
        # Get the raw text content
        content = await super().process(file_content, file_path)
        
        # Return the raw Markdown content
        return content
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from Markdown file.
        
        Args:
            file_content: Raw Markdown file content bytes
            file_path: Path to the Markdown file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Get basic metadata
        metadata = await super().get_metadata(file_content, file_path)
        
        try:
            # Decode content
            content = await super().process(file_content, file_path)
            
            # Extract headings
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            
            if headings:
                metadata['headings'] = [
                    {'level': len(h[0]), 'text': h[1].strip()} 
                    for h in headings
                ]
                
                # Use first heading as title if available
                if metadata['headings']:
                    metadata['title'] = metadata['headings'][0]['text']
            
            # Extract links
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            
            if links:
                metadata['links'] = [
                    {'text': l[0], 'url': l[1]} 
                    for l in links
                ]
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting Markdown metadata: {e}")
            return metadata


class JSONProcessor(TextProcessor):
    """
    Processor for JSON files.
    Handles: application/json
    """
    
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process JSON content by converting it to a structured string.
        
        Args:
            file_content: Raw JSON file content bytes
            file_path: Path to the JSON file
            
        Returns:
            str: Structured string representation of the JSON content
        """
        # Get the raw text content
        content = await super().process(file_content, file_path)
        
        try:
            # Parse JSON
            json_data = json.loads(content)
            
            # Convert to a formatted string with indentation for readability
            formatted_json = json.dumps(json_data, indent=2)
            
            # Add a header to indicate this is a JSON document
            return f"JSON Document:\n{formatted_json}"
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            # Return the raw content if parsing fails
            return f"Invalid JSON Document:\n{content}"
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from JSON file.
        
        Args:
            file_content: Raw JSON file content bytes
            file_path: Path to the JSON file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Get basic metadata
        metadata = await super().get_metadata(file_content, file_path)
        
        try:
            # Decode content
            content = await super().process(file_content, file_path)
            
            # Parse JSON
            json_data = json.loads(content)
            
            # Add JSON-specific metadata
            if isinstance(json_data, dict):
                metadata['top_level_keys'] = list(json_data.keys())
                metadata['key_count'] = len(json_data.keys())
            elif isinstance(json_data, list):
                metadata['array_length'] = len(json_data)
                
                # Sample structure of first item if list is not empty
                if json_data and isinstance(json_data[0], dict):
                    metadata['sample_keys'] = list(json_data[0].keys())
            
            # Determine if it's an array or object
            metadata['json_type'] = 'array' if isinstance(json_data, list) else 'object'
            
            return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Error extracting JSON metadata: {e}")
            metadata['is_valid_json'] = False
            return metadata
        except Exception as e:
            logger.error(f"Error extracting JSON metadata: {e}")
            return metadata 