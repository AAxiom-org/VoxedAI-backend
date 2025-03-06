"""
Text processor module for extracting text from plain text and markdown files.
"""

import os
import re
import io
import base64
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
        try:
            text = await self.process(file_content, file_path)
            
            # Basic text analysis for metadata
            line_count = len(text.splitlines())
            word_count = len(text.split())
            char_count = len(text)
            
            return {
                'line_count': line_count,
                'word_count': word_count,
                'char_count': char_count,
                'file_size': len(file_content),
                'file_extension': os.path.splitext(file_path)[1],
            }
        except Exception as e:
            logger.error(f"Error extracting text metadata: {e}")
            return {}
    
    async def _extract_text_and_description_from_image(self, image: Image.Image) -> str:
        """
        Extract text and generate description from an image using Google Gemini.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Extracted text and generated description
        """
        try:
            from google import genai
            import os
            
            # Initialize the Gemini client
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Convert PIL Image to bytes for the API
            with io.BytesIO() as output:
                image.save(output, format="PNG")
                image_bytes = output.getvalue()
            
            # Create the request parts
            from google.genai import types
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            
            combined_prompt = """
                1. Extract all text visible in this image. If no text is visible, respond with 'No text detected.'
                2. Provide a detailed description of what's in this image.
            """

            response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=[combined_prompt, image_part]
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error extracting text and description from image: {e}")
            return "Error processing image content."


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
        Process Markdown content and convert to plain text.
        
        Args:
            file_content: Raw Markdown file content bytes
            file_path: Path to the Markdown file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Decode bytes to string
            md_text = file_content.decode('utf-8')
            
            # Extract images from markdown before converting to HTML
            # This will find both URL and Base64 embedded images
            image_matches = await self._extract_markdown_images(md_text)
            
            # Process images and get their content
            image_contents = []
            for alt_text, image_src, position in image_matches:
                # Skip file path images (local files)
                if self._is_local_file_path(image_src):
                    continue
                
                try:
                    # Process URL or Base64 image
                    img = await self._load_image_from_src(image_src)
                    if img:
                        img_content = await self._extract_text_and_description_from_image(img)
                        if img_content:
                            caption = f"Image: {alt_text}" if alt_text else "Image"
                            image_contents.append((f"[{caption}]\n{img_content}", position))
                except Exception as e:
                    logger.error(f"Error processing markdown image: {e}")
            
            # Convert markdown to HTML
            html = markdown.markdown(md_text)
            
            # Basic HTML to text conversion
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            
            # Handle special HTML entities
            text = text.replace('&nbsp;', ' ')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            text = text.replace('&amp;', '&')
            text = text.replace('&quot;', '"')
            
            # Split text into lines for easier insertion of image content
            text_lines = text.splitlines()
            
            # Insert image content at appropriate positions
            result_lines = []
            current_line = 0
            
            # Sort image contents by position
            image_contents.sort(key=lambda x: x[1])
            
            for img_content, position in image_contents:
                # Add text lines up to the image position
                while current_line < position and current_line < len(text_lines):
                    if text_lines[current_line].strip():
                        result_lines.append(text_lines[current_line])
                    current_line += 1
                
                # Add image content
                result_lines.append("\n[IMAGE CONTENT START]")
                result_lines.append(img_content)
                result_lines.append("[IMAGE CONTENT END]\n")
            
            # Add remaining text lines
            while current_line < len(text_lines):
                if text_lines[current_line].strip():
                    result_lines.append(text_lines[current_line])
                current_line += 1
            
            # Join lines and normalize whitespace
            return "\n".join(result_lines)
        except Exception as e:
            logger.error(f"Error processing markdown file: {e}")
            # Fallback to base text processor if markdown processing fails
            return await super().process(file_content, file_path)
    
    async def _extract_markdown_images(self, md_text: str) -> List[Tuple[str, str, int]]:
        """
        Extract images from markdown text.
        
        Args:
            md_text: Markdown text
            
        Returns:
            List[Tuple[str, str, int]]: List of tuples containing (alt_text, image_src, line_position)
        """
        images = []
        
        # Match both inline and reference images in markdown
        # Inline: ![alt text](image_url)
        # Reference: ![alt text][ref] ... [ref]: image_url
        
        # Find inline images
        inline_pattern = r'!\[(.*?)\]\((.*?)\)'
        for match in re.finditer(inline_pattern, md_text):
            alt_text = match.group(1)
            image_src = match.group(2)
            
            # Calculate line position
            line_position = md_text[:match.start()].count('\n')
            images.append((alt_text, image_src, line_position))
        
        # Find reference images
        ref_pattern = r'!\[(.*?)\]\[(.*?)\]'
        ref_def_pattern = r'\[(.*?)\]:\s*(.*?)(\s+["\'](.*?)["\'])?$'
        
        # Extract reference definitions
        ref_defs = {}
        for match in re.finditer(ref_def_pattern, md_text, re.MULTILINE):
            ref_id = match.group(1)
            ref_url = match.group(2)
            ref_defs[ref_id] = ref_url
        
        # Match reference images
        for match in re.finditer(ref_pattern, md_text):
            alt_text = match.group(1)
            ref_id = match.group(2)
            
            # If ref_id is empty, use alt_text as the reference
            if not ref_id:
                ref_id = alt_text
            
            # Look up the reference
            if ref_id in ref_defs:
                image_src = ref_defs[ref_id]
                line_position = md_text[:match.start()].count('\n')
                images.append((alt_text, image_src, line_position))
        
        return images
    
    def _is_local_file_path(self, src: str) -> bool:
        """
        Check if the image source is a local file path.
        
        Args:
            src: Image source
            
        Returns:
            bool: True if the source is a local file path
        """
        # Check if it's a URL
        parsed = urlparse(src)
        if parsed.scheme and parsed.netloc:
            return False
        
        # Check if it's a Base64 embedded image
        if src.startswith('data:image/'):
            return False
        
        # Otherwise, assume it's a local file path
        return True
    
    async def _load_image_from_src(self, src: str) -> Image.Image:
        """
        Load an image from a URL or Base64 string.
        
        Args:
            src: Image source (URL or Base64 string)
            
        Returns:
            Image.Image: PIL Image object
        """
        try:
            # Handle Base64 embedded images
            if src.startswith('data:image/'):
                # Extract the Base64 data
                header, data = src.split(',', 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data))
            
            # Handle URL images
            parsed = urlparse(src)
            if parsed.scheme and parsed.netloc:
                response = requests.get(src, timeout=10)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            
            return None
        except Exception as e:
            logger.error(f"Error loading image from source: {e}")
            return None
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from Markdown file.
        
        Args:
            file_content: Raw Markdown file content bytes
            file_path: Path to the Markdown file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        base_metadata = await super().get_metadata(file_content, file_path)
        
        try:
            # Decode bytes to string
            md_text = file_content.decode('utf-8')
            
            # Try to extract front matter if present (YAML between --- lines)
            import re
            front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', md_text, re.DOTALL)
            
            if front_matter_match:
                try:
                    import yaml
                    front_matter = yaml.safe_load(front_matter_match.group(1))
                    if isinstance(front_matter, dict):
                        base_metadata['front_matter'] = front_matter
                except Exception:
                    pass
            
            # Count headings by level
            heading_counts = {}
            for i in range(1, 7):
                pattern = r'^#{' + str(i) + r'}\s+.+$'
                headings = re.findall(pattern, md_text, re.MULTILINE)
                if headings:
                    heading_counts[f'h{i}_count'] = len(headings)
            
            if heading_counts:
                base_metadata['heading_counts'] = heading_counts
            
            # Count images by type
            image_matches = await self._extract_markdown_images(md_text)
            
            url_image_count = 0
            base64_image_count = 0
            local_image_count = 0
            
            for _, image_src, _ in image_matches:
                if image_src.startswith('data:image/'):
                    base64_image_count += 1
                elif not self._is_local_file_path(image_src):
                    url_image_count += 1
                else:
                    local_image_count += 1
            
            if url_image_count or base64_image_count or local_image_count:
                base_metadata['image_counts'] = {
                    'url_images': url_image_count,
                    'base64_images': base64_image_count,
                    'local_images': local_image_count,
                    'total_images': len(image_matches)
                }
            
            return base_metadata
        except Exception as e:
            logger.error(f"Error extracting markdown metadata: {e}")
            return base_metadata 