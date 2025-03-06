"""
Image processor module for extracting text and descriptions from image files.
"""
import io
import os
from typing import Dict, Any

from PIL import Image
import PIL.ExifTags

from app.core.config import settings
from app.core.logging import logger
from app.services.file_processors import FileProcessor


class ImageProcessor(FileProcessor):
    """
    Processor for image files.
    Handles: image/jpeg, image/png, image/gif, image/webp
    """
    
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process image content and extract text and description.
        
        Args:
            file_content: Raw image file content bytes
            file_path: Path to the image file
            
        Returns:
            str: Extracted text content and description
        """
        try:
            # Load the image
            img = Image.open(io.BytesIO(file_content))
            
            # Use Gemini to extract text and generate description
            return await self._extract_text_and_description(img)
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            raise
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from image file.
        
        Args:
            file_content: Raw image file content bytes
            file_path: Path to the image file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        base_metadata = {
            'file_size': len(file_content),
            'file_extension': os.path.splitext(file_path)[1],
        }
        
        try:
            # Load the image
            img = Image.open(io.BytesIO(file_content))
            
            # Basic image properties
            metadata = {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
            }
            
            # Try to extract EXIF data if available
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = {
                    PIL.ExifTags.TAGS[k]: v
                    for k, v in img._getexif().items()
                    if k in PIL.ExifTags.TAGS
                }
                
                # Filter out binary data and select common EXIF fields
                for key in ['Make', 'Model', 'DateTime', 'ExposureTime', 'FNumber', 
                            'ISOSpeedRatings', 'FocalLength', 'Flash']:
                    if key in exif and isinstance(exif[key], (str, int, float)):
                        exif_data[key] = exif[key]
                
                if exif_data:
                    metadata['exif'] = exif_data
            
            # Add to base metadata
            base_metadata.update(metadata)
            
            return base_metadata
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
            return base_metadata
    
    async def _extract_text_and_description(self, image: Image.Image) -> str:
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
            
            # First, try to extract any text in the image
            text_prompt = "Extract all text visible in this image. If no text is visible, respond with 'No text detected.'"
            text_response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=[text_prompt, image_part]
            )
            
            # Then, generate a description of the image content
            desc_prompt = "Provide a detailed description of what's in this image."
            desc_response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=[desc_prompt, image_part]
            )
            
            # Combine both responses
            result = []
            if text_response.text and text_response.text != "No text detected.":
                result.append(f"Extracted Text:\n{text_response.text}\n")
            
            result.append(f"Image Description:\n{desc_response.text}")
            
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Error extracting text and description from image: {e}")
            return "Error processing image content." 