"""
Video processor module for extracting content from video files.
"""

### TODO: Add frame extraction and analysis using Gemini for paid users only

import io
import os
import tempfile
import subprocess
import asyncio
import uuid
from typing import Dict, Any, List

import cv2
import numpy as np
import whisper
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.services.file_processors import FileProcessor


class VideoProcessor(FileProcessor):
    """
    Processor for video files.
    Handles: video/mp4, video/webm
    """
    
    def __init__(self, model_size: str = "tiny", frame_interval: int = 10):
        """
        Initialize the video processor.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            frame_interval: Interval between frames to extract (in seconds)
        """
        self.whisper_model = None
        self.model_size = model_size
        self.frame_interval = frame_interval
    
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process video content and extract audio transcription and frame descriptions.
        
        Args:
            file_content: Raw video file content bytes
            file_path: Path to the video file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as video_file, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
                
                # Write content to temp file
                video_file.write(file_content)
                video_file.flush()
                
                video_path = video_file.name
                audio_path = audio_file.name
            
            # Extract audio from video
            self._extract_audio(video_path, audio_path)
            
            # Transcribe audio
            audio_text = await self._transcribe_audio(audio_path)
            
            # Extract and analyze key frames
            frame_texts = await self._extract_and_analyze_frames(video_path)
            
            # Combine results
            result_parts = []
            
            if audio_text:
                result_parts.append("## Audio Transcription\n")
                result_parts.append(audio_text)
                result_parts.append("\n")
            
            if frame_texts:
                result_parts.append("## Frame Analysis\n")
                for i, text in enumerate(frame_texts, 1):
                    result_parts.append(f"### Frame {i}\n")
                    result_parts.append(text)
                    result_parts.append("\n")
            
            # Clean up temporary files
            os.unlink(video_path)
            os.unlink(audio_path)
            
            return "\n".join(result_parts)
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            raise
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from video file.
        
        Args:
            file_content: Raw video file content bytes
            file_path: Path to the video file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        base_metadata = {
            'file_size': len(file_content),
            'file_extension': os.path.splitext(file_path)[1],
        }
        
        try:
            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                temp_path = temp_file.name
            
            # Use ffprobe to extract video metadata
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout:
                import json
                metadata = json.loads(result.stdout)
                
                # Extract relevant metadata
                video_metadata = {}
                
                if 'format' in metadata:
                    format_data = metadata['format']
                    video_metadata.update({
                        'duration': format_data.get('duration'),
                        'bit_rate': format_data.get('bit_rate'),
                        'format_name': format_data.get('format_name'),
                    })
                
                if 'streams' in metadata:
                    # Find video stream
                    video_stream = next((s for s in metadata['streams'] if s.get('codec_type') == 'video'), None)
                    if video_stream:
                        video_metadata.update({
                            'width': video_stream.get('width'),
                            'height': video_stream.get('height'),
                            'codec': video_stream.get('codec_name'),
                            'frame_rate': eval(video_stream.get('r_frame_rate', '0/1')) if '/' in video_stream.get('r_frame_rate', '0/1') else video_stream.get('r_frame_rate'),
                            'bit_depth': video_stream.get('bits_per_raw_sample'),
                        })
                    
                    # Find audio stream
                    audio_stream = next((s for s in metadata['streams'] if s.get('codec_type') == 'audio'), None)
                    if audio_stream:
                        video_metadata.update({
                            'audio_codec': audio_stream.get('codec_name'),
                            'audio_channels': audio_stream.get('channels'),
                            'audio_sample_rate': audio_stream.get('sample_rate'),
                        })
                
                # Add to base metadata
                base_metadata.update(video_metadata)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return base_metadata
        except Exception as e:
            logger.error(f"Error extracting video metadata: {e}")
            return base_metadata
    
    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to save the extracted audio
        """
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz
            '-vn',  # No video
            '-y',  # Overwrite output file if it exists
            audio_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio from video: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise
    
    async def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Lazy load the model
            if self.whisper_model is None:
                self.whisper_model = whisper.load_model(self.model_size)
            
            # Transcribe the audio
            result = self.whisper_model.transcribe(audio_path)
            
            return result["text"]
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    async def _extract_and_analyze_frames(self, video_path: str) -> List[str]:
        """
        Extract key frames from video and analyze them.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List[str]: List of frame descriptions
        """
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Determine frame positions to extract
            frame_interval_frames = int(self.frame_interval * fps)
            frame_positions = []
            
            current_pos = 0
            while current_pos < total_frames:
                frame_positions.append(current_pos)
                current_pos += frame_interval_frames
            
            # Limit the number of frames to analyze (to avoid too many API calls)
            max_frames = 5
            if len(frame_positions) > max_frames:
                # Distribute frames evenly
                step = len(frame_positions) // max_frames
                frame_positions = frame_positions[::step][:max_frames]
            
            # Extract and analyze frames
            frame_descriptions = []
            
            for pos in frame_positions:
                # Set position and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Analyze frame
                description = await self._analyze_frame(img, pos / fps)
                if description:
                    frame_descriptions.append(description)
            
            cap.release()
            
            return frame_descriptions
        except Exception as e:
            logger.error(f"Error extracting and analyzing frames: {e}")
            return []
    
    async def _analyze_frame(self, frame: Image.Image, timestamp: float) -> str:
        """
        Analyze a video frame using Google Gemini.
        
        Args:
            frame: PIL Image object of the frame
            timestamp: Timestamp of the frame in seconds
            
        Returns:
            str: Frame description
        """
        try:
            from google import genai
            import os
            
            # Initialize the Gemini client
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Convert PIL Image to bytes for the API
            with io.BytesIO() as output:
                frame.save(output, format="PNG")
                image_bytes = output.getvalue()
            
            # Create the request parts
            from google.genai import types
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            
            # Prompt to analyze the frame
            prompt = "Describe what's happening in this video frame in detail. Include any visible text."
            
            # Generate content using Gemini
            response = client.models.generate_content(
                model="gemini-1.5-flash-8b",
                contents=[prompt, image_part]
            )
            
            # Format the response with timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            return f"Time: {timestamp_str}\n\n{response.text}"
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return "" 