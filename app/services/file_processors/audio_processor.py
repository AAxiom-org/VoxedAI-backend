"""
Audio processor module for extracting text from audio files.
"""
import os
import tempfile
import subprocess
from typing import Dict, Any

import whisper

from app.core.logging import logger
from app.services.file_processors import FileProcessor


class AudioProcessor(FileProcessor):
    """
    Processor for audio files using Whisper for transcription.
    Handles: audio/mpeg, audio/wav, audio/mp4, audio/webm
    """
    
    def __init__(self, model_size: str = "tiny"):
        """
        Initialize the audio processor.
        
        Args:
            model_size: Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model = None
        self.model_size = model_size
        
    async def process(self, file_content: bytes, file_path: str) -> str:
        """
        Process audio content and transcribe it to text.
        
        Args:
            file_content: Raw audio file content bytes
            file_path: Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Lazy load the model
            if self.model is None:
                self.model = whisper.load_model(self.model_size)
            
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as input_file, \
                 tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                
                # Write content to temp file
                input_file.write(file_content)
                input_file.flush()
                
                input_path = input_file.name
                output_path = output_file.name
            
            # Convert audio to format suitable for Whisper
            self._convert_audio(input_path, output_path)
            
            # Transcribe the audio
            result = self.model.transcribe(output_path)
            
            # Clean up temporary files
            os.unlink(input_path)
            os.unlink(output_path)
            
            return result["text"]
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise
    
    async def get_metadata(self, file_content: bytes, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file.
        
        Args:
            file_content: Raw audio file content bytes
            file_path: Path to the audio file
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Create a temporary file to analyze
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name
            
            try:
                # Use ffprobe to get audio metadata
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    temp_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                if result.stdout:
                    import json
                    metadata = json.loads(result.stdout)
                    
                    # Extract relevant metadata
                    extracted_data = {}
                    
                    if 'format' in metadata:
                        format_data = metadata['format']
                        extracted_data.update({
                            'duration': format_data.get('duration'),
                            'bit_rate': format_data.get('bit_rate'),
                            'format_name': format_data.get('format_name'),
                        })
                    
                    if 'streams' in metadata and len(metadata['streams']) > 0:
                        audio_stream = next((s for s in metadata['streams'] if s.get('codec_type') == 'audio'), None)
                        if audio_stream:
                            extracted_data.update({
                                'codec': audio_stream.get('codec_name'),
                                'channels': audio_stream.get('channels'),
                                'sample_rate': audio_stream.get('sample_rate'),
                            })
                    
                    return extracted_data
                
                return {}
            except Exception as e:
                # Clean up the temporary file in case of error
                os.unlink(temp_path)
                logger.error(f"Error extracting audio metadata: {e}")
                return {}
    
    def _convert_audio(self, input_file: str, output_file: str) -> None:
        """
        Convert audio to a format suitable for Whisper (16kHz mono WAV).
        
        Args:
            input_file: Path to the input audio file
            output_file: Path to save the converted audio
        """
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ac', '1',  # Mono
            '-ar', '16000',  # 16kHz
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting audio: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr.decode() if e.stderr else 'None'}")
            raise 