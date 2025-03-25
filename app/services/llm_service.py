"""
LLM service module for handling interactions with language models.
"""
import json
import asyncio
from typing import Any, AsyncGenerator, Dict, Union

import requests
import httpx
from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class LLMService:
    """
    Service for handling interactions with language models.
    """

    def __init__(self):
        """
        Initializes the LLM service with API clients.
        """
        # Base URL for OpenRouter API
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Authorization header for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {settings.OPEN_ROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Keep httpx clients for any custom requests
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info("LLM service initialized with OpenRouter API client")

    async def close(self):
        """
        Closes all API clients.
        """
        await self.http_client.aclose()
        logger.info("LLM service clients closed")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_file_description(self, file_content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """
        Generates a description and metadata for a file using LLM.
        
        Args:
            file_content: The content of the file.
            file_name: The name of the file.
            file_type: The MIME type of the file.
            
        Returns:
            Dict[str, Any]: A dictionary containing the description and metadata.
        """
        try:
            # Truncate file content if it's too long
            max_content_length = 10000
            truncated_content = file_content[:max_content_length]
            if len(file_content) > max_content_length:
                truncated_content += "\n... [content truncated]"
            
            prompt = f"""
            You are analyzing a file to extract useful information. Here are the file details:
            
            File Name: {file_name}
            File Type: {file_type}
            
            File Content:
            {truncated_content}
            
            Please provide:
            1. A concise description of the file (1-2 sentences)
            2. Key metadata about the file content (e.g., topics, entities, dates, etc.)
            
            Format your response as a JSON object with the following structure:
            {{
                "description": "Your concise description here",
                "metadata": {{
                    "topics": ["topic1", "topic2", ...],
                    "entities": ["entity1", "entity2", ...],
                    "key_points": ["point1", "point2", ...],
                    "additional_info": {{ ... any other relevant metadata ... }}
                }}
            }}
            """
            
            # Use the consolidated LLM call function
            response_text = await self._call_llm(
                prompt=prompt,
                model_name="deepseek/deepseek-chat:free",
                stream=False
            )
            
            # Extract JSON from the response
            json_str = self._extract_json_from_text(response_text)
            result = json.loads(json_str)
            
            return result
        except Exception as e:
            logger.error(f"Error generating file description: {e}")
            # Return a basic description if generation fails
            return {
                "description": f"File: {file_name}",
                "metadata": {
                    "topics": [],
                    "entities": [],
                    "key_points": [],
                    "additional_info": {}
                }
            }

    async def _call_llm(
        self, 
        prompt: str, 
        model_name: str = "deepseek/deepseek-chat:free", 
        stream: bool = False,
        temperature: float = 0.4,
        max_tokens: int = 2048
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Consolidated function to call any LLM through OpenRouter using requests.
        
        Args:
            prompt: The prompt to send to the model.
            model_name: The name of the model to use (OpenRouter compatible model name).
            stream: Whether to stream the response.
            temperature: The temperature parameter for the model (controls randomness).
            max_tokens: The maximum number of tokens to generate.
            
        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated response or a stream of tokens.
        """
        try:
            # Set default model if not specified
            if not model_name or model_name.lower() == "gemini":
                model_name = "deepseek/deepseek-chat:free"
                
            # Map some common model aliases to their OpenRouter equivalents
            model_mapping = {
                "anthropic": "anthropic/claude-3.5-sonnet",
                "openai": "openai/gpt-4o-mini",
                "gpt-4": "openai/gpt-4-turbo"
            }
            
            if model_name.lower() in model_mapping:
                model_name = model_mapping[model_name.lower()]
                
            logger.info(f"DEBUG - Using model {model_name} through OpenRouter")
            
            # Prepare messages in the format expected by OpenRouter
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare the payload
            include_reasoning = False
            if model_name.lower() == "deepseek/deepseek-r1:free" or "thinking" in model_name.lower() or "o3" in model_name.lower() or "o1" in model_name.lower():
                include_reasoning = True
            
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "include_reasoning": include_reasoning
            }
            
            # URL for OpenRouter API
            url = f"{self.base_url}/chat/completions"
            
            # Handle streaming vs non-streaming
            if stream:
                logger.info(f"DEBUG - Using streaming mode")
                
                # This method returns an async generator
                async def stream_generator():
                    try:
                        # Use httpx for async HTTP requests
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            async with client.stream('POST', url, headers=self.headers, json=payload) as response:
                                buffer = ""
                                async for chunk in response.aiter_text():
                                    buffer += chunk
                                    while True:
                                        # Find the next complete SSE line
                                        line_end = buffer.find('\n')
                                        if line_end == -1:
                                            break

                                        line = buffer[:line_end].strip()
                                        buffer = buffer[line_end + 1:]

                                        if line.startswith('data: '):
                                            data = line[6:]
                                            if data == '[DONE]':
                                                break

                                            try:
                                                data_obj = json.loads(data)
                                                content = data_obj["choices"][0]["delta"].get("content")
                                                if content:
                                                    logger.info(f"DEBUG - Streaming chunk: {content}")
                                                    yield content
                                                
                                                # Check for reasoning content
                                                reasoning = data_obj["choices"][0]["delta"].get("reasoning")
                                                if reasoning:
                                                    logger.info(f"DEBUG - Streaming reasoning: {reasoning}")
                                                    # Wrap reasoning in a special delimiter for client identification
                                                    yield f"<reasoning>{reasoning}</reasoning>"
                                            except json.JSONDecodeError:
                                                pass
                    except Exception as e:
                        logger.error(f"Error in OpenRouter stream: {e}")
                        yield f"\nError: {str(e)}"
                
                # Return the generator
                return stream_generator()
            else:
                # Non-streaming mode
                logger.info(f"DEBUG - Using non-streaming mode")
                
                # Use requests in a thread to avoid blocking
                response = await asyncio.to_thread(
                    requests.post,
                    url=url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=500, detail=error_msg)
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                logger.info(f"DEBUG - Non-streaming response received")
                return content
                
        except Exception as e:
            logger.error(f"Error calling LLM via OpenRouter: {e}")
            if stream:
                # Return an error generator for streaming
                async def error_generator():
                    yield f"Error from LLM API: {str(e)}"
                return error_generator()
            else:
                # For non-streaming, just raise the exception
                raise

    def _extract_json_from_text(self, text: str) -> str:
        """
        Extracts a JSON object from text that might contain additional content.
        
        Args:
            text: The text containing a JSON object.
            
        Returns:
            str: The extracted JSON string.
        """
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in the text")
        
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    return text[start_idx:end_idx]
        
        raise ValueError("Malformed JSON object in the text")


# Global instance of the LLM service
llm_service = LLMService() 