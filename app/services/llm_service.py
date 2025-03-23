"""
LLM service module for handling interactions with language models.
"""
import os
import re
import json
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def optimize_query(self, query: str) -> str:
        """
        Optimizes a user query for better retrieval results.
        
        Args:
            query: The original user query.
            
        Returns:
            str: The optimized query.
        """
        try:
            # Debug info about the input query
            logger.info(f"DEBUG - optimize_query input: '{query}'")
            
            # If the query already contains streaming data, don't try to optimize it
            if query.strip().startswith('data:'):
                logger.info(f"DEBUG - Query appears to contain streaming data, returning original query")
                return query
                
            prompt = f"""
            You are helping to optimize a search query for a retrieval system. Your task is to rewrite the query to make it more effective for semantic search.
            
            Original Query: {query}
            
            Please:
            1. Expand any abbreviations or acronyms
            2. Add relevant synonyms or related terms
            3. Make the query more specific and detailed
            4. Remove any unnecessary words or phrases
            
            Return ONLY the optimized query text, with no additional explanation or formatting.
            """
            
            # Use the consolidated LLM call function
            response_text = await self._call_llm(
                prompt=prompt,
                model_name="deepseek/deepseek-chat:free",
                stream=False
            )
            
            logger.info(f"DEBUG - LLM response text: '{response_text.strip()}'")
            
            return response_text.strip()
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            # Return the original query if optimization fails
            return query

    async def generate_answer(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        model_name: str = "deepseek/deepseek-chat:free",
        stream: bool = False,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates an answer to a query using the specified LLM model.
        
        Args:
            query: The user query.
            context: The retrieved context documents.
            model_name: The name of the LLM model to use.
            stream: Whether to stream the response.
        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated answer or a stream of tokens.
        """
        try:
            logger.info(f"DEBUG - Generate answer with model={model_name}, stream={stream}")
            logger.info(f"DEBUG - Received {len(context)} context documents")
            
            # Format context for the prompt
            formatted_context_parts = []
            
            for i, doc in enumerate(context):
                # Get the source/filename
                source = doc.get("source", "Unknown file")
                logger.info(f"DEBUG - Source: {source}")
                
                # Get text content
                text_content = doc.get("text", "No content available")
                
                # Get metadata
                metadata = doc.get("metadata", {})
                
                # Get description from metadata if available
                if source.endswith(".json"):
                    description = metadata.get("description", "This is a notes file that the user wrote. The user has written this note themselves.")
                else:
                    description = metadata.get("description", "No description available")
                
                # Process additional_info if it's a JSON string
                additional_info_str = ""
                if "additional_info" in metadata and metadata["additional_info"]:
                    try:
                        # Try to parse it as JSON if it's a string
                        if isinstance(metadata["additional_info"], str) and metadata["additional_info"].startswith("{"):
                            additional_info = json.loads(metadata["additional_info"])
                            # Format key fields from additional_info
                            info_parts = []
                            for key, value in additional_info.items():
                                if value:  # Only include non-empty values
                                    info_parts.append(f"{key}: {value}")
                            if info_parts:
                                additional_info_str = "\nAdditional Info:\n" + "\n".join(info_parts)
                    except json.JSONDecodeError:
                        # If not valid JSON, just use as is
                        additional_info_str = f"\nAdditional Info: {metadata['additional_info']}"
                
                # Get entities if available
                entities_str = ""
                if "entities" in metadata and metadata["entities"]:
                    if isinstance(metadata["entities"], list):
                        entities_str = f"\nEntities: {', '.join(metadata['entities'])}"
                    else:
                        entities_str = f"\nEntities: {metadata['entities']}"
                
                # Get key points if available
                key_points_str = ""
                if "key_points" in metadata and metadata["key_points"]:
                    if isinstance(metadata["key_points"], list):
                        key_points_str = f"\nKey Points:\n- " + "\n- ".join(metadata["key_points"])
                    else:
                        key_points_str = f"\nKey Points: {metadata['key_points']}"
                
                # Get topics if available
                topics_str = ""
                if "topics" in metadata and metadata["topics"]:
                    if isinstance(metadata["topics"], list):
                        topics_str = f"\nTopics: {', '.join(metadata['topics'])}"
                    else:
                        topics_str = f"\nTopics: {metadata['topics']}"
                
                # Combine all metadata
                metadata_str = f"{additional_info_str}{entities_str}{key_points_str}{topics_str}"
                
                # Format this document with clear section headers
                formatted_doc = f"""
                    File #{i+1}: {source}
                    Description: {description}{metadata_str}
                    Content:
                    {text_content}
                """
                formatted_context_parts.append(formatted_doc)
                
                # Log what we're including for debugging
                logger.info(f"DEBUG - Added document {i+1}: {source}, description: {description[:50]}...")
            
            # Join all formatted documents
            formatted_context = "\n" + "\n".join(formatted_context_parts)
            
            # Log the total formatted context length
            logger.info(f"DEBUG - Total formatted context length: {len(formatted_context)} characters")
            
            prompt = f"""
               You are an AI assistant helping to answer questions based on provided context, but also capable of responding to direct requests.


               Context:
               {formatted_context}


               User request:
               {query}


               Guidelines:
               1. Always prioritize directly addressing the user's request first.
               2. If the request is a direct instruction (like "Tell a story" or "Write a poem"), fulfill it to the best of your ability regardless of the context.
               3. Only when the request is a question seeking information should you primarily rely on the provided context.
               4. When using the context to answer questions, reference specific sources when applicable (e.g., "According to [filename]...").
               5. If the context is irrelevant to the request, simply fulfill the request using your general knowledge and abilities.
               6. Maintain a helpful, informative, and friendly tone.
               7. Format all responses in beautiful and informative markdown.
               8. Use $...$ for inline LaTeX expressions. Example: The formula $E = mc^2$ shows the relationship between energy and mass.
               9. Use $$...$$ for block/display LaTeX expressions, with each formula on its own line. Example:
               $$
               F = G \frac{{m_1 m_2}}{{r^2}}
               $$
               10. For multi-line equations, use the align environment within block delimiters:
               $$
               \begin{{align*}}
               y &= mx + b \\
               &= 2x + 3
               \end{{align*}}
               $$
               11. Ensure all special characters in LaTeX expressions are properly escaped with a single backslash.
               12. Format step-by-step mathematical solutions with clear delineation between text explanation and LaTeX formulas.
               13. When boxing final answers, use the \boxed{{}} command within a display math environment:
               $$
               \boxed{{x = \frac{{-b \pm \sqrt{{b^2 - 4ac}}}}{{2a}}}}
               $$


               Response:
           """
            
            # Log the total prompt length
            logger.info(f"DEBUG - Total prompt length: {len(prompt)} characters")
            logger.info(f"DEBUG - Full prompt including context: {prompt}")
            
            # Use the consolidated LLM call function
            return await self._call_llm(prompt=prompt, model_name=model_name, stream=stream)
            
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            raise

    async def generate_answer_with_coding_question(self, query: str, model_name: str = "deepseek/deepseek-chat:free") -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates an answer to a coding question using the specified LLM model.
        
        Args:
            query: The user query.
            model_name: The name of the LLM model to use.
            
        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated answer or a stream of tokens.
        """
        stream = True

        prompt = f"""
            You are an AI teaching assistant in a coding education platform. Your primary role is to guide students through their learning journey rather than simply providing answers.

            User query, code, and output: 
            {query}

            Teaching Guidelines:
            1. Foster learning through guided discovery rather than direct correction.
            2. When reviewing code:
            - Ask thoughtful questions that help the student discover issues themselves
            - Suggest improvement areas as learning opportunities
            - Point out potential concerns by explaining relevant concepts
            - Reference specific code lines when discussing concepts (e.g., "Looking at line [line number]...")
            3. Balance encouragement with constructive feedback.
            4. Only provide direct code corrections if explicitly requested.
            5. When concepts arise, briefly explain the underlying principles to deepen understanding.
            6. If the query is unrelated to the provided code, respond appropriately while maintaining an educational tone.
            7. Use the Socratic method where appropriate - guide through questions rather than simply providing answers.

            Remember that your goal is to develop the student's problem-solving skills and coding intuition, not just to fix their immediate issues.

            Response:
        """

        # Use the consolidated LLM call function
        return await self._call_llm(prompt=prompt, model_name=model_name, stream=stream)

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