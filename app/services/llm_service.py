"""
LLM service module for handling interactions with language models.
"""
import os
import re
import json
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import anthropic
import google.generativeai as genai
import httpx
import openai
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
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        # Initialize Gemini client
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Default model, but we'll create specific models as needed
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b')
        # Keep a cache of model instances
        self.gemini_models = {'gemini-1.5-flash-8b': self.gemini_model}
        
        # Keep httpx clients for any custom requests
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        logger.info("LLM service initialized with official API clients")

    async def close(self):
        """
        Closes all API clients.
        """
        await self.http_client.aclose()
        logger.info("LLM service clients closed")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_file_description(self, file_content: str, file_name: str, file_type: str) -> Dict[str, Any]:
        """
        Generates a description and metadata for a file using Gemini.
        
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
            
            # Use the official Gemini API client
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            # Extract JSON from the response
            response_text = response.text
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
        Optimizes a user query for better retrieval results using Gemini.
        
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
            
            # Use the official Gemini API client
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            
            logger.info(f"DEBUG - Gemini response text: '{response.text.strip()}'")
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            # Return the original query if optimization fails
            return query

    async def generate_answer(
        self, 
        query: str, 
        context: List[Dict[str, Any]], 
        model_name: str = "gemini",
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

                Response:
            """
            
            # Log the total prompt length
            logger.info(f"DEBUG - Total prompt length: {len(prompt)} characters")
            logger.info(f"DEBUG - Full prompt including context: {prompt}")
            
            result = None
            
            # Check if model name starts with "gemini" (to handle model versions like "gemini-1.5-flash-8b")
            if model_name.lower().startswith("gemini"):
                if stream:
                    logger.info("DEBUG - Using Gemini streaming")
                    result = await self._stream_gemini(prompt, model_name)
                else:
                    logger.info("DEBUG - Using Gemini non-streaming")
                    result = await self._call_gemini(prompt, model_name)
            elif model_name.lower() == "anthropic":
                if stream:
                    logger.info("DEBUG - Using Anthropic streaming")
                    result = await self._stream_anthropic(prompt)
                else:
                    logger.info("DEBUG - Using Anthropic non-streaming")
                    result = await self._call_anthropic(prompt)
            elif model_name.lower() == "openai":
                if stream:
                    logger.info("DEBUG - Using OpenAI streaming")
                    result = await self._stream_openai(prompt)
                else:
                    logger.info("DEBUG - Using OpenAI non-streaming")
                    result = await self._call_openai(prompt)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
                
            logger.info(f"DEBUG - Result type from generate_answer: {type(result)}")
            return result
        except Exception as e:
            logger.error(f"Error in generate_answer: {e}")
            raise

    async def generate_answer_with_coding_question(self, query: str, model_name: str = "gemini") -> Union[str, AsyncGenerator[str, None]]:
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

        if model_name.lower().startswith("gemini"):
            if stream:
                logger.info("DEBUG - Using Gemini streaming")
                result = await self._stream_gemini(prompt, model_name)
            else:
                logger.info("DEBUG - Using Gemini non-streaming")
                result = await self._call_gemini(prompt, model_name)
        elif model_name.lower() == "anthropic":
            if stream:
                logger.info("DEBUG - Using Anthropic streaming")
                result = await self._stream_anthropic(prompt)
            else:
                logger.info("DEBUG - Using Anthropic non-streaming")
                result = await self._call_anthropic(prompt)
        elif model_name.lower() == "openai":
            if stream:
                logger.info("DEBUG - Using OpenAI streaming")
                result = await self._stream_openai(prompt)
            else:
                logger.info("DEBUG - Using OpenAI non-streaming")
                result = await self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"DEBUG - Result type from generate_answer_with_coding_question: {type(result)}")
        return result

    def _get_gemini_model(self, model_name: str) -> Any:
        """
        Gets or creates a Gemini model instance for the specified model name.
        
        Args:
            model_name: The name of the model to get or create.
            
        Returns:
            Any: The Gemini model instance.
        """
        # Default to gemini-1.5-flash-8b if just "gemini" is specified
        if model_name.lower() == "gemini":
            model_name = "gemini-1.5-flash-8b"
            
        # Create the model if it doesn't exist in our cache
        if model_name not in self.gemini_models:
            logger.info(f"Creating new Gemini model instance for {model_name}")
            try:
                self.gemini_models[model_name] = genai.GenerativeModel(model_name)
            except Exception as e:
                logger.error(f"Error creating Gemini model {model_name}: {e}")
                # Fall back to default model
                logger.info(f"Falling back to default model gemini-1.5-flash-8b")
                model_name = "gemini-1.5-flash-8b"
                
        return self.gemini_models[model_name]
        
    async def _call_gemini(self, prompt: str, model_name: str = "gemini") -> str:
        """
        Calls the Gemini API to generate a response.
        
        Args:
            prompt: The prompt to send to the model.
            model_name: The name of the model to use.
            
        Returns:
            str: The generated response.
        """
        try:
            # Get the appropriate model instance
            model = self._get_gemini_model(model_name)
            
            # Use the official Gemini API client
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise

    async def _stream_gemini(self, prompt: str, model_name: str = "gemini") -> AsyncGenerator[str, None]:
        """
        Streams responses from the Gemini API.
        
        Args:
            prompt: The prompt to send to the model.
            model_name: The name of the model to use.
            
        Yields:
            str: Chunks of the generated response.
        """
        try:
            # This method should return an async generator, not execute it
            # Create and return an async generator that will produce chunks on demand
            
            async def stream_generator():
                # Get the appropriate model instance
                model = self._get_gemini_model(model_name)
                logger.info(f"DEBUG - _stream_gemini using model: {model_name}")
                
                try:
                    # Use the official Gemini API client with streaming
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.4,
                            top_p=0.95,
                            top_k=40,
                            max_output_tokens=2048,
                        ),
                        stream=True
                    )
                    
                    # Process the streaming response
                    # Handle the response depending on its type
                    if hasattr(response, '__iter__') and not hasattr(response, '__aiter__'):
                        # It's a synchronous iterator, not an async iterator
                        for chunk in response:
                            if hasattr(chunk, 'text'):
                                yield chunk.text
                            await asyncio.sleep(0)  # Allow other tasks to run
                    else:
                        # It should be an async iterator
                        async for chunk in response:
                            if hasattr(chunk, 'text'):
                                yield chunk.text
                            await asyncio.sleep(0)  # Allow other tasks to run
                    
                except Exception as e:
                    logger.error(f"Error in stream_generator: {e}")
                    yield f"Error: {str(e)}"
            
            # Return the generator without awaiting it
            return stream_generator()
        except Exception as e:
            logger.error(f"Error setting up Gemini streaming: {e}")
            # Return a generator that just yields the error
            async def error_generator():
                yield f"Error streaming from Gemini API: {str(e)}"
            return error_generator()

    async def _call_anthropic(self, prompt: str) -> str:
        """
        Calls the Anthropic API to generate a response.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            str: The generated response.
        """
        try:
            # Use the official Anthropic API client
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=2048,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise

    async def _stream_anthropic(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Streams responses from the Anthropic API.
        
        Args:
            prompt: The prompt to send to the model.
            
        Yields:
            str: Chunks of the generated response.
        """
        try:
            # This method should return an async generator, not execute it
            async def stream_generator():
                try:
                    # Use the official Anthropic API client with streaming
                    with self.anthropic_client.messages.stream(
                        model="claude-3-opus-20240229",
                        max_tokens=2048,
                        temperature=0.4,
                        messages=[{"role": "user", "content": prompt}]
                    ) as stream:
                        # Process the streaming response
                        for chunk in stream:
                            if chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
                                yield chunk.delta.text
                            await asyncio.sleep(0)  # Allow other tasks to run
                    
                except Exception as e:
                    logger.error(f"Error in Anthropic stream_generator: {e}")
                    yield f"\nError: {str(e)}"
            
            # Return the generator without awaiting it
            return stream_generator()
        except Exception as e:
            logger.error(f"Error setting up Anthropic streaming: {e}")
            # Return a generator that just yields the error
            async def error_generator():
                yield f"Error setting up Anthropic stream: {str(e)}"
            return error_generator()

    async def _call_openai(self, prompt: str) -> str:
        """
        Calls the OpenAI API to generate a response.
        
        Args:
            prompt: The prompt to send to the model.
            
        Returns:
            str: The generated response.
        """
        try:
            # Use the official OpenAI API client
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    async def _stream_openai(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Streams responses from the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model.
            
        Yields:
            str: Chunks of the generated response.
        """
        try:
            # This method should return an async generator, not execute it
            async def stream_generator():
                try:
                    # Use the official OpenAI API client with streaming
                    response = await asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model="gpt-4-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                        max_tokens=2048,
                        stream=True
                    )
                    
                    # Process the streaming response
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            yield content
                        await asyncio.sleep(0)  # Allow other tasks to run
                    
                except Exception as e:
                    logger.error(f"Error in OpenAI stream_generator: {e}")
                    yield f"\nError: {str(e)}"
            
            # Return the generator without awaiting it
            return stream_generator()
        except Exception as e:
            logger.error(f"Error setting up OpenAI streaming: {e}")
            # Return a generator that just yields the error
            async def error_generator():
                yield f"Error setting up OpenAI stream: {str(e)}"
            return error_generator()
    
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