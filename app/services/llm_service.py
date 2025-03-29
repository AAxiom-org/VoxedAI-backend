"""
LLM service module for handling interactions with language models.
"""
import json
import asyncio
import yaml
from typing import Any, AsyncGenerator, Dict, Union, List

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
                logger.debug(f"Response data: {data}")
                
                # Check if the expected keys exist in the response
                if "choices" not in data:
                    error_msg = f"Unexpected response format from OpenRouter. Missing 'choices' key. Response: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if not data["choices"] or "message" not in data["choices"][0]:
                    error_msg = f"Unexpected response format from OpenRouter. Empty choices or missing 'message'. Response: {data}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
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

    async def get_decision(
        self, 
        query: str, 
        context: Dict[str, Any] = None, 
        action_history: List[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Uses google/gemini-2.0-flash-001 to decide the next action in the agent workflow.
        
        Args:
            query: The user's question or command
            context: Any context gathered so far
            action_history: Previous actions taken by the agent
            stream: Whether to stream the response
            
        Returns:
            Dict[str, Any]: A decision object containing action and parameters
        """
        try:
            # Format previous actions for the prompt
            history_text = ""
            if action_history:
                for i, action in enumerate(action_history):
                    history_text += f"Action {i+1}: {action.get('action', 'unknown')} with parameters {action.get('parameters', {})}\n"
            
            # Format context for the prompt
            context_text = ""
            if context:
                for key, value in context.items():
                    if isinstance(value, str) and len(value) > 500:
                        value = value[:500] + "... (truncated)"
                    context_text += f"{key}: {value}\n"
            
            prompt = f"""
                You are a decision-making component in an AI agent workflow. Your job is to decide the next action based on the user query and current context. **Only route to an action when additional information or an operation is explicitly necessary.** For simple queries or interactions (e.g., greetings, basic chat, or queries clearly answerable from existing knowledge), immediately proceed to finish.

                ## USER QUERY
                {query}

                ## CURRENT CONTEXT
                {context_text or "No context available yet."}

                ## ACTION HISTORY
                {history_text or "No previous actions taken."}

                ## AVAILABLE ACTIONS
                You can choose from these actions only if clearly required:

                1. **rag** - Use Retrieval Augmented Generation to fetch relevant context  
                **When to use:** You need specific information from the user's files, notes, or research to accurately respond.

                2. **tool** - Use a specialized tool from the tool shed  
                **When to use:** You must perform a particular operation (e.g., file reading, note editing, code execution, data analysis).
                - Current tools available:
                    - file_interaction: Read all files in the workspace, edit note files, and create new note files

                3. **finish** - Generate the final answer using the context and knowledge you already possess  
                **When to use:** You already have sufficient information or the query does not require external resources or operations.

                ## DECISION INSTRUCTIONS
                1. Consider carefully if external context or specialized operations are genuinely needed.
                2. For basic interactions (such as greetings or casual conversation), choose **finish** directly without additional actions.
                3. Clearly articulate your decision-making process step-by-step.

                Return your decision in this YAML format:

                ```yaml
                thinking: |
                    <your step-by-step reasoning>
                action: <action_name>
                parameters:
                    <parameter_name>: <parameter_value>
                ```
            """
            
            # Use the gemini-2.0-flash-001 model for making decisions
            # Always use non-streaming for the decision to ensure we get a complete response
            # We'll handle exposing the thinking separately
            response_text = await self._call_llm(
                prompt=prompt,
                model_name="google/gemini-2.0-flash-001",
                stream=False,  # Decision node always uses non-streaming
                temperature=0,
                max_tokens=1024
            )
            
            # Extract YAML from the response
            yaml_text = self._extract_yaml_from_text(response_text)
            
            try:
                logger.info(f"Decision: {yaml.safe_load(yaml_text)}")
                decision = yaml.safe_load(yaml_text)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML from Gemini response: {e}")
                logger.error(f"Raw YAML content that failed to parse: {yaml_text}")
                
                # Create a fallback decision with the model's raw response as thinking
                return {
                    "action": "finish",
                    "thinking": f"Error parsing model response as YAML. Raw response: {response_text}",
                    "parameters": {
                        "reason": f"YAML parsing error: {str(e)}"
                    }
                }
            
            # Validate decision format
            if not isinstance(decision, dict):
                raise ValueError("Decision must be a dictionary")
            
            if "action" not in decision:
                raise ValueError("Decision must include 'action'")
            
            if decision["action"] not in ["rag", "tool", "finish"]:
                raise ValueError(f"Invalid action '{decision['action']}'. Must be one of: rag, tool, finish")
            
            if "parameters" not in decision:
                decision["parameters"] = {}
            
            # Make sure the thinking is included
            if "thinking" not in decision:
                decision["thinking"] = "No detailed reasoning provided."
                
            # Log the decision and thinking
            logger.info(f"Decision: {decision['action']} with parameters {decision['parameters']}")
            logger.info(f"Thinking: {decision['thinking'][:100]}...")
            
            # If streaming is requested, immediately stream the thinking part to the client
            if stream:
                logger.info("Streaming thinking to client")
                # Note: The thinking content is preserved in the decision object
                # It will be extracted and sent to the client by the agent flow
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting decision from Gemini: {e}")
            # Default to finish action if there's an error
            return {
                "action": "finish",
                "thinking": f"Error during decision making: {str(e)}",
                "parameters": {
                    "error": f"Error during decision making: {str(e)}"
                }
            }
            
    def _extract_yaml_from_text(self, text: str) -> str:
        """
        Extracts YAML content from text that might contain additional content.
        
        Args:
            text: The text containing YAML content.
            
        Returns:
            str: The extracted YAML string.
        """
        # Find YAML block denoted by ```yaml and ``` markers
        yaml_start = text.find("```yaml")
        if yaml_start != -1:
            # Move past the ```yaml marker
            yaml_start += 7
            yaml_end = text.find("```", yaml_start)
            if yaml_end != -1:
                return text[yaml_start:yaml_end].strip()
        
        # If no markers, try to find what looks like YAML content
        # Look for "action:" as a key indicator
        action_idx = text.find("action:")
        if action_idx != -1:
            # Find the start of the YAML-like content
            yaml_start = text.rfind("\n", 0, action_idx)
            if yaml_start == -1:
                yaml_start = 0
            else:
                yaml_start += 1  # Skip the newline
                
            return text[yaml_start:].strip()
            
        # Handle cases where the model returns a natural language response first and then the YAML
        # This happens often with Gemini - it gives explanations before structured output
        lines = text.split('\n')
        yaml_lines = []
        yaml_section_started = False
        
        for line in lines:
            line = line.strip()
            # Check for key YAML indicators
            if line.startswith('action:'):
                yaml_section_started = True
                yaml_lines.append(line)
            elif yaml_section_started and (line.startswith('thinking:') or line.startswith('parameters:') 
                                          or line.startswith('-') or ':' in line):
                yaml_lines.append(line)
        
        if yaml_lines:
            # If we only found action but no thinking, add a minimal thinking section
            if not any(line.startswith('thinking:') for line in yaml_lines):
                yaml_lines.insert(0, 'thinking: |')
                yaml_lines.insert(1, '  Based on the query, I\'ve decided on the appropriate action.')
            
            return '\n'.join(yaml_lines)
        
        # If we can't extract YAML, try to construct a minimal valid YAML from the text
        logger.warning("Could not extract YAML from decision text. Constructing minimal YAML.")
        # Create a synthetic response with the text as thinking
        return f"""thinking: |
  {text.replace('\n', '\n  ')}
action: finish
parameters:
  reason: No structured decision found
"""


# Global instance of the LLM service
llm_service = LLMService() 