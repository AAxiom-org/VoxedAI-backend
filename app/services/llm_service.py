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
            logger.info(f"DEBUG - Prompt being sent to llm: {prompt}")
            
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
            logger.info(f"DEBUG - Payload: {payload}")
            
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
                
                # Check for reasoning field in non-streaming response
                if include_reasoning and "reasoning" in data["choices"][0]["message"]:
                    reasoning = data["choices"][0]["message"]["reasoning"]
                    if reasoning:
                        logger.info(f"DEBUG - Non-streaming reasoning detected: {reasoning}")
                        # Add reasoning with special tags that will be extracted later
                        content = f"<reasoning>{reasoning}</reasoning>{content}"
                
                logger.info(f"DEBUG - Non-streaming response received")
                logger.info(f"DEBUG - LLM Response (non-streaming): {content}")
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
        stream: bool = False,
        active_file_id: str = None
    ) -> Dict[str, Any]:
        """
        Uses a language model to decide the next action in the agent workflow.
        Returns just the action name for speed.
        
        Args:
            query: The user's question or command
            context: Any context gathered so far
            action_history: Previous actions taken by the agent
            stream: Whether to stream the response
            
        Returns:
            Dict[str, Any]: A decision object containing action and parameters
        """
        try:
            # Check if we have a successful file edit in the recent history
            file_edit_success = False
            for action in action_history[-5:] if action_history else []:
                if (action.get('action') == "tool" and 
                    action.get('tool_name') == "file_interaction" and
                    action.get('success', False) and 
                    "success" in action.get('message', '').lower()):
                    file_edit_success = True
                    break
            
            # Format previous actions for the prompt
            history_text = ""
            
            if action_history:
                for i, action in enumerate(action_history[-3:]):  # Only use last 3 actions for brevity
                    action_type = action.get('action', 'unknown')
                    if action_type == "tool":
                        # Include more details about the tool action
                        tool_name = action.get('tool_name', 'unknown tool')
                        result_type = action.get('result_type', 'unknown result')
                        success = action.get('success', False)
                        success_str = "✅ SUCCESS" if success else "❌ FAILED"
                        message = action.get('message', '')
                        
                        # Extract any detailed information about changes if available
                        detailed_info = ""
                        result = action.get('result', {})
                        if isinstance(result, dict) and "changes" in result:
                            detailed_info = f" - Changes: {result['changes']}"
                        
                        # Include brief parameter info if available
                        params_info = ""
                        if 'parameters' in action and action['parameters']:
                            params = action['parameters']
                            if 'action' in params:
                                params_info = f" ({params.get('action', '')})"
                            elif 'content' in params:
                                # For file edits, show content summary
                                content = params.get('content', '')
                                if len(content) > 30:
                                    content = content[:27] + "..."
                                params_info = f" (content: '{content}')"
                        
                        history_text += f"Action {i+1}: {action_type} - Used {tool_name}{params_info}, Result: {result_type} [{success_str}] - {message}{detailed_info}\n"
                    else:
                        history_text += f"Action {i+1}: {action_type}\n"
            
            # TODO: Add rag back when implemented
            no_file_tool = """1. **tool** - Use a specialized tool: The only tool available is for file reading. THERE IS NO NOTE/FILE EDITING! Do not
                choose tool unless you need to read a file to respond to the users query."""
            active_file_tool = """1. **tool** - Use a specialized tool: The only tools available are for file reading, and note editing. There is
            an active note open, so you can use the tool to read it or edit it. Since there is an active note open you should choose tool if you need to read or write to it."""
            
            # Task state indicator for file edit success
            task_state = "✅ FILE EDIT ALREADY COMPLETED SUCCESSFULLY" if file_edit_success else ""
            
            prompt = f"""
                You are a decision-making component in an AI agent workflow. Choose the next action based on the user query.
                
                ## USER QUERY
                {query}

                ## ACTION HISTORY (MOST RECENT)
                {history_text or "No previous actions taken."}
                
                ## TASK STATE
                {task_state}

                ## AVAILABLE ACTIONS
                {active_file_tool if active_file_id else no_file_tool}
                2. **finish** - Generate final answer with context you already have
                
                ## DECISION GUIDELINES
                - CRITICAL: If a file edit has ALREADY been completed successfully, you MUST choose "finish"
                - For simple greetings or basic questions: choose "finish" immediately
                - For general questions about anything (story telling, quick facts, non-contextual question answering): choose "finish" immediately
                - If it is something ambiguous, or you are unsure of like "write a story about a cat": choose "finish"
                - If a previous tool action failed (shows as "FAILED"), choose "tool" again to retry
                - If a file edit has been completed successfully (shows "SUCCESS"), choose "finish" unless the user explicitly requests additional different edits
                {"- IMPORTANT OVERRIDE: Choose 'finish' as this task has already been completed successfully" if file_edit_success else ""}

                ## INSTRUCTIONS
                Respond with ONE WORD ONLY - either "tool", or "finish".
                NO explanation. NO reasoning. Just the action name.
            """
            
            # Use a fast model for decision-making
            response_text = await self._call_llm(
                prompt=prompt,
                model_name="google/gemini-2.0-flash-001", 
                stream=False,  # Decision is always non-streaming
                temperature=0,
                max_tokens=5  # Just need one word
            )
            
            # Clean up response and extract just the action
            response_text = response_text.strip().lower()
            
            # Force "finish" if a file edit was successful
            if file_edit_success and "finish" not in response_text:
                logger.info("Override: Forcing 'finish' action because file edit was successful")
                action = "finish"
            # Extract the action from the response
            elif "rag" in response_text:
                action = "rag"
            elif "tool" in response_text:
                action = "tool"
            else:
                action = "finish"  # Default to finish if unclear
                
            # Create minimal decision object
            decision = {
                "action": action,
                "thinking": f"Quick decision: {action}" + (f" (forced finish due to successful file edit)" if file_edit_success and "tool" in response_text else ""),
                "parameters": {}
            }
            
            logger.info(f"Quick decision: {action}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting decision: {e}")
            # Default to finish action if there's an error
            return {
                "action": "finish",
                "thinking": f"Error during decision making: {str(e)}",
                "parameters": {}
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