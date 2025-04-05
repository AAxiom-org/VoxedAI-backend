# app/toolshed/nodes.py

from typing import Dict, Any, List, Optional
from pocketflow import AsyncNode
from app.services.llm_service import llm_service
from app.core.logging import logger
import yaml
import pickle

class ToolShedDecisionNode(AsyncNode):
    """
    Tool Shed Decision node that determines if a tool is needed and which specific tool to use.
    Routes directly to the appropriate tool node with parameters.
    """
    
    # Available tools with their descriptions
    AVAILABLE_TOOLS = {
        "file_interaction": {
            "description": "Read or edit files in the workspace. Only notes can be edited. If the user is working on a note then you can access and edit it without needing any additional parameters.",
            "parameters": [
                {"name": "action", "description": "The action to perform: 'view'/'read' for any file, 'edit'/'append'/'replace_snippet' only for note files."},
                {"name": "content", "description": "For edit operations, any content to be added/modified. Not needed for 'view'/'read' operations."}
            ]
        },
        "web_search": {
            "description": "Search the web for information",
            "parameters": [
                {"name": "query", "description": "The search query to send to the web search engine"},
                {"name": "num_results", "description": "Optional number of results to return"}
            ]
        },
        # Add more tools as needed
    }
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        action_history = shared.get("action_history", [])
        stream = shared.get("stream", False)
        active_file_id = shared.get("active_file_id")
        event_queue = shared.get("event_queue")
        
        logger.info(f"ToolShedDecisionNode: Processing query '{query}', active_file_id={active_file_id}")
        
        # Send event about tool decision starting
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "tool_decision_start",
                    "message": "Determining which tool to use for the query",
                    "query": query,
                    "active_file_id": active_file_id
                })
            except Exception as e:
                logger.error(f"Error sending tool_decision_start event: {str(e)}")
        
        return {
            "query": query,
            "context": context,
            "action_history": action_history,
            "stream": stream,
            "available_tools": self.AVAILABLE_TOOLS,
            "active_file_id": active_file_id,
            "_shared": shared  # Pass along shared context for event queue access
        }
    
    async def exec_async(self, prep_res):
        query = prep_res["query"]
        context = prep_res["context"]
        available_tools = prep_res["available_tools"]
        active_file_id = prep_res.get("active_file_id")
        shared = prep_res.get("_shared", {})
        event_queue = shared.get("event_queue")
        
        # Send event that LLM is being queried for tool decision
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "tool_decision_llm_query",
                    "message": "Querying language model to determine the appropriate tool",
                    "available_tools": list(available_tools.keys())
                })
            except Exception as e:
                logger.error(f"Error sending tool_decision_llm_query event: {str(e)}")
        
        # Note about the active file ID for the prompt
        active_file_info = f"""
        # ACTIVE FILE INFORMATION
        Active File ID: {active_file_id if active_file_id else "None"}
        
        IMPORTANT: If the operation involves a file, you should use the provided active_file_id above 
        and should NOT include file_id in your parameters. The system already knows which file to use.
        """ if active_file_id else ""
        
        # Construct a prompt that asks for both tool selection and parameters in one call
        prompt = f"""
        # USER QUERY
        {query}
        {active_file_info}
        # CONTEXT INFORMATION
        {self._safe_yaml_dump(context)}
        
        # AVAILABLE TOOLS
        {yaml.dump(available_tools, default_flow_style=False)}
        
        # TASK
        Based on the user's query and context, determine:
        1. If a tool is needed to answer this query
        2. If so, which specific tool would be best
        3. What parameters should be passed to this tool
        
        IMPORTANT NOTES:
        - If the user wants to work with a file, the active_file_id is already available to the system.
        - For file_interaction tool, only specify action and content (if needed). DO NOT specify file_id.
        - For note files (is_note=true), you can use actions: view, read, edit, append, replace_snippet
        - For non-note files, you can ONLY use actions: view, read
        - If the user mentions something ambigouos like "my code" or "this file" or "the file" or "the code", 
          you should assume they are referring to the active file. Which is already known by the system.
        
        NOTE EDITING GUIDANCE:
        - For queries about writing, continuing, or finishing notes/plans: use file_interaction with action="edit"
        - For generating new content like "write a research plan": use file_interaction with action="edit" or "append"
        - The active file's content will be available to you in the next step, so don't worry about not having seen it yet
        - Sample queries that should use file_interaction:
          * "finish writing my plan" → action="edit" 
          * "continue writing" → action="edit"
          * "generate a research outline" → action="edit"
          * "write more content" → action="edit"
        
        Respond in YAML format:
        ```yaml
        thinking: |
            <your step-by-step reasoning>
        action: <tool_name_if_needed or "none">
        parameters:
            <param1>: <value1>
            <param2>: <value2>
        ```
        
        If no tool is needed, set action to "none" and leave parameters empty.
        """
        
        # Send event that LLM is being queried for tool decision
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "tool_decision_prompt_sent",
                    "message": "Sent prompt to language model to determine tool selection"
                })
            except Exception as e:
                logger.error(f"Error sending tool_decision_prompt_sent event: {str(e)}")
        
        # Call LLM with the prompt
        llm_response = await llm_service._call_llm(
            prompt=prompt,
            temperature=0.1,  # Lower temperature for more deterministic responses
            max_tokens=1024
        )
        
        # Send event that LLM response was received
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "tool_decision_llm_response",
                    "message": "Received response from language model for tool selection"
                })
            except Exception as e:
                logger.error(f"Error sending tool_decision_llm_response event: {str(e)}")
        
        # Parse the YAML response
        try:
            # Extract YAML content between triple backticks if present
            if "```yaml" in llm_response and "```" in llm_response.split("```yaml", 1)[1]:
                yaml_content = llm_response.split("```yaml", 1)[1].split("```", 1)[0].strip()
            else:
                yaml_content = llm_response
                
            decision = yaml.safe_load(yaml_content)
            
            # Send event about the parsed decision
            if event_queue is not None:
                try:
                    action = decision.get("action", "none")
                    parameters = decision.get("parameters", {})
                    await event_queue.put({
                        "type": "tool_decision_parsed",
                        "message": f"Parsed tool decision: {action}",
                        "action": action,
                        "parameters": parameters
                    })
                except Exception as e:
                    logger.error(f"Error sending tool_decision_parsed event: {str(e)}")
            
        except yaml.YAMLError:
            logger.error(f"Failed to parse LLM response as YAML: {llm_response}")
            decision = {
                "action": "none",
                "parameters": {},
                "thinking": "Failed to parse response"
            }
            
            # Send error event
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "tool_decision_parse_error",
                        "message": "Failed to parse language model response as YAML",
                        "raw_response": llm_response[:500]  # Include part of the response for debugging
                    })
                except Exception as e:
                    logger.error(f"Error sending tool_decision_parse_error event: {str(e)}")
        
        return decision
    
    async def post_async(self, shared, prep_res, decision):
        # Store thinking for later reference
        thinking_history = shared.get("thinking_history", [])
        thinking_history.append({
            "step": "tool_decision",
            "thinking": decision.get("thinking", "")
        })
        shared["thinking_history"] = thinking_history
        
        # Store the parameters in shared context for the tool to use
        shared["tool_parameters"] = decision.get("parameters", {})
        
        # Get access to the event queue
        event_queue = shared.get("event_queue")
        
        # Add more debugging about the decision
        logger.info(f"ToolShedDecisionNode post_async - Full decision: {decision}")
        
        # Send decision thinking as a reasoning event if available
        if event_queue is not None and decision.get("thinking"):
            try:
                # Format the reasoning with a clear header
                reasoning_content = f"Decision Reasoning: I'm deciding what tool to use for '{prep_res.get('query', '')}'\n\n{decision['thinking']}"
                await event_queue.put({
                    "type": "reasoning",
                    "content": reasoning_content
                })
            except Exception as e:
                logger.error(f"Error sending decision reasoning event: {str(e)}")
        
        # If no tool is needed, return None (will end the flow)
        if decision.get("action") == "none":
            logger.info("ToolShedDecisionNode: No tool needed, returning None to end the flow")
            
            # Send no tool needed event
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "tool_decision_no_tool",
                        "message": "Determined that no tool is needed for this query",
                        "thinking": decision.get("thinking", "")[:500]  # Include part of the thinking
                    })
                except Exception as e:
                    logger.error(f"Error sending tool_decision_no_tool event: {str(e)}")
            
            # Set a tool_results with a specific no_tool flag to avoid NoneType errors
            shared["tool_results"] = {
                "result_type": "no_tool_needed",
                "message": "No tool action was required to complete this request."
            }
            return None
            
        # Return the selected tool name for routing
        action = decision.get("action")
        logger.info(f"ToolShedDecisionNode: Routing to tool '{action}'")
        
        # Send tool selection event
        if event_queue is not None:
            try:
                parameters = decision.get("parameters", {})
                await event_queue.put({
                    "type": "tool_selected",
                    "message": f"Selected tool: {action}",
                    "tool": action,
                    "parameters": parameters,
                    "thinking": decision.get("thinking", "")  # Include the decision thinking
                })
            except Exception as e:
                logger.error(f"Error sending tool_selected event: {str(e)}")
        
        return action

    def _safe_yaml_dump(self, obj):
        """Helper method to safely dump objects to YAML, filtering out non-serializable items."""
        try:
            # First try with the standard yaml dump
            return yaml.dump(obj, default_flow_style=False)
        except Exception as e:
            # Log the error for debugging
            logger.debug(f"Error in YAML serialization: {str(e)}, falling back to safe mode")
            
            # If it fails, handle containers recursively
            if isinstance(obj, dict):
                # Create a new dict with only serializable values
                safe_dict = {}
                for k, v in obj.items():
                    try:
                        safe_dict[k] = self._safe_yaml_dump_value(v)
                    except Exception:
                        # If anything goes wrong, use string representation
                        safe_dict[k] = f"<non-serializable: {type(v).__name__}>"
                return yaml.dump(safe_dict, default_flow_style=False)
            elif isinstance(obj, (list, tuple)):
                # Process lists recursively
                safe_list = []
                for item in obj:
                    try:
                        safe_list.append(self._safe_yaml_dump_value(item))
                    except Exception:
                        safe_list.append(f"<non-serializable: {type(item).__name__}>")
                return yaml.dump(safe_list, default_flow_style=False)
            else:
                # For simple non-serializable objects, return a string representation
                return yaml.dump(str(obj), default_flow_style=False)

    def _safe_yaml_dump_value(self, val):
        """Recursively prepare values for safe YAML dumping."""
        if val is None or isinstance(val, (str, int, float, bool)):
            # Basic types are safe
            return val
        elif isinstance(val, dict):
            return {k: self._safe_yaml_dump_value(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [self._safe_yaml_dump_value(item) for item in val]
        else:
            # For any other type, use string representation
            return str(val)