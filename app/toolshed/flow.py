# app/toolshed/flow.py

from typing import Union, Dict, Any, AsyncGenerator, List, Optional
from pocketflow import AsyncFlow
from app.toolshed.nodes import ToolShedDecisionNode
from app.tools.file import FileInteraction
from app.tools.web_search import WebSearch
from app.core.logging import logger
import asyncio

def create_toolshed_flow() -> AsyncFlow:
    """
    Create a toolshed flow that dynamically routes to the appropriate tool node.
    
    The flow works like this:
    1. ToolShedDecisionNode decides whether a tool is needed, which specific tool, and with what parameters
    2. If a tool is selected, flow routes directly to that specific tool node with the parameters
    3. Each tool handles its own execution and response formatting
    
    Returns:
        AsyncFlow: A complete toolshed flow
    """
    logger.info("Creating toolshed flow")
    
    # Create instances of nodes
    decision_node = ToolShedDecisionNode()
    
    # Create instances of all tool nodes
    file_interaction = FileInteraction()
    web_search = WebSearch()
    
    # Connect decision node to each tool node
    decision_node - "file_interaction" >> file_interaction
    decision_node - "web_search" >> web_search
    
    # Create and return the flow, starting with the decision node
    return AsyncFlow(start=decision_node)

async def run_toolshed_flow(
    query: str, 
    context: Dict[str, Any], 
    action_history: List[Dict], 
    stream: bool = False,
    active_file_id: Optional[str] = None,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,
    event_queue: Optional[asyncio.Queue] = None
) -> Dict[str, Any]:
    """
    Run the toolshed flow with the given inputs.
    
    Args:
        query: The user's question or command
        context: Context gathered so far
        action_history: History of actions taken
        stream: Whether to stream the response
        active_file_id: ID of the file the user is currently viewing/editing
        space_id: ID of the current space
        user_id: ID of the current user
        event_queue: Optional queue for sending events back to the client
        
    Returns:
        Dict[str, Any]: Results of the tool execution with metadata
    """
    logger.info(f"Running toolshed flow for query='{query}', active_file_id={active_file_id}")
    
    # If we have an event queue, send a toolshed start event
    if event_queue is not None:
        try:
            await event_queue.put({
                "type": "toolshed_start",
                "message": "Starting toolshed flow to determine which tool to use",
                "query": query
            })
        except Exception as e:
            logger.error(f"Error sending toolshed_start event: {str(e)}")
    
    # Create a shared context for the flow
    shared = {
        "query": query,
        "context": context,
        "action_history": action_history,
        "stream": stream,
        "active_file_id": active_file_id,  # Pass the active_file_id to the tools
        "space_id": space_id,
        "user_id": user_id,
        "tool_parameters": {},   # Will be populated by decision node
        "tool_results": None,    # Will be populated by the specific tool node
        "event_queue": event_queue  # Pass the event queue to the tools
    }
    
    try:
        # Create and run the flow
        flow = create_toolshed_flow()
        await flow.run_async(shared)
        
        # Check if tool_results was set by a tool
        if shared.get("tool_results") is None:
            logger.warning("ToolShed flow completed but no tool_results were set")
            
            # Send event about no tool being executed
            if event_queue is not None:
                try:
                    await event_queue.put({
                        "type": "toolshed_no_tool",
                        "message": "No tool action was performed to complete this request"
                    })
                except Exception as e:
                    logger.error(f"Error sending toolshed_no_tool event: {str(e)}")
                    
            # Set a default tool_results to avoid NoneType errors
            shared["tool_results"] = {
                "result_type": "no_tool_executed",
                "message": "No tool action was performed to complete this request."
            }
        else:
            # Send event about tool execution completion
            if event_queue is not None:
                try:
                    tool_results = shared.get("tool_results", {})
                    await event_queue.put({
                        "type": "toolshed_complete",
                        "message": "Toolshed flow completed successfully",
                        "tool_used": tool_results.get("tool_used", "unknown"),
                        "result_type": tool_results.get("result_type", "unknown")
                    })
                except Exception as e:
                    logger.error(f"Error sending toolshed_complete event: {str(e)}")
    except Exception as e:
        logger.error(f"Error running toolshed flow: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Toolshed flow error traceback: {error_traceback}")
        
        # Send error event
        if event_queue is not None:
            try:
                await event_queue.put({
                    "type": "toolshed_error",
                    "message": f"Error in toolshed flow: {str(e)}",
                    "error": str(e),
                    "traceback": error_traceback
                })
            except Exception as event_e:
                logger.error(f"Error sending toolshed_error event: {str(event_e)}")
        
        # Set an error result
        shared["tool_results"] = {
            "result_type": "error",
            "error": f"Error in toolshed flow: {str(e)}",
            "tool_name": "unknown"
        }
    
    # Log the result before returning
    logger.info(f"Toolshed flow completed. Result type: {shared.get('tool_results', {}).get('result_type', 'unknown')}")
    
    # Return the results (handled by the specific tool)
    return shared.get("tool_results", {"result_type": "no_tool_needed"})