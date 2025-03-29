# DecisionModel - "tool" >> ToolShed
# DecisionModel - "rag" >> RAGService
# DecisionModel - "finish" >> Finish

# ToolShed - "decide" >> DecisionModel
# RAGService - "decide" >> DecisionModel

from typing import Union, Dict, Any, AsyncGenerator, Optional
from pocketflow import AsyncFlow
from app.agents.base.nodes import DecisionNode, RAGNode, ToolShedNode, FinishNode
from app.core.logging import logger
import asyncio
import traceback


def create_agent_flow() -> AsyncFlow:
    """
    Create and connect the nodes to form a complete agent flow.
    
    The flow works like this:
    1. DecisionNode decides whether to use RAG, Tool, or Finish
    2. If RAG, use RAGNode to retrieve context then go back to DecisionNode
    3. If Tool, use ToolShedNode to execute the tool then go back to DecisionNode
    4. If Finish, use FinishNode to generate the final answer
    
    Returns:
        AsyncFlow: A complete agent flow
    """
    logger.info("Creating agent flow")
    
    # Create instances of each node
    decision_node = DecisionNode()
    rag_node = RAGNode()
    tool_shed_node = ToolShedNode()
    finish_node = FinishNode()
    
    # Connect the nodes
    # If DecisionNode returns "rag", go to RAGNode
    decision_node - "rag" >> rag_node
    
    # If DecisionNode returns "tool", go to ToolShedNode
    decision_node - "tool" >> tool_shed_node
    
    # If DecisionNode returns "finish", go to FinishNode
    decision_node - "finish" >> finish_node
    
    # After RAGNode completes and returns "decide", go back to DecisionNode
    rag_node - "decide" >> decision_node
    
    # After ToolShedNode completes and returns "decide", go back to DecisionNode
    tool_shed_node - "decide" >> decision_node
    
    # Create and return the flow, starting with the DecisionNode
    return AsyncFlow(start=decision_node)


async def run_agent_flow(
    space_id: str, 
    query: str, 
    active_file_id: Optional[str] = None,
    stream: bool = False,
    model_name: Optional[str] = None,
    top_k: Optional[int] = None,
    user_id: Optional[str] = None
) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
    """
    Run the agent flow with the given inputs.
    
    Args:
        space_id: ID of the space the query pertains to
        query: The user's question or command
        active_file_id: ID of the file the user is actively working with, or None if no file is active
        stream: Whether to stream the response
        model_name: Optional model name to use for the response
        top_k: Optional number of top results to consider
        user_id: Optional user ID for tracking or personalization
        
    Returns:
        Union[str, AsyncGenerator[str, None], Dict[str, Any]]: 
            - If stream=True: AsyncGenerator yielding response chunks
            - If stream=False: String response or Dict with response and metadata
    """
    logger.info(f"Running agent flow for space_id={space_id}, query='{query}', stream={stream}, active_file_id={active_file_id}, model_name={model_name}, top_k={top_k}, user_id={user_id}")
    
    # Create a shared context for the flow
    shared = {
        "space_id": space_id,
        "active_file_id": active_file_id,
        "query": query,
        "stream": stream,
        "model_name": model_name,
        "top_k": top_k,
        "user_id": user_id,
        "context": {},
        "action_history": [],
        "thinking_history": []
    }
    
    if stream:
        # For streaming, we'll create a generator that yields response chunks
        async def stream_generator():
            # Create and run the flow
            flow = create_agent_flow()
            
            # First, yield a special marker to indicate the start of thinking
            yield "[THINKING_START]\n"
            
            # Set up a queue for async communication between nodes and this generator
            shared["event_queue"] = asyncio.Queue()
            
            # Start a task to run the flow asynchronously
            flow_task = asyncio.create_task(flow.run_async(shared))
            
            # Process events as they come in
            running = True
            while running or not shared["event_queue"].empty():
                try:
                    # Try to get an event from the queue with a timeout
                    try:
                        event = await asyncio.wait_for(shared["event_queue"].get(), 0.1)
                        
                        # Process the event based on its type
                        if event["type"] == "decision":
                            # Send a special marker for decision events
                            yield f"[EVENT:{event['type']}]{event['decision']}[/EVENT]\n"
                        elif event["type"] == "file_edit_start":
                            # Send a special marker for file edit start events
                            yield f"[EVENT:{event['type']}]{event['file_id']}[/EVENT]\n"
                        elif event["type"] == "file_edit_complete":
                            # Send a special marker for file edit complete events
                            yield f"[EVENT:{event['type']}]{event['file_id']}[/EVENT]\n"
                        elif event["type"] == "thinking":
                            # Send thinking steps as before
                            yield f"Step {event['step']} Reasoning:\n{event['thinking']}\n\n"
                        elif event["type"] == "flow_complete":
                            running = False
                    except asyncio.TimeoutError:
                        # Check if the flow has completed
                        if flow_task.done():
                            running = False
                        continue
                except Exception as e:
                    logger.error(f"Error processing event: {str(e)}")
                    yield f"[EVENT:error]Error processing event: {str(e)}[/EVENT]\n"
            
            # Wait for the flow task to complete
            try:
                await flow_task
            except Exception as e:
                logger.error(f"Error in flow execution: {str(e)}")
                logger.error(f"Flow execution traceback: {traceback.format_exc()}")
                yield f"[EVENT:error]Error in flow execution: {str(e)}[/EVENT]\n"
            
            # Yield a marker to indicate the end of thinking and start of response
            yield "[THINKING_END]\n[RESPONSE_START]\n"
            
            # Get the final response
            if "final_response" not in shared:
                logger.error("No final_response in shared context - shared keys: " + str(shared.keys()))
                final_response = "I encountered an error while processing your request. Please try again."
                yield final_response
                yield "[RESPONSE_END]"
                return
                
            final_response = shared.get("final_response", "No response generated")
            
            # Log the type of final_response for debugging
            logger.info(f"Final response type: {type(final_response)}, value preview: {str(final_response)[:100]}")
            
            # If the final response is a generator (streaming from the LLM), yield from it
            if hasattr(final_response, '__aiter__'):
                buffer = ""
                async for chunk in final_response:
                    # Check if the chunk contains a reasoning tag
                    if "<reasoning>" in chunk and "</reasoning>" in chunk:
                        # Extract reasoning and yield it separately with special formatting
                        start_idx = chunk.find("<reasoning>") + len("<reasoning>")
                        end_idx = chunk.find("</reasoning>")
                        reasoning = chunk[start_idx:end_idx].strip()
                        
                        # Yield the reasoning chunk separately
                        yield f"<reasoning>{reasoning}</reasoning>"
                        
                        # Remove the reasoning part from the chunk before processing the rest
                        chunk = chunk[:start_idx - len("<reasoning>")] + chunk[end_idx + len("</reasoning>"):]
                    
                    # Handle the regular token content
                    if chunk:
                        buffer += chunk
                        # Split on natural boundaries and yield complete segments
                        segments = buffer.split(" ")
                        if len(segments) > 1:
                            # Yield all segments except the last one (which might be incomplete)
                            yield " ".join(segments[:-1]) + " "
                            # Keep the last segment in the buffer
                            buffer = segments[-1]
                            
                # Yield any remaining content in the buffer
                if buffer:
                    yield buffer
            else:
                # Otherwise, yield the entire response
                yield final_response
                
            # Yield a marker to indicate the end of the response
            yield "[RESPONSE_END]"
            
        # Return the generator
        return stream_generator()
    else:
        # For non-streaming, run the flow and return the full response with metadata
        flow = create_agent_flow()
        await flow.run_async(shared)
        
        # Get the thinking history and final response
        thinking_history = shared.get("thinking_history", [])
        final_response = shared.get("final_response", "No response generated")
        
        # Return a dictionary with both the final response and thinking history
        return {
            "response": final_response,
            "thinking": thinking_history
        }