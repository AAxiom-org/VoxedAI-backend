# DecisionModel - "tool" >> ToolShed
# DecisionModel - "rag" >> RAGService
# DecisionModel - "finish" >> Finish

# ToolShed - "decide" >> DecisionModel
# RAGService - "decide" >> DecisionModel

from typing import Union, Dict, Any, AsyncGenerator
from pocketflow import AsyncFlow
from app.agents.base.nodes import DecisionNode, RAGNode, ToolShedNode, FinishNode
from app.core.logging import logger


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


async def run_agent_flow(space_id: str, query: str, view: str = None, stream: bool = False) -> Union[str, AsyncGenerator[str, None], Dict[str, Any]]:
    """
    Run the agent flow with the given inputs.
    
    Args:
        space_id: ID of the space the query pertains to
        query: The user's question or command
        view: Optional content the user is actively working with
        stream: Whether to stream the response
        
    Returns:
        Union[str, AsyncGenerator[str, None], Dict[str, Any]]: 
            - If stream=True: AsyncGenerator yielding response chunks
            - If stream=False: String response or Dict with response and metadata
    """
    logger.info(f"Running agent flow for space_id={space_id}, query='{query}', stream={stream}")
    
    # Create a shared context for the flow
    shared = {
        "space_id": space_id,
        "query": query,
        "view": view,
        "stream": stream,
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
            
            # Run the flow asynchronously
            await flow.run_async(shared)
            
            # After the flow is done, get the thinking history
            thinking_history = shared.get("thinking_history", [])
            
            # Yield each thinking step
            for step in thinking_history:
                yield f"Step {step['step']} Reasoning:\n{step['thinking']}\n\n"
            
            # Yield a marker to indicate the end of thinking and start of response
            yield "[THINKING_END]\n[RESPONSE_START]\n"
            
            # Get the final response
            final_response = shared.get("final_response", "No response generated")
            
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