"""
API endpoints for agent operations.
"""
import json
import time
from typing import Any, Dict
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.core.logging import logger
from app.schemas.agent import AgentRequest, AgentResponse
from app.agents.base.flow import run_agent_flow

router = APIRouter()


@router.post("/run", response_model=AgentResponse)
async def run_agent(
    request: AgentRequest,
    background_tasks: BackgroundTasks
) -> Any:
    """
    Executes the agent workflow.
    
    This endpoint:
    1. Takes a user query and space ID
    2. Processes it through the agent workflow (Decision → RAG → Tool Shed → Finish)
    3. Returns the response
    
    Args:
        request: Contains space_id, query, view, and stream parameters
        background_tasks: For handling cleanup tasks
        
    Returns:
        AgentResponse or StreamingResponse: Contains the agent's response and metadata
    """
    try:
        logger.info(f"Executing agent workflow for space_id={request.space_id}, query='{request.query}', stream={request.stream}")
        
        start_time = time.time()
        
        # If streaming is requested, set up a streaming response with SSE
        if request.stream:
            async def event_generator():
                try:
                    # Run the agent flow with streaming enabled
                    response_generator = await run_agent_flow(
                        space_id=request.space_id,
                        query=request.query,
                        view=request.view,
                        stream=True
                    )
                    
                    # Track if we've sent the sources event
                    sources_sent = False
                    
                    # Stream the response chunks as SSE events
                    async for chunk in response_generator:
                        # Check for special markers in the chunk
                        if chunk.startswith("[THINKING_START]"):
                            # Send a sources event with empty data (placeholder)
                            if not sources_sent:
                                sources_data = {
                                    "type": "sources",
                                    "sources": []  # Empty sources for now
                                }
                                yield f"data: {json.dumps(sources_data)}\n\n"
                                sources_sent = True
                            continue
                            
                        elif "[THINKING_END]" in chunk or "[RESPONSE_START]" in chunk or "[RESPONSE_END]" in chunk:
                            # Skip these marker chunks
                            continue
                            
                        elif "Step " in chunk and "Reasoning:" in chunk:
                            # This is thinking/reasoning content
                            reasoning_data = {
                                "type": "reasoning",
                                "content": chunk
                            }
                            yield f"data: {json.dumps(reasoning_data)}\n\n"
                            
                        else:
                            # This is a token from the LLM
                            token_data = {
                                "type": "token",
                                "content": chunk
                            }
                            yield f"data: {json.dumps(token_data)}\n\n"
                    
                    # Send a final "done" event
                    query_time_ms = int((time.time() - start_time) * 1000)
                    done_data = {
                        "type": "done",
                        "query_time_ms": query_time_ms
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in event generator: {str(e)}")
                    error_data = {
                        "type": "error",
                        "error": str(e)
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # For non-streaming responses, run the agent flow and return the complete response
        response_data = await run_agent_flow(
            space_id=request.space_id,
            query=request.query,
            view=request.view,
            stream=False
        )
        
        # Calculate query time
        query_time_ms = int((time.time() - start_time) * 1000)
        
        # The response_data is now a dict with 'response' and 'thinking' keys
        return AgentResponse(
            success=True,
            response=response_data["response"],
            metadata={
                "flow_type": "pocketflow_agent",
                "thinking": response_data["thinking"],
                "query_time_ms": query_time_ms,
                "sources": []  # Empty sources placeholder
            }
        )
        
    except ValidationError as e:
        logger.error(f"Validation error in agent request: {str(e)}")
        return AgentResponse(
            success=False,
            response=f"Invalid request: {str(e)}",
            metadata={"error_type": "validation"}
        )
    except HTTPException as e:
        logger.error(f"HTTP error executing agent: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent: {str(e)}"
        ) 