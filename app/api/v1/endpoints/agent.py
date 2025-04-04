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
        request: Contains space_id, query, active_file_id, stream, model_name, top_k, and user_id parameters
        background_tasks: For handling cleanup tasks
        
    Returns:
        AgentResponse or StreamingResponse: Contains the agent's response and metadata
    """
    try:
        # Force streaming to true unless explicitly set to false
        if request.stream is None:
            request.stream = True
            
        logger.info(f"Executing agent workflow for space_id={request.space_id}, query='{request.query}', stream={request.stream}, active_file_id={request.active_file_id}, model_name={request.model_name}, top_k={request.top_k}, user_id={request.user_id}")
        
        start_time = time.time()
        
        # If streaming is requested, set up a streaming response with SSE
        if request.stream:
            async def event_generator():
                try:
                    # Run the agent flow with streaming enabled
                    response_generator = await run_agent_flow(
                        space_id=request.space_id,
                        query=request.query,
                        active_file_id=request.active_file_id,
                        stream=True,
                        model_name=request.model_name,
                        top_k=request.top_k,
                        user_id=request.user_id
                    )
                    
                    # Track if we've sent the sources event
                    sources_sent = False
                    # Buffer for collecting reasoning content
                    reasoning_buffer = ""
                    
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
                            
                        elif chunk.startswith("[EVENT:"):
                            # Parse the event type and data
                            event_end = chunk.find("]", 7)
                            if event_end > 7:
                                event_type = chunk[7:event_end]
                                event_data = chunk[event_end+1:chunk.find("[/EVENT]")]
                                
                                # Handle different event types
                                if event_type == "decision":
                                    event_obj = {
                                        "type": "agent_event",
                                        "event_type": "decision",
                                        "decision": event_data
                                    }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                                    
                                elif event_type == "file_edit_start":
                                    event_obj = {
                                        "type": "agent_event",
                                        "event_type": "file_edit_start",
                                        "file_id": event_data
                                    }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                                    
                                elif event_type == "file_edit_complete":
                                    event_obj = {
                                        "type": "agent_event", 
                                        "event_type": "file_edit_complete",
                                        "file_id": event_data
                                    }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                                    
                                elif event_type == "tool_complete":
                                    event_obj = {
                                        "type": "agent_event",
                                        "event_type": "tool_complete", 
                                        "tool": event_data
                                    }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                                    
                                elif event_type == "rag_complete":
                                    event_obj = {
                                        "type": "agent_event",
                                        "event_type": "rag_complete",
                                        "message": event_data
                                    }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                                    
                                # Handle all other event types with a standard format
                                else:
                                    # Special handling for error events to maintain compatibility
                                    if event_type == "error":
                                        event_obj = {
                                            "type": "error",
                                            "message": event_data
                                        }
                                    else:
                                        # Pass through all other events with their type and data
                                        event_obj = {
                                            "type": "agent_event",
                                            "event_type": event_type,
                                            "data": event_data
                                        }
                                    yield f"data: {json.dumps(event_obj)}\n\n"
                            continue
                            
                        elif "<reasoning>" in chunk:
                            # Extract reasoning tags and send immediately
                            start_idx = chunk.find("<reasoning>") + len("<reasoning>")
                            end_idx = chunk.find("</reasoning>", start_idx)
                            
                            if end_idx > start_idx:
                                # Extract the reasoning content
                                reasoning = chunk[start_idx:end_idx].strip()
                                
                                # Append to reasoning buffer
                                reasoning_buffer += reasoning
                                
                                # Send the reasoning event
                                reasoning_data = {
                                    "type": "reasoning",
                                    "content": reasoning_buffer
                                }
                                yield f"data: {json.dumps(reasoning_data)}\n\n"
                                
                                # Clear reasoning buffer after sending
                                reasoning_buffer = ""
                                
                                # Remove the reasoning part from the chunk before processing the rest
                                chunk = chunk[:start_idx - len("<reasoning>")] + chunk[end_idx + len("</reasoning>"):]
                            
                            # If chunk still has content, process it normally
                            if chunk.strip():
                                token_data = {
                                    "type": "token",
                                    "content": chunk
                                }
                                yield f"data: {json.dumps(token_data)}\n\n"
                            
                        elif "Step " in chunk and "Reasoning:" in chunk:
                            # This is thinking/reasoning content
                            reasoning_data = {
                                "type": "reasoning",
                                "content": chunk
                            }
                            yield f"data: {json.dumps(reasoning_data)}\n\n"
                            
                        else:
                            # This is a token from the LLM - send immediately without buffering
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
        
        # For non-streaming responses (explicit request for non-streaming)
        response_data = await run_agent_flow(
            space_id=request.space_id,
            query=request.query,
            active_file_id=request.active_file_id,
            stream=False,
            model_name=request.model_name,
            top_k=request.top_k,
            user_id=request.user_id
        )
        
        # Calculate query time
        query_time_ms = int((time.time() - start_time) * 1000)
        
        # Handle reasoning tokens in the non-streaming response
        response_text = response_data.get("response", "")
        reasoning = ""
        
        # Extract reasoning if present in tags
        if "<reasoning>" in response_text and "</reasoning>" in response_text:
            start_idx = response_text.find("<reasoning>") + len("<reasoning>")
            end_idx = response_text.find("</reasoning>")
            if end_idx > start_idx:
                reasoning = response_text[start_idx:end_idx].strip()
                response_text = response_text[:start_idx - len("<reasoning>")] + response_text[end_idx + len("</reasoning>"):]
        
        # The response_data is now a dict with 'response' and 'thinking' keys
        return AgentResponse(
            success=True,
            response=response_text.strip(),
            metadata={
                "flow_type": "pocketflow_agent",
                "thinking": response_data.get("thinking", []),
                "reasoning": reasoning,  # Add extracted reasoning
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