"""
API endpoints for agent operations.
"""
import json
import time
import uuid
from typing import Any, Dict, List
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.core.logging import logger
from app.schemas.agent import AgentRequest, AgentResponse, AgentEvent
from app.agents.base.flow import run_agent_flow
from app.db.supabase import supabase_client
from app.models.chat_models import ChatSessionCreate, ChatMessageCreate

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
        request: Contains space_id, query, active_file_id, stream, model_name, top_k, user_id, and chat_session_id parameters
        background_tasks: For handling cleanup tasks
        
    Returns:
        AgentResponse or StreamingResponse: Contains the agent's response and metadata
    """
    try:
        # Force streaming to true unless explicitly set to false
        if request.stream is None:
            request.stream = True
            
        logger.info(f"Executing agent workflow for space_id={request.space_id}, query='{request.query}', stream={request.stream}, active_file_id={request.active_file_id}, model_name={request.model_name}, top_k={request.top_k}, user_id={request.user_id}, chat_session_id={request.chat_session_id}")
        
        start_time = time.time()
        
        # Create a chat session if one wasn't provided but we need to save to DB
        chat_session_id = request.chat_session_id
        if request.save_to_db and not chat_session_id and request.user_id:
            try:
                # Create a title from the first few words of the query
                title = request.query[:30] + ("..." if len(request.query) > 30 else "")
                
                # Create session data
                session_data = ChatSessionCreate(
                    user_id=request.user_id,
                    space_id=request.space_id,
                    title=title
                )
                
                # Create the session
                session = await supabase_client.create_chat_session(session_data)
                chat_session_id = session.id
                logger.info(f"Created new chat session: {chat_session_id}")
            except Exception as e:
                logger.error(f"Error creating chat session: {e}")
                # Continue without chat session if creation fails
        
        # Save the user message to the database if requested
        if request.save_to_db and chat_session_id and request.user_id:
            try:
                # Create message data
                user_message_data = ChatMessageCreate(
                    chat_session_id=chat_session_id,
                    space_id=request.space_id,
                    user_id=request.user_id,
                    content=request.query,
                    is_user=True,
                    workflow=None  # User messages don't have workflow data
                )
                
                # Save the message
                await supabase_client.save_chat_message(user_message_data)
                logger.info(f"Saved user message to chat session: {chat_session_id}")
            except Exception as e:
                logger.error(f"Error saving user message: {e}")
                # Continue even if saving the message fails
        
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
                    # Buffer for collecting agent events
                    agent_events: List[AgentEvent] = []
                    # Buffer for collecting response content
                    response_buffer = ""
                    
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
                                
                                # Create an AgentEvent object
                                event_obj = AgentEvent(
                                    type="agent_event",
                                    event_type=event_type
                                )
                                
                                # Set the appropriate field based on event type
                                if event_type == "decision":
                                    event_obj.decision = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'decision', 'decision': event_data})}\n\n"
                                    
                                elif event_type == "file_edit_start":
                                    event_obj.file_id = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'file_edit_start', 'file_id': event_data})}\n\n"
                                    
                                elif event_type == "file_edit_complete":
                                    event_obj.file_id = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'file_edit_complete', 'file_id': event_data})}\n\n"
                                    
                                elif event_type == "tool_complete":
                                    event_obj.tool = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'tool_complete', 'tool': event_data})}\n\n"
                                    
                                elif event_type == "rag_complete":
                                    event_obj.message = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'rag_complete', 'message': event_data})}\n\n"
                                    
                                elif event_type == "tool_selected":
                                    if isinstance(event_data, dict) and "tool" in event_data:
                                        tool_name = event_data["tool"]
                                    else:
                                        tool_name = event_data
                                        
                                    event_obj.tool = tool_name
                                    # Include parameters if available
                                    parameters = event.get("parameters", {})
                                    
                                    # Send event to client with tool and parameters
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': 'tool_selected', 'tool': tool_name, 'parameters': parameters})}\n\n"
                                    
                                # Handle error events
                                elif event_type == "error":
                                    event_obj.message = event_data
                                    yield f"data: {json.dumps({'type': 'error', 'message': event_data})}\n\n"
                                    
                                # Handle all other event types
                                else:
                                    event_obj.data = event_data
                                    yield f"data: {json.dumps({'type': 'agent_event', 'event_type': event_type, 'data': event_data})}\n\n"
                                
                                # Store the event for later saving
                                agent_events.append(event_obj)
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
                                
                                # Remove the reasoning part from the chunk before processing the rest
                                chunk = chunk[:start_idx - len("<reasoning>")] + chunk[end_idx + len("</reasoning>"):]
                            
                            # If chunk still has content, process it normally
                            if chunk.strip():
                                token_data = {
                                    "type": "token",
                                    "content": chunk
                                }
                                response_buffer += chunk
                                yield f"data: {json.dumps(token_data)}\n\n"
                            
                        elif "Step " in chunk and "Reasoning:" in chunk:
                            # This is thinking/reasoning content
                            reasoning_data = {
                                "type": "reasoning",
                                "content": chunk
                            }
                            # Accumulate reasoning in the buffer
                            reasoning_buffer += chunk + "\n"
                            yield f"data: {json.dumps(reasoning_data)}\n\n"
                            
                        else:
                            # This is a token from the LLM - send immediately without buffering
                            token_data = {
                                "type": "token",
                                "content": chunk
                            }
                            response_buffer += chunk
                            yield f"data: {json.dumps(token_data)}\n\n"
                    
                    # Once streaming is complete, save the AI response to the database
                    if request.save_to_db and chat_session_id and request.user_id:
                        try:
                            # Create message data
                            ai_message_data = ChatMessageCreate(
                                chat_session_id=chat_session_id,
                                space_id=request.space_id,
                                user_id=request.user_id,
                                content=response_buffer.strip(),
                                is_user=False,
                                workflow=agent_events and [event.dict() for event in agent_events],
                                reasoning={"content": reasoning_buffer.strip()} if reasoning_buffer.strip() else None
                            )
                            
                            # Save the message
                            await supabase_client.save_chat_message(ai_message_data)
                            logger.info(f"Saved AI response to chat session: {chat_session_id}")
                        except Exception as e:
                            logger.error(f"Error saving AI response: {e}")
                            # Continue even if saving the message fails
                    
                    # Send a final "done" event
                    query_time_ms = int((time.time() - start_time) * 1000)
                    done_data = {
                        "type": "done",
                        "query_time_ms": query_time_ms,
                        "chat_session_id": chat_session_id
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
        
        # If we're saving to the database and have a chat session ID
        if request.save_to_db and chat_session_id and request.user_id:
            try:
                # Create AI message data
                ai_message_data = ChatMessageCreate(
                    chat_session_id=chat_session_id,
                    space_id=request.space_id,
                    user_id=request.user_id,
                    content=response_text.strip(),
                    is_user=False,
                    workflow=None,  # No workflow data for non-streaming responses
                    reasoning={"content": reasoning} if reasoning else None
                )
                
                # Save the AI message
                await supabase_client.save_chat_message(ai_message_data)
                logger.info(f"Saved AI response to chat session: {chat_session_id}")
            except Exception as e:
                logger.error(f"Error saving AI response: {e}")
                # Continue even if saving the message fails
        
        # The response_data is now a dict with 'response' and 'thinking' keys
        return AgentResponse(
            success=True,
            response=response_text.strip(),
            chat_session_id=chat_session_id,
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