"""
Node implementations for the agent workflow using PocketFlow.
"""
from typing import Dict, Any, List
import logging

from pocketflow import Node, AsyncNode
from app.services.llm_service import llm_service
from app.core.logging import logger
from app.toolshed.flow import run_toolshed_flow


class DecisionNode(AsyncNode):
    """
    Decision node that determines the next action in the workflow.
    Uses google/gemini-2.0-flash-001 to decide between RAG, Tool, or Finish actions.
    """
    
    async def prep_async(self, shared):
        # Get the user query and any context gathered so far
        query = shared.get("query", "")
        context = shared.get("context", {})
        action_history = shared.get("action_history", [])
        stream = shared.get("stream", False)
        
        logger.info(f"DecisionNode: Processing query '{query}' (stream={stream})")
        
        return {
            "query": query,
            "context": context,
            "action_history": action_history,
            "stream": stream
        }
    
    async def exec_async(self, prep_res):
        query = prep_res["query"]
        context = prep_res["context"]
        action_history = prep_res["action_history"]
        stream = prep_res.get("stream", False)
        
        # Use the LLM service to get the decision
        decision = await llm_service.get_decision(
            query=query,
            context=context,
            action_history=action_history,
            stream=stream
        )
        
        return decision
    
    async def post_async(self, shared, prep_res, decision):
        # Store the decision in the shared context
        action_history = shared.get("action_history", [])
        thinking_history = shared.get("thinking_history", [])
        
        # Store the thinking separately for client exposure
        if "thinking" in decision:
            thinking = decision.get("thinking", "")
            thinking_history.append({
                "step": len(action_history) + 1,
                "thinking": thinking
            })
            shared["thinking_history"] = thinking_history
            
            # Also add it to the context for the final response
            context = shared.get("context", {})
            if "decision_thinking" not in context:
                context["decision_thinking"] = []
            context["decision_thinking"].append({
                "step": len(action_history) + 1,
                "thinking": thinking
            })
            shared["context"] = context
            
            # If we have an event queue, send the thinking step
            if "event_queue" in shared and shared["event_queue"] is not None:
                event = {
                    "type": "thinking",
                    "step": len(action_history) + 1,
                    "thinking": thinking
                }
                await shared["event_queue"].put(event)
        
        action_history.append(decision)
        shared["action_history"] = action_history
        shared["last_decision"] = decision
        
        # Get the action from the decision
        action = decision.get("action", "finish")
        logger.info(f"DecisionNode: Next action '{action}'")
        
        # If we have an event queue, send the decision event
        if "event_queue" in shared and shared["event_queue"] is not None:
            event = {
                "type": "decision",
                "decision": action
            }
            await shared["event_queue"].put(event)
        
        # Return the action as the next node to transition to
        return action


class RAGNode(AsyncNode):
    """
    Retrieval-Augmented Generation node.
    Retrieves relevant context from the knowledge base.
    Currently a placeholder that returns a message.
    """
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        space_id = shared.get("space_id", "")
        stream = shared.get("stream", False)
        
        logger.info(f"RAGNode: Processing query for space '{space_id}' (stream={stream})")
        
        return {
            "query": query,
            "context": context,
            "space_id": space_id,
            "stream": stream
        }
    
    async def exec_async(self, prep_res):
        # Placeholder implementation
        return {
            "message": "RAG has not been implemented yet, continue",
            "result_type": "placeholder"
        }
    
    async def post_async(self, shared, prep_res, results):
        # Update the context with the RAG results
        context = shared.get("context", {})
        context["rag_results"] = results
        shared["context"] = context
        
        # If we have an event queue, send the RAG complete event
        if "event_queue" in shared and shared["event_queue"] is not None:
            event = {
                "type": "rag_complete",
                "message": results.get("message", "RAG processing complete")
            }
            await shared["event_queue"].put(event)
        
        # Always go back to the decision node
        return "decide"


class ToolShedNode(AsyncNode):
    """
    Node for executing tools based on the decision made.
    """
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        action_history = shared.get("action_history", [])
        active_file_id = shared.get("active_file_id")
        space_id = shared.get("space_id")
        user_id = shared.get("user_id")
        stream = shared.get("stream", False)
        
        logger.info(f"ToolShedNode: Processing query for toolshed, active_file_id={active_file_id}")
        
        return {
            "query": query,
            "context": context,
            "action_history": action_history,
            "active_file_id": active_file_id,
            "space_id": space_id,
            "user_id": user_id,
            "stream": stream,
            "_shared": shared  # Store the shared context for later access
        }
    
    async def exec_async(self, prep_res):
        """Execute the tool and return results."""
        query = prep_res.get("query", "")
        context = prep_res.get("context", {})
        action_history = prep_res.get("action_history", [])
        active_file_id = prep_res.get("active_file_id")
        space_id = prep_res.get("space_id")
        user_id = prep_res.get("user_id")
        stream = prep_res.get("stream", False)
        shared = prep_res.get("_shared", {})  # Get the shared context
        
        # If we have an event queue, send the tool execution start event
        if shared and "event_queue" in shared and shared["event_queue"] is not None:
            event = {
                "type": "tool_execution_start",
                "message": "Starting tool selection and execution process",
                "query": query
            }
            await shared["event_queue"].put(event)
        
        try:
            from app.toolshed.flow import run_toolshed_flow
            
            # Run the toolshed flow to execute the selected tool
            tool_results = await run_toolshed_flow(
                query=query,
                context=context,
                action_history=action_history,
                stream=stream,
                active_file_id=active_file_id,
                space_id=space_id,
                user_id=user_id,
                event_queue=shared.get("event_queue") if shared else None  # Pass the event queue to toolshed
            )
            
            # Add extensive debugging about tool_results
            logger.info(f"ToolShedNode: Tool execution completed. Results type: {type(tool_results)}")
            logger.info(f"ToolShedNode: Tool execution results: {tool_results}")
            
            if tool_results is None:
                logger.error("ToolShedNode: Tool results is None. Setting default response.")
                
                # Send error event
                if shared and "event_queue" in shared and shared["event_queue"] is not None:
                    event = {
                        "type": "tool_execution_error",
                        "message": "Tool execution returned no results",
                        "error": "No results returned"
                    }
                    await shared["event_queue"].put(event)
                    
                return {
                    "result_type": "error",
                    "error": "Tool execution returned no results",
                    "tool_name": "unknown"
                }
                
            return tool_results
            
        except Exception as e:
            # Handle any exceptions from tool execution
            logger.error(f"Error executing tool: {str(e)}")
            import traceback
            logger.error(f"Tool execution error traceback: {traceback.format_exc()}")
            
            # Send error event
            if shared and "event_queue" in shared and shared["event_queue"] is not None:
                event = {
                    "type": "tool_execution_error",
                    "message": f"Error executing tool: {str(e)}",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                await shared["event_queue"].put(event)
            
            return {
                "result_type": "error",
                "error": f"Error executing tool: {str(e)}",
                "tool_name": "unknown"
            }
    
    async def post_async(self, shared, prep_res, tool_results):
        """Process tool results and decide next step."""
        # Add tool results to context
        context = shared.get("context", {})
        context["tool_results"] = tool_results
        shared["context"] = context
        
        # Add tool action to action history
        action_history = shared.get("action_history", [])
        action_history.append({
            "action": "tool",
            "tool_name": tool_results.get("tool_name", "unknown"),
            "result_type": tool_results.get("result_type", "unknown"),
            "parameters": tool_results.get("parameters", {})
        })
        shared["action_history"] = action_history
        
        # Add an event to the queue if streaming is enabled and the queue exists
        if shared.get("stream", False) and "event_queue" in shared:
            # If tool is making file edits, send special markers
            if tool_results.get("result_type") == "file_edit" and "file_id" in tool_results:
                try:
                    # First, send a marker that edit is starting
                    start_event = {
                        "type": "file_edit_start",
                        "file_id": tool_results["file_id"]
                    }
                    await shared["event_queue"].put(start_event)
                    
                    # Then, after a short delay, send a marker that edit is complete
                    complete_event = {
                        "type": "file_edit_complete",
                        "file_id": tool_results["file_id"]
                    }
                    await shared["event_queue"].put(complete_event)
                except Exception as e:
                    logger.error(f"Error sending file edit events: {str(e)}")
            
            # Send a general tool complete event
            try:
                tool_event = {
                    "type": "tool_complete",
                    "tool": tool_results.get("tool_name", "unknown")
                }
                await shared["event_queue"].put(tool_event)
            except Exception as e:
                logger.error(f"Error sending tool complete event: {str(e)}")
        
        # Always go back to the decision node after tool execution
        return "decide"


class FinishNode(AsyncNode):
    """
    Finish node that generates the final response using all gathered context.
    """
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        action_history = shared.get("action_history", [])
        stream = shared.get("stream", False)
        model_name = shared.get("model_name", "deepseek/deepseek-chat:free")
        
        logger.info(f"FinishNode: Generating final response (stream={stream})")
        
        return {
            "query": query,
            "context": context,
            "action_history": action_history,
            "stream": stream,
            "model_name": model_name
        }
    
    async def exec_async(self, prep_res):
        query = prep_res["query"]
        context = prep_res.get("context", {})
        stream = prep_res.get("stream", False)
        model_name = prep_res.get("model_name", "deepseek/deepseek-chat:free")
        # Format the context for the final response
        rag_results = context.get("rag_results", {}).get("message", "No RAG results available")
        tool_results = context.get("tool_results", {}).get("message", "No tool results available")
        
        # Also include any thinking/reasoning from decision steps
        decision_thinking = ""
        if "decision_thinking" in context:
            for item in context["decision_thinking"]:
                decision_thinking += f"Step {item['step']} thinking:\n{item['thinking']}\n\n"
        
        prompt = f"""
        # USER QUERY
        {query}
        
        # CONTEXT INFORMATION
        RAG Results: {rag_results}
        Tool Results: {tool_results}
        
        # DECISION REASONING
        {decision_thinking or "No decision reasoning available"}
        
        # TASK
        Based on the user's query and the context information, provide a comprehensive and accurate response.
        Be direct, concise, and helpful. If you cannot provide a complete answer due to missing information,
        acknowledge this and provide the best response possible with the available information.
        """
                
        # Generate the final response
        response_text = await llm_service._call_llm(
            prompt=prompt,
            stream=stream,
            temperature=0.3,
            max_tokens=2048,
            model_name=model_name
        )
        
        return response_text
    
    async def post_async(self, shared, prep_res, response_text):
        # Set the final response in the shared context
        shared["final_response"] = response_text
        
        # If we have an event queue, send the flow complete event
        if "event_queue" in shared and shared["event_queue"] is not None:
            event = {
                "type": "flow_complete",
                "message": "Agent flow completed successfully"
            }
            await shared["event_queue"].put(event)
        
        # Return finish to indicate completion
        return "complete"
