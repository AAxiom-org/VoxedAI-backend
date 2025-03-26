"""
Node implementations for the agent workflow using PocketFlow.
"""
from typing import Dict, Any, List
import logging

from pocketflow import Node, AsyncNode
from app.services.llm_service import llm_service
from app.core.logging import logger


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
        
        action_history.append(decision)
        shared["action_history"] = action_history
        shared["last_decision"] = decision
        
        # Return the action as the next node to transition to
        action = decision.get("action", "finish")
        logger.info(f"DecisionNode: Next action '{action}'")
        
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
        
        # Always go back to the decision node
        return "decide"


class ToolShedNode(AsyncNode):
    """
    Tool Shed node that executes specialized tools.
    Currently a placeholder that returns a message.
    """
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        last_decision = shared.get("last_decision", {})
        tool_params = last_decision.get("parameters", {})
        stream = shared.get("stream", False)
        
        logger.info(f"ToolShedNode: Executing tool with parameters: {tool_params} (stream={stream})")
        
        return {
            "query": query,
            "context": context,
            "tool_params": tool_params,
            "stream": stream
        }
    
    async def exec_async(self, prep_res):
        # Placeholder implementation
        return {
            "message": "Tool Shed has not been implemented yet, continue",
            "result_type": "placeholder"
        }
    
    async def post_async(self, shared, prep_res, results):
        # Update the context with the tool results
        context = shared.get("context", {})
        context["tool_results"] = results
        shared["context"] = context
        
        # Always go back to the decision node
        return "decide"


class FinishNode(AsyncNode):
    """
    Finish node that generates the final response using all gathered context.
    """
    
    async def prep_async(self, shared):
        query = shared.get("query", "")
        context = shared.get("context", {})
        action_history = shared.get("action_history", [])
        view = shared.get("view", "")
        stream = shared.get("stream", False)
        
        logger.info(f"FinishNode: Generating final response (stream={stream})")
        
        return {
            "query": query,
            "context": context,
            "action_history": action_history,
            "view": view,
            "stream": stream
        }
    
    async def exec_async(self, prep_res):
        query = prep_res["query"]
        context = prep_res.get("context", {})
        view = prep_res.get("view", "")
        stream = prep_res.get("stream", False)
        
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
        
        # USER VIEW (active content)
        {view or "No active content"}
        
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
            max_tokens=2048
        )
        
        return response_text
    
    async def post_async(self, shared, prep_res, response_text):
        # Set the final response in the shared context
        shared["final_response"] = response_text
        
        # Return finish to indicate completion
        return "complete"
