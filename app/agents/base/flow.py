# DecisionModel - "tool" >> ToolShed
# DecisionModel - "rag" >> RAGService
# DecisionModel - "finish" >> Finish

# ToolShed - "decide" >> DecisionModel
# RAGService - "decide" >> DecisionModel

from pocketflow import Flow
from nodes import DecisionModel, ToolShed, RAGService, Finish

def create_agent_flow():
    """
    Create and connect the nodes to form a complete agent flow.
    
    The flow works like this:
    1. DecideAction node decides whether to use a tool or rag
    2. If tool, use ToolShed to execute the tool then go back to DecideAction
    3. If rag, use RAGService to search the knowledge base then go back to DecideAction
    4. If finish, return the result
    
    Returns:
        Flow: A complete base agent flow
    """
    # Create instances of each node
    decide = DecisionModel()
    tool = ToolShed()
    rag = RAGService()
    finish = Finish()
    
    # Connect the nodes
    # If DecideAction returns "tool", go to ToolShed
    decide - "tool" >> tool
    
    # If DecideAction returns "rag", go to RAGService
    decide - "rag" >> rag
    
    # If DecideAction returns "finish", go to Finish
    decide - "finish" >> finish
    
    # After ToolShed completes and returns "decide", go back to DecideAction
    tool - "decide" >> decide
    rag - "decide" >> decide
    
    # Create and return the flow, starting with the DecideAction node
    return Flow(start=decide)