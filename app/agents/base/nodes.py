from typing import List, Dict, Any
from app.services.llm_service import llm_service
from app.services.tool_shed import tool_shed
from app.services.rag_service import rag_service

from pocketflow import Node
import yaml

class DecisionNode(Node):
    def prep(self, shared):
        # Get the task and current context
        context = {
            "task": shared.get("task", ""),
            "previous_actions": shared.get("action_history", []),
            "current_state": shared.get("current_state", {}),
        }
        return context
    
    def exec(self, context):
        prompt = f"""
### CONTEXT
Task: {context['task']}
Previous Actions: {context['previous_actions']}
Current State: {context['current_state']}

### ACTION SPACE
[1] search_web
    Description: Search the web for information
    Parameters:
        - query (str): What to search for

[2] calculate
    Description: Perform a calculation
    Parameters:
        - expression (str): Math expression to evaluate

[3] finish
    Description: Complete the task and provide final answer
    Parameters:
        - result (str): Final answer/result

### NEXT ACTION
Based on the current context and available actions, decide what to do next.
Return your response in YAML format:

```yaml
thinking: |
    <your step-by-step reasoning>
action: <action_name>
parameters:
    <parameter_name>: <parameter_value>
```
"""
        response = llm_service.get_decision(prompt)
        # Extract YAML part
        yaml_str = response.split("```yaml")[1].split("```")[0].strip()
        decision = yaml.safe_load(yaml_str)
        
        # Validate decision format
        assert "action" in decision, "Decision must include action"
        assert "parameters" in decision, "Decision must include parameters"
        assert decision["action"] in ["search_web", "calculate", "finish"]
        
        return decision

    def post(self, shared, prep_res, decision):
        # Store the decision in action history
        history = shared.get("action_history", [])
        history.append(decision)
        shared["action_history"] = history
        shared["last_decision"] = decision
        
        # Return the action name as the next transition
        return decision["action"]
        
class ToolShed:
    def __init__(self, input: str):
        self.input = input
    
    def exec(self):
        tools = tool_shed.get_tools()
        tool = llm_service.get_tool(self.input, tools)
        results = tool.execute(self.input)
        return results
    
    def post(self, shared, prep_res, results):
        current_state = shared.get("current_state", {})
        current_state["last_search_results"] = results
        shared["current_state"] = current_state
        return "decide"
        
class RAGService:
    def __init__(self, query: str):
        self.query = query
        
    def exec(self):
        rag_query = llm_service.get_rag_query(self.query)
        results = rag_service.search(rag_query)
        return results
    
    def post(self, shared, prep_res, results):
        current_state = shared.get("current_state", {})
        current_state["last_search_results"] = results
        shared["current_state"] = current_state
        return "decide"
    
    
class Finish:
    def __init__(self, input: str):
        self.input = input
        
    def exec(self):
        return llm_service.generate_answer(self.input)
    
    def post(self, shared, prep_res, results):
        current_state = shared.get("current_state", {})
        current_state["last_search_results"] = results
        shared["current_state"] = current_state
        return "finish"
