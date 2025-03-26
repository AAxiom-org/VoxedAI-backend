# Agent Implementation Plan

## Overview

This document outlines the plan to implement the agent endpoint using the PocketFlow framework. The agent will follow the "Decision Node → RAG → Tool Shed → Finish" workflow as described in the Requirements. The implementation will use the PocketFlow graph-based approach where nodes are connected to create a flow.

## Understanding the Core Components

### 1. Decision Node
- Uses google/gemini-2.0-flash-001 to decide the next action (RAG, Tool, or Finish)
- Acts as a router in the flow
- Makes decisions based on user query and context

### 2. RAG Node
- Responsible for retrieving context from database/files
- Will initially return a placeholder message

### 3. Tool Shed Node
- Collection of specialized functions/tools
- Will initially return a placeholder message

### 4. Finish Node
- Uses a Base LLM to generate the final answer
- Combines all context gathered

## Implementation Steps

### Step 1: Update Schema Definitions

Update the `app/schemas/agent.py` file to include all necessary fields for the agent request and response:

- `AgentRequest` schema should include:
  - `space_id` - Identifies which Space the query pertains to
  - `query` - The user's question or command
  - `view` - (Optional) The content the user is actively working with
  - `stream` - Boolean indicating whether response should be streamed

- `AgentResponse` schema should include:
  - `success` - Status of the request
  - `response` - Final answer or streamer object
  - `metadata` - Additional information

### Step 2: Create PocketFlow Implementation

Use existing directory `app/agents/base` with:

1. **nodes.py**: Define all nodes for the workflow
   - `DecisionNode`: Using google/gemini-2.0-flash-001 to determine next action
   - `RagNode`: For retrieving context (placeholder for now)
   - `ToolShedNode`: For executing tools (placeholder for now)
   - `FinishNode`: For generating final answer using LLM

2. **flow.py**: Define the flow, connecting all nodes together
   - Decision node as the starting point
   - Connect nodes with proper action-based branching

3. **utils.py**: Helper utilities specific to the agent

### Step 3: Add google/gemini-2.0-flash-001 to LLM Service

Update `app/services/llm_service.py` to add a function for the google/gemini-2.0-flash-001 decision model:
- Implement `get_decision` function to call google/gemini-2.0-flash-001
- Ensure proper parameters (temperature, etc.)
- Handle response parsing for decision outcomes (yaml)

### Step 4: Implement Agent Controller

Update `app/api/v1/endpoints/agent.py` to:
- Process agent requests
- Set up the PocketFlow nodes and flow
- Handle response streaming
- Manage error conditions

### Step 5: Connect to Data Layer

Ensure the agent can access:
- Space data from Supabase
- User's toggled files
- User's toggled research

### Step 6: Testing and Validation

- Create test cases for each component
- Verify the flow works correctly
- Ensure error handling is robust

## Code Structure Outline

```
app/
└── agents/
    └── base/
        ├── __init__.py
        ├── nodes.py      # All node definitions
        ├── flow.py       # Flow construction logic
        └── utils.py      # Agent-specific utilities
```

## Detailed Node Implementation Plan

### DecisionNode
- **Input**: User query + current context
- **Process**: Call google/gemini-2.0-flash-001 via `llm_service.get_decision()`
- **Output**: Return action (rag/tool/finish)

### RagNode (Placeholder)
- **Input**: User query + space information
- **Process**: Return placeholder message
- **Output**: Retrieved context information. Routes back to decision for loop

### ToolShedNode (Placeholder)
- **Input**: User query + current context
- **Process**: Return placeholder message
- **Output**: Tool execution results. Routes back to decision for loop

### FinishNode
- **Input**: User query + all gathered context
- **Process**: Call LLM to generate final answer
- **Output**: Formatted response for user

## Action Items Checklist

- [ ] Update `app/schemas/agent.py` with complete request/response schemas
- [ ] Create `app/agents/pocketflow_agent` directory
- [ ] Implement `nodes.py` with all node definitions
- [ ] Implement `flow.py` to connect nodes in PocketFlow graph
- [ ] Add `get_decision` function to `llm_service.py` for google/gemini-2.0-flash-001
- [ ] Update agent endpoint in `app/api/v1/endpoints/agent.py`
- [ ] Write placeholder implementations for RAG and Tool Shed
- [ ] Implement basic response streaming
- [ ] Write validation and error handling
- [ ] Test the complete flow 