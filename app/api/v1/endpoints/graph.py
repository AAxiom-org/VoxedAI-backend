"""
API endpoints for graph operations.
"""
import json
import time
import uuid
import yaml
from typing import Any, Dict, List, Optional, Tuple
import traceback

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.services.llm_service import llm_service
from app.db.supabase import supabase_client

router = APIRouter()


class GraphGenerationRequest(BaseModel):
    """Schema for graph generation request."""
    space_id: str = Field(..., description="ID of the space to generate graph for")
    user_id: str = Field(..., description="ID of the user")
    stream: bool = Field(True, description="Whether to stream the response")


class GraphGenerationResponse(BaseModel):
    """Schema for graph generation response."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    graph_id: Optional[str] = Field(None, description="ID of the generated graph")


@router.post("/generate", response_model=GraphGenerationResponse)
async def generate_graph(
    request: GraphGenerationRequest,
    background_tasks: BackgroundTasks
) -> Any:
    """
    Generates a knowledge graph based on user's notes in a space.
    
    This endpoint:
    1. Retrieves all notes from the user's space
    2. Processes them with an LLM to generate a knowledge graph structure
    3. Generates detailed research content for each node
    4. Stores the graph and research content in the database
    
    Args:
        request: Contains space_id, user_id, and stream parameters
        background_tasks: For handling cleanup tasks
        
    Returns:
        GraphGenerationResponse or StreamingResponse: Contains the operation status
    """
    try:
        start_time = time.time()
        
        # If streaming is requested, set up a streaming response with SSE
        if request.stream:
            async def event_generator():
                try:
                    # Send initial status
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Starting graph generation...'})}\n\n"
                    
                    # Step 1: Fetch all notes from the space
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching notes from space...'})}\n\n"
                    
                    notes = await fetch_space_notes(request.space_id, request.user_id)
                    if not notes:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'No notes found in space'})}\n\n"
                        return
                    
                    note_count = len(notes)
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Found {note_count} notes in space'})}\n\n"
                    
                    # Step 2: Generate graph structure using LLM
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing notes and generating graph structure...'})}\n\n"
                    
                    graph_data, node_map = await generate_graph_structure_from_notes(notes, request.space_id)
                    
                    if not graph_data:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate graph structure'})}\n\n"
                        return
                    
                    # Step 3: Store graph in database
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Storing graph structure in database...'})}\n\n"
                    
                    graph_id = await store_graph_in_database(graph_data, request.space_id, request.user_id)
                    
                    if not graph_id:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to store graph in database'})}\n\n"
                        return
                    
                    # Step 4: Generate research content for each node
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Generating research content for each node...'})}\n\n"
                    
                    research_entries = await generate_research_content_for_graph(notes, graph_data, node_map, request.space_id)
                    
                    if not research_entries:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to generate research content'})}\n\n"
                        return
                    
                    # Step 5: Store research entries in database
                    entry_count = len(research_entries)
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Storing {entry_count} research entries in database...'})}\n\n"
                    
                    success = await store_research_entries(research_entries, request.space_id, request.user_id)
                    
                    if not success:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to store research entries in database'})}\n\n"
                        return
                    
                    # Calculate query time
                    query_time_ms = int((time.time() - start_time) * 1000)
                    
                    # Send a final "done" event
                    yield f"data: {json.dumps({'type': 'done', 'message': 'Graph generation complete', 'query_time_ms': query_time_ms, 'graph_id': graph_id})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in graph generation: {str(e)}")
                    error_data = {
                        "type": "error",
                        "message": f"Error generating graph: {str(e)}"
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        
        # For non-streaming responses
        try:
            # Step 1: Fetch all notes from the space
            notes = await fetch_space_notes(request.space_id, request.user_id)
            if not notes:
                return GraphGenerationResponse(
                    success=False,
                    message="No notes found in space",
                    graph_id=None
                )
            
            # Step 2: Generate graph structure using LLM
            graph_data, node_map = await generate_graph_structure_from_notes(notes, request.space_id)
            
            if not graph_data:
                return GraphGenerationResponse(
                    success=False,
                    message="Failed to generate graph structure",
                    graph_id=None
                )
            
            # Step 3: Store graph in database
            graph_id = await store_graph_in_database(graph_data, request.space_id, request.user_id)
            
            if not graph_id:
                return GraphGenerationResponse(
                    success=False,
                    message="Failed to store graph in database",
                    graph_id=None
                )
            
            # Step 4: Generate research content for each node
            research_entries = await generate_research_content_for_graph(notes, graph_data, node_map, request.space_id)
            
            if not research_entries:
                return GraphGenerationResponse(
                    success=False,
                    message="Failed to generate research content",
                    graph_id=graph_id
                )
            
            # Step 5: Store research entries in database
            success = await store_research_entries(research_entries, request.space_id, request.user_id)
            
            if not success:
                return GraphGenerationResponse(
                    success=False,
                    message="Failed to store research entries in database",
                    graph_id=graph_id
                )
            
            # Return success response
            return GraphGenerationResponse(
                success=True,
                message="Graph generation complete",
                graph_id=graph_id
            )
        
        except Exception as e:
            logger.error(f"Error in non-streaming graph generation: {str(e)}")
            return GraphGenerationResponse(
                success=False,
                message=f"Error generating graph: {str(e)}",
                graph_id=None
            )
    
    except Exception as e:
        logger.error(f"Error in graph generation endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating graph: {str(e)}"
        )


async def fetch_space_notes(space_id: str, user_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all notes from a space.
    
    Args:
        space_id: The ID of the space
        user_id: The ID of the user
        
    Returns:
        List[Dict[str, Any]]: List of notes
    """
    try:
        # Query the space_files table for all notes in the space
        client = supabase_client.client
        response = client.table("space_files").select("*").eq("space_id", space_id).eq("is_note", True).execute()
        
        if not response.data:
            logger.warning(f"No notes found in space {space_id}")
            return []
        
        # Process each note to extract content
        notes = []
        for note in response.data:
            note_content = note.get("note_content")
            if note_content:
                notes.append({
                    "id": note["id"],
                    "file_name": note["file_name"],
                    "file_path": note["file_path"],
                    "content": note_content,
                    "metadata": note.get("metadata", {})
                })
        
        logger.info(f"Found {len(notes)} notes in space {space_id}")
        return notes
    
    except Exception as e:
        logger.error(f"Error fetching space notes: {e}")
        raise


async def generate_graph_structure_from_notes(notes: List[Dict[str, Any]], space_id: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Generates a knowledge graph structure from notes using LLM.
    This function focuses only on creating the graph structure without detailed content.
    
    Args:
        notes: List of notes
        space_id: The ID of the space
        
    Returns:
        Tuple[Dict[str, Any], Dict[str, List[str]]]: The graph data and node-to-note mapping
    """
    try:
        # Prepare template examples
        template_graph, template_entries = await fetch_template_examples()
        
        # Prepare notes context
        notes_context = ""
        for i, note in enumerate(notes):
            notes_context += f"NOTE {i+1}:\nTitle: {note['file_name']}\nID: {note['id']}\nContent: {note['content']}\n\n"
        
        # Prepare the prompt for the LLM
        prompt = f"""
        # Graph Structure Generation Task
        
        You are a knowledge graph agent that analyzes notes and creates a connected graph structure. Your task is to:
        
        1. Analyze the notes content
        2. Extract key concepts and relationships
        3. Generate a graph with nodes representing concepts and links representing relationships
        4. For each node, track which source note IDs are related to it
        
        ## Notes Content
        
        {notes_context}
        
        ## Template Examples
        
        Here is an example of how to structure the graph:
        ```json
        {json.dumps(template_graph, indent=2)}
        ```
        
        ## Output Format
        
        Provide your output in YAML format:
        
        ```yaml
        graph:
          mainGraph:
            nodes:
              - id: "concept1"
                size: 15
                color: "#4361EE"
                group: 1
                label: "Concept 1 Name"
                noteId: "concept1"
              # More nodes...
            links:
              - source: "concept1"
                target: "concept2"
              # More links...
          detailedGraphs:
            concept1:
              nodes:
                - id: "concept1"
                  size: 20
                  type: "circle"
                  group: 1
                  label: "Concept 1 Name"
                  noteId: "concept1"
                # More nodes...
              links:
                - source: "concept1"
                  target: "note1"
                # More links...
            # More detailed graphs...
        node_map:
          concept1:
            - "note_id_1"
            - "note_id_2"
          # Map of node IDs to related note IDs
        ```
        
        Use ONLY the format shown in the example, but replace with ACTUAL content based on the notes. Generate a meaningful graph that connects the concepts found in the notes.
        """
        
        # Call the LLM with a model that has a very large context window
        logger.info(f"Calling LLM to generate graph structure from {len(notes)} notes...")
        response_text = await llm_service._call_llm(
            prompt=prompt,
            model_name="google/gemini-2.0-flash-exp:free",  # Very large context window
            stream=False,  # We don't need streaming here
            temperature=0.2,  # Slightly creative but mostly deterministic
            max_tokens=100000  # Large output size to handle the graph
        )
        
        # Extract YAML from the LLM response
        yaml_str = extract_yaml_from_text(response_text)
        result = yaml.safe_load(yaml_str)
        
        graph_data = result.get("graph", {})
        node_map = result.get("node_map", {})
        
        logger.info(f"Generated graph structure with {len(graph_data.get('mainGraph', {}).get('nodes', []))} main nodes")
        
        return graph_data, node_map
    
    except Exception as e:
        logger.error(f"Error generating graph structure from notes: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def generate_research_content_for_graph(
    notes: List[Dict[str, Any]], 
    graph_data: Dict[str, Any], 
    node_map: Dict[str, List[str]], 
    space_id: str
) -> List[Dict[str, Any]]:
    """
    Generates research content for each node in the graph.
    
    Args:
        notes: List of notes
        graph_data: The graph structure data
        node_map: Mapping from node IDs to related note IDs
        space_id: The ID of the space
        
    Returns:
        List[Dict[str, Any]]: The research entries
    """
    try:
        # Prepare template examples
        _, template_entries = await fetch_template_examples()
        
        # Create a map of note IDs to note content for lookup
        note_content_map = {note["id"]: note for note in notes}
        
        # Organize nodes from the graph
        main_nodes = graph_data.get("mainGraph", {}).get("nodes", [])
        
        # Prepare research entries for each node
        research_entries = []
        
        # Process each main node (concept)
        for node in main_nodes:
            node_id = node.get("id")
            if not node_id:
                continue
                
            # Get related note IDs for this node
            related_note_ids = node_map.get(node_id, [])
            
            # Get the detailed graph for this node if it exists
            detailed_graph = graph_data.get("detailedGraphs", {}).get(node_id, {})
            detailed_nodes = detailed_graph.get("nodes", [])
            
            # Prepare context for the LLM with the related notes
            node_context = f"CONCEPT: {node.get('label', 'Unnamed Concept')}\n\n"
            
            # Add related notes content
            for note_id in related_note_ids:
                if note_id in note_content_map:
                    note = note_content_map[note_id]
                    node_context += f"RELATED NOTE: {note['file_name']}\nCONTENT: {note['content']}\n\n"
            
            # Prepare the prompt for the LLM
            prompt = f"""
            # Research Content Generation Task
            
            You are a knowledge graph content creator. Your task is to create detailed markdown content for a concept node in a knowledge graph.
            
            ## Node Information
            
            {node_context}
            
            ## Detailed Graph Structure
            
            This node has the following sub-nodes in its detailed view:
            
            {json.dumps(detailed_nodes, indent=2)}
            
            ## Template Example
            
            Here is an example of a research entry:
            ```json
            {json.dumps(template_entries[0] if template_entries else {}, indent=2)}
            ```
            
            ## Output Format
            
            Provide your output in YAML format:
            
            ```yaml
            id: "{node_id}"
            content: |
              # Markdown Title
              
              Rich markdown content with sections, bullet points, etc.
              
              ## Section 1
              
              Content for section 1...
              
              ## Section 2
              
              Content for section 2...
            metadata:
              id: "{node_id}"
              color: "{node.get('color', '#4361EE')}"
              group: {node.get('group', 1)}
              label: "{node.get('label', 'Concept')}"
            related_data:
              type: "concept"
              notes:
                - "sub_note_id_1"
                - "sub_note_id_2"
            related_note_ids:
              - "{related_note_ids[0] if related_note_ids else ''}"
              # List all related note IDs here
            ```
            
            Create rich, informative markdown content based on the related notes. Use proper formatting, sections, and include relevant information from the notes.
            """
            
            # Call the LLM to generate content for this node
            logger.info(f"Generating research content for node {node_id}...")
            response_text = await llm_service._call_llm(
                prompt=prompt,
                model_name="google/gemini-2.0-flash-exp:free",
                stream=False,
                temperature=0.4,  # Slightly more creative for content
                max_tokens=30000  # Large enough for detailed content
            )
            
            # Extract YAML from the LLM response
            yaml_str = extract_yaml_from_text(response_text)
            entry_data = yaml.safe_load(yaml_str)
            
            # Ensure related_note_ids is properly formatted
            if "related_note_ids" in entry_data:
                related_note_ids = entry_data["related_note_ids"]
                # Filter out any empty strings
                related_note_ids = [note_id for note_id in related_note_ids if note_id]
                entry_data["related_note_ids"] = related_note_ids
            else:
                # If not provided by the LLM, use our node_map
                entry_data["related_note_ids"] = node_map.get(node_id, [])
            
            research_entries.append(entry_data)
            
            # Now process sub-nodes (notes) in the detailed graph
            if detailed_nodes:
                for detail_node in detailed_nodes:
                    # Skip the main concept node (already processed)
                    if detail_node.get("id") == node_id:
                        continue
                        
                    detail_id = detail_node.get("id")
                    if not detail_id or detail_node.get("type") != "text":
                        continue
                        
                    # Find related sub-note content
                    sub_note_context = f"SUB-NOTE: {detail_node.get('label', 'Unnamed Sub-Note')}\nPARENT CONCEPT: {node.get('label', 'Unnamed Concept')}\n\n"
                    
                    # Generate content for the sub-note
                    sub_prompt = f"""
                    # Sub-Note Content Generation Task
                    
                    You are creating content for a sub-note in a knowledge graph. This is a specific point related to a larger concept.
                    
                    ## Sub-Note Information
                    
                    {sub_note_context}
                    
                    ## Parent Concept Content
                    
                    This sub-note is related to the following concept:
                    
                    {node_context}
                    
                    ## Output Format
                    
                    Provide your output in YAML format:
                    
                    ```yaml
                    id: "{detail_id}"
                    content: |
                      # {detail_node.get('label', 'Sub-Note')}
                      
                      Detailed markdown content about this specific aspect of the concept.
                      
                      ## Key Points
                      
                      - Point 1
                      - Point 2
                      
                      ## Further Details
                      
                      More detailed explanation...
                    metadata:
                      id: "{detail_id}"
                      group: {detail_node.get('group', node.get('group', 1))}
                      label: "{detail_node.get('label', 'Sub-Note')}"
                    related_data:
                      type: "note"
                      parent: "{node_id}"
                    related_note_ids:
                      - "{related_note_ids[0] if related_note_ids else ''}"
                      # List relevant related note IDs here
                    ```
                    
                    Create focused, detailed content specific to this sub-note topic.
                    """
                    
                    # Call the LLM to generate content for this sub-node
                    logger.info(f"Generating research content for sub-node {detail_id}...")
                    sub_response_text = await llm_service._call_llm(
                        prompt=sub_prompt,
                        model_name="google/gemini-2.0-flash-exp:free",
                        stream=False,
                        temperature=0.4,
                        max_tokens=15000
                    )
                    
                    # Extract YAML from the LLM response
                    sub_yaml_str = extract_yaml_from_text(sub_response_text)
                    sub_entry_data = yaml.safe_load(sub_yaml_str)
                    
                    # Ensure related_note_ids is properly formatted
                    if "related_note_ids" in sub_entry_data:
                        sub_related_note_ids = sub_entry_data["related_note_ids"]
                        # Filter out any empty strings
                        sub_related_note_ids = [note_id for note_id in sub_related_note_ids if note_id]
                        sub_entry_data["related_note_ids"] = sub_related_note_ids
                    else:
                        # Use a subset of the parent node's related notes, or an empty list
                        sub_entry_data["related_note_ids"] = node_map.get(node_id, [])[:1]
                    
                    research_entries.append(sub_entry_data)
        
        logger.info(f"Generated {len(research_entries)} research entries")
        return research_entries
    
    except Exception as e:
        logger.error(f"Error generating research content: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


async def fetch_template_examples() -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Fetches template examples from the database.
    
    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: Template graph and research entries
    """
    try:
        client = supabase_client.client
        
        # Fetch template graph
        template_space_id = "ff824499-7c30-4a41-870f-1e28e5661872"
        template_user_id = "template"
        
        # Get graph template
        graph_response = client.table("graphs").select("data").eq("space_id", template_space_id).eq("user_id", template_user_id).execute()
        
        template_graph = {}
        if graph_response.data:
            template_graph = graph_response.data[0].get("data", {})
        
        # Get research entry templates (limit to 5)
        research_response = client.table("space_research").select("*").eq("space_id", template_space_id).eq("user_id", template_user_id).limit(5).execute()
        
        template_entries = research_response.data or []
        
        return template_graph, template_entries
    
    except Exception as e:
        logger.error(f"Error fetching template examples: {e}")
        # Return empty templates if there's an error
        return {}, []


async def store_graph_in_database(graph_data: Dict[str, Any], space_id: str, user_id: str) -> Optional[str]:
    """
    Stores the generated graph in the database.
    
    Args:
        graph_data: The graph data
        space_id: The ID of the space
        user_id: The ID of the user
        
    Returns:
        Optional[str]: The ID of the stored graph, or None if storage failed
    """
    try:
        client = supabase_client.client
        
        # Check if a graph already exists for this space
        existing_response = client.table("graphs").select("id").eq("space_id", space_id).execute()
        
        if existing_response.data:
            # Update existing graph
            graph_id = existing_response.data[0]["id"]
            client.table("graphs").update({"data": graph_data}).eq("id", graph_id).execute()
            logger.info(f"Updated existing graph with ID {graph_id}")
        else:
            # Insert new graph
            insert_data = {
                "space_id": space_id,
                "user_id": user_id,
                "data": graph_data
            }
            response = client.table("graphs").insert(insert_data).execute()
            
            if response.data:
                graph_id = response.data[0]["id"]
                logger.info(f"Inserted new graph with ID {graph_id}")
            else:
                logger.error("Failed to insert graph: no data returned")
                return None
        
        return graph_id
    
    except Exception as e:
        logger.error(f"Error storing graph in database: {e}")
        return None


async def store_research_entries(entries: List[Dict[str, Any]], space_id: str, user_id: str) -> bool:
    """
    Stores the generated research entries in the database.
    
    Args:
        entries: The research entries
        space_id: The ID of the space
        user_id: The ID of the user
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        client = supabase_client.client
        
        # For each entry, either update existing or insert new
        for entry in entries:
            # Check if entry_id is a valid UUID; if not, generate a new one
            try:
                # Try to parse the ID as UUID to validate it
                entry_id_str = entry.get("id")
                if entry_id_str and uuid.UUID(entry_id_str):
                    entry_id = entry_id_str
                else:
                    # If not a valid UUID, generate a new one
                    entry_id = str(uuid.uuid4())
            except (ValueError, TypeError, AttributeError):
                # If any error occurs during validation, generate a new UUID
                entry_id = str(uuid.uuid4())
            
            # Extract related_note_ids if present
            related_note_ids = entry.get("related_note_ids", [])
            
            # Remove related_note_ids from entry as we'll store it separately
            if "related_note_ids" in entry:
                del entry["related_note_ids"]
            
            # Add required fields to the entry
            db_entry = {
                "id": entry_id,
                "space_id": space_id,
                "user_id": user_id,
                "content": entry.get("content", ""),
                "metadata": entry.get("metadata", {}),
                "related_data": entry.get("related_data", {}),
                "related_note_ids": related_note_ids,
                "file_name": entry.get("metadata", {}).get("label", "Untitled Research")
            }
            
            # Check if this entry already exists
            existing_response = client.table("space_research").select("id").eq("id", entry_id).execute()
            
            if existing_response.data:
                # Update existing entry
                client.table("space_research").update(db_entry).eq("id", entry_id).execute()
                logger.info(f"Updated existing research entry with ID {entry_id}")
            else:
                # Insert new entry
                client.table("space_research").insert(db_entry).execute()
                logger.info(f"Inserted new research entry with ID {entry_id}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error storing research entries in database: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def extract_yaml_from_text(text: str) -> str:
    """
    Extracts YAML from text that might contain extra content.
    
    Args:
        text: Text containing YAML
        
    Returns:
        str: Extracted YAML string
    """
    # Look for YAML block
    yaml_start = text.find("```yaml")
    if yaml_start != -1:
        yaml_start += 7  # Skip the ```yaml marker
        yaml_end = text.find("```", yaml_start)
        if yaml_end != -1:
            return text[yaml_start:yaml_end].strip()
    
    # If no YAML block markers but has a known YAML structure, try to extract it
    known_yaml_keys = ["graph:", "id:", "content:", "metadata:", "related_data:"]
    for key in known_yaml_keys:
        key_index = text.find(key)
        if key_index != -1:
            # Find the start of the line
            line_start = text.rfind("\n", 0, key_index) + 1
            if line_start == 0:  # Not found or at beginning
                line_start = 0
            return text[line_start:].strip()
    
    # If all else fails, just return the whole text
    logger.warning("Could not find YAML block in response, returning entire text")
    return text


def extract_json_from_text(text: str) -> str:
    """
    Extracts JSON from text that might contain extra content.
    
    Args:
        text: Text containing JSON
        
    Returns:
        str: Extracted JSON string
    """
    # Look for JSON block
    json_start = text.find("```json")
    if json_start != -1:
        json_start += 7  # Skip the ```json marker
        json_end = text.find("```", json_start)
        if json_end != -1:
            return text[json_start:json_end].strip()
    
    # If no JSON block markers, try to find JSON object
    json_start = text.find("{")
    if json_start != -1:
        # Find the matching closing brace by counting braces
        brace_count = 0
        for i in range(json_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    return text[json_start:json_end].strip()
    
    # If all else fails, just return the whole text
    logger.warning("Could not find JSON block in response, returning entire text")
    return text 