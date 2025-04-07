"""
API endpoints for research operations.
"""
import json
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.db.supabase import supabase_client
from app.agents.research import (
    DigestNode,
    generate_search_queries,
    perform_web_search,
    generate_digest
)

router = APIRouter()


class DigestGenerationRequest(BaseModel):
    """Schema for digest generation request."""
    space_id: str = Field(..., description="ID of the space to generate digest for")
    user_id: str = Field(..., description="ID of the user")
    stream: bool = Field(True, description="Whether to stream the response")


class DigestGenerationResponse(BaseModel):
    """Schema for digest generation response."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Response message")
    digest_ids: List[str] = Field(default_factory=list, description="IDs of the generated digests")


@router.post("/digest", response_model=DigestGenerationResponse)
async def generate_research_digest(
    request: DigestGenerationRequest,
    background_tasks: BackgroundTasks
) -> Any:
    """
    Generates multiple focused research digests based on user's notes and web search.
    
    This endpoint:
    1. Retrieves all notes from the user's space
    2. Identifies 2-4 distinct topics in the notes
    3. Generates search queries for each topic
    4. Performs web searches for each query
    5. Generates a focused research digest for each topic
    6. Stores each digest in the database
    
    Args:
        request: Contains space_id, user_id, and stream parameters
        background_tasks: For handling cleanup tasks
        
    Returns:
        DigestGenerationResponse or StreamingResponse: Contains the operation status
    """
    try:
        start_time = time.time()
        
        # If streaming is requested, set up a streaming response with SSE
        if request.stream:
            async def event_generator():
                try:
                    # Send initial status
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Starting digest generation...'})}\n\n"
                    
                    # Step 1: Fetch all notes from the space
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Fetching notes from space...'})}\n\n"
                    
                    notes = await fetch_space_notes(request.space_id, request.user_id)
                    if not notes:
                        yield f"data: {json.dumps({'type': 'status', 'message': 'No notes found, using default content'})}\n\n"
                        notes_content = "No notes available in this space yet."
                        note_ids = []
                    else:
                        note_count = len(notes)
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Found {note_count} notes in space'})}\n\n"
                        
                        # Extract note IDs
                        note_ids = [note["id"] for note in notes]
                        # Combine note content
                        notes_content = combine_notes_content(notes)
                    
                    # Step 2: Generate topic groups and search queries
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Identifying topics and generating search queries...'})}\n\n"
                    
                    topic_groups = await generate_search_queries(notes_content, request.space_id, note_ids)
                    
                    group_count = len(topic_groups)
                    yield f"data: {json.dumps({'type': 'status', 'message': f'Identified {group_count} topics for research'})}\n\n"
                    
                    # Track all digest IDs
                    all_digest_ids = []
                    
                    # Process each topic group
                    for group_idx, topic_group in enumerate(topic_groups):
                        topic = topic_group.get("topic", f"Topic {group_idx+1}")
                        description = topic_group.get("description", "")
                        queries = topic_group.get("queries", [])
                        related_note_ids = topic_group.get("related_note_ids", note_ids)
                        
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Processing topic {group_idx+1} of {group_count}: {topic}'})}\n\n"
                        
                        # Step 3: Perform web searches for this topic's queries
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Performing web searches for topic: {topic}...'})}\n\n"
                        
                        search_results = []
                        query_count = len(queries)
                        
                        for i, query in enumerate(queries):
                            yield f"data: {json.dumps({'type': 'status', 'message': f'Searching query {i+1} of {query_count} for topic {topic}: {query}'})}\n\n"
                            
                            result = await perform_web_search(query)
                            search_results.append((query, result))
                            
                            yield f"data: {json.dumps({'type': 'status', 'message': f'Completed search {i+1} of {query_count} for topic {topic}'})}\n\n"
                        
                        # Step 4: Generate digest for this topic
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Generating research digest for topic: {topic}...'})}\n\n"
                        
                        digest = await generate_digest(
                            notes_content, 
                            search_results, 
                            topic, 
                            description, 
                            related_note_ids
                        )
                        
                        # Step 5: Store digest in database
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Storing digest for topic: {topic} in database...'})}\n\n"
                        
                        # Add space and user IDs before storing
                        digest.space_id = request.space_id
                        digest.user_id = request.user_id
                        
                        digest_id = await store_digest_in_database(digest)
                        
                        if digest_id:
                            all_digest_ids.append(digest_id)
                            yield f"data: {json.dumps({'type': 'digest_created', 'message': f'Created digest for topic: {topic}', 'digest_id': digest_id})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to store digest for topic: {topic} in database'})}\n\n"
                    
                    # Calculate query time
                    query_time_ms = int((time.time() - start_time) * 1000)
                    
                    # Send a final "done" event
                    yield f"data: {json.dumps({'type': 'done', 'message': f'Created {len(all_digest_ids)} digests', 'query_time_ms': query_time_ms, 'digest_ids': all_digest_ids})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in digest generation: {str(e)}")
                    error_data = {
                        "type": "error",
                        "message": f"Error generating digest: {str(e)}"
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
                notes_content = "No notes available in this space yet."
                note_ids = []
            else:
                # Extract note IDs
                note_ids = [note["id"] for note in notes]
                # Combine note content
                notes_content = combine_notes_content(notes)
            
            # Step 2: Generate topic groups and search queries
            topic_groups = await generate_search_queries(notes_content, request.space_id, note_ids)
            
            # Process all topics and store their digest IDs
            all_digest_ids = []
            
            for topic_group in topic_groups:
                topic = topic_group.get("topic", "Untitled Topic")
                description = topic_group.get("description", "")
                queries = topic_group.get("queries", [])
                related_note_ids = topic_group.get("related_note_ids", note_ids)
                
                # Step 3: Perform web searches for this topic
                search_results = []
                for query in queries:
                    result = await perform_web_search(query)
                    search_results.append((query, result))
                
                # Step 4: Generate digest for this topic
                digest = await generate_digest(
                    notes_content, 
                    search_results, 
                    topic, 
                    description, 
                    related_note_ids
                )
                
                # Add space and user IDs before storing
                digest.space_id = request.space_id
                digest.user_id = request.user_id
                
                # Step 5: Store digest in database
                digest_id = await store_digest_in_database(digest)
                
                if digest_id:
                    all_digest_ids.append(digest_id)
            
            # Return success response
            return DigestGenerationResponse(
                success=True,
                message=f"Created {len(all_digest_ids)} digests",
                digest_ids=all_digest_ids
            )
        
        except Exception as e:
            logger.error(f"Error in non-streaming digest generation: {str(e)}")
            return DigestGenerationResponse(
                success=False,
                message=f"Error generating digest: {str(e)}",
                digest_ids=[]
            )
    
    except Exception as e:
        logger.error(f"Error in digest generation endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating digest: {str(e)}"
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
                    "content": note_content,
                    "metadata": note.get("metadata", {})
                })
        
        logger.info(f"Found {len(notes)} notes in space {space_id}")
        return notes
    
    except Exception as e:
        logger.error(f"Error fetching space notes: {e}")
        return []


def combine_notes_content(notes: List[Dict[str, Any]]) -> str:
    """
    Combines content from multiple notes into a single string.
    
    Args:
        notes: List of notes
        
    Returns:
        str: Combined note content
    """
    combined_content = ""
    
    for note in notes:
        note_title = note.get("file_name", "Untitled Note").replace(".json", "")
        note_content = note.get("content", "")
        
        # Try to process JSON content if it appears to be JSON
        if note_content.strip().startswith("{") and "content" in note_content:
            try:
                content_obj = json.loads(note_content)
                # Extract text content from BlockNote format
                extracted_text = extract_text_from_blocknote(content_obj)
                note_content = extracted_text
            except Exception as e:
                logger.warning(f"Error parsing note JSON: {e}")
        
        combined_content += f"\n\n## NOTE: {note_title}\n\n{note_content}"
    
    return combined_content


def extract_text_from_blocknote(content_obj: Dict[str, Any]) -> str:
    """
    Extracts plain text from BlockNote formatted content.
    
    Args:
        content_obj: BlockNote content object
        
    Returns:
        str: Extracted text
    """
    try:
        extracted_text = ""
        
        if not isinstance(content_obj, dict):
            return str(content_obj)
        
        if "type" in content_obj and content_obj["type"] == "doc" and "content" in content_obj:
            blocks = content_obj["content"]
            
            for block in blocks:
                if "type" in block:
                    block_type = block["type"]
                    
                    # Handle heading blocks
                    if block_type == "heading" and "content" in block:
                        level = block.get("attrs", {}).get("level", 1)
                        heading_text = ""
                        
                        for text_node in block["content"]:
                            if "text" in text_node:
                                heading_text += text_node["text"]
                        
                        extracted_text += f"{'#' * level} {heading_text}\n\n"
                    
                    # Handle paragraph blocks
                    elif block_type == "paragraph" and "content" in block:
                        para_text = ""
                        
                        for text_node in block["content"]:
                            if "text" in text_node:
                                para_text += text_node["text"]
                        
                        extracted_text += f"{para_text}\n\n"
                    
                    # Handle list blocks
                    elif block_type in ["bulletList", "orderedList"] and "content" in block:
                        for list_item in block["content"]:
                            if "content" in list_item:
                                for item_content in list_item["content"]:
                                    if "content" in item_content:
                                        item_text = ""
                                        for text_node in item_content["content"]:
                                            if "text" in text_node:
                                                item_text += text_node["text"]
                                        
                                        prefix = "- " if block_type == "bulletList" else "1. "
                                        extracted_text += f"{prefix}{item_text}\n"
                        
                        extracted_text += "\n"
        
        return extracted_text
    
    except Exception as e:
        logger.error(f"Error extracting text from BlockNote: {e}")
        return str(content_obj)


async def store_digest_in_database(digest: DigestNode) -> Optional[str]:
    """
    Stores the generated digest in the database.
    
    Args:
        digest: The digest to store
        
    Returns:
        Optional[str]: The ID of the stored digest, or None if storage failed
    """
    try:
        client = supabase_client.client
        
        # Insert new digest
        insert_data = {
            "id": digest.id,
            "space_id": digest.space_id,
            "user_id": digest.user_id,
            "title": digest.title,
            "content": digest.content,
            "all_links": digest.all_links,
            "related_note_ids": digest.related_note_ids,
        }
        
        response = client.table("space_digests").insert(insert_data).execute()
        
        if response.data:
            digest_id = response.data[0]["id"]
            logger.info(f"Inserted new digest with ID {digest_id} and {len(digest.related_note_ids)} related notes")
            return digest_id
        else:
            logger.error("Failed to insert digest: no data returned")
            return None
    
    except Exception as e:
        logger.error(f"Error storing digest in database: {e}")
        return None 