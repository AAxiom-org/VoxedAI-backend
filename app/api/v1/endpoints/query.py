"""
API endpoints for query operations.
"""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query as QueryParam
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from app.core.logging import logger
from app.schemas.file import QueryRequest, QueryResponse, QueryResult
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service
from app.db.pinecone import pinecone_client
from app.db.supabase import supabase_client

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query(
    request: QueryRequest,
) -> QueryResponse:
    """
    Queries the system using RAG.
    
    This endpoint:
    1. Retrieves relevant context from the vector database
    2. Generates an answer using the specified LLM model
    3. Returns the answer and sources
    
    Args:
        request: The query request.
        
    Returns:
        QueryResponse: The query response.
    """
    try:
        logger.info(f"DEBUG - Original query: {request.query}")
        # Handle streaming separately
        if request.is_coding_question:
            return await _stream_query_coding(request)
        else:
            if request.stream:
                return await _stream_query(request)
            
        # Default filter if none provided
        filter_dict = request.filter or {}
        
        # Handle user_id if provided to get toggled files
        if request.user_id:
            # Get pinecone_ids from user's toggled files
            pinecone_ids = await supabase_client.get_user_toggled_files(request.user_id)
            
            # First get all vector IDs that match the prefixes
            if pinecone_ids:
                vector_ids = []
                for pinecone_id in pinecone_ids:
                    ids = await pinecone_client.list_vectors(prefix=pinecone_id, namespace=request.namespace)
                    # Collect all IDs
                    vector_ids.extend(ids)
                
                # If we found matching vector IDs, create a proper filter
                if vector_ids:
                    # Use the helper method to create a properly formatted filter
                    filter_dict = await pinecone_client.create_id_filter(vector_ids)
        
        # Perform RAG query
        answer, context, query_time = await rag_service.query(
            query=request.query,
            top_k=request.top_k,
            model_name=request.model_name,
            use_rag=request.use_rag,
            stream=False,
            namespace=request.namespace,
            filter=filter_dict
        )
        
        # Process context: group by file_id and fetch complete metadata
        grouped_context = {}
        file_metadata_cache = {}
        
        # First group all chunks by file_id
        for doc in context:
            file_id = doc.get("file_id", "")
            if not file_id:
                continue
                
            if file_id not in grouped_context:
                grouped_context[file_id] = []
                
            grouped_context[file_id].append(doc)
        
        # Log the number of unique files found
        logger.info(f"Found {len(grouped_context)} unique files in RAG results")
        
        # Now create sources with complete file content and metadata
        sources = []
        for file_id, docs in grouped_context.items():
            # Get file metadata if we haven't cached it yet
            if file_id not in file_metadata_cache:
                file_metadata = await supabase_client.get_file_metadata(file_id)
                file_metadata_cache[file_id] = file_metadata
                logger.info(f"Retrieved metadata for file_id {file_id}: {file_metadata}")
            else:
                file_metadata = file_metadata_cache[file_id]
                
            # Sort chunks by their index to maintain order
            docs.sort(key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
            
            # Combine all chunks for this file
            full_text = "\n\n".join([doc.get("text", "") for doc in docs])
            
            # Extract filename from file_path
            file_path = docs[0].get("file_path", "") if docs else ""
            filename = file_path.split("/")[-1] if "/" in file_path else file_path
            
            logger.info(f"Processed file {filename} with {len(docs)} chunks and {len(full_text)} characters")
            
            # Create a single source entry with the complete context
            sources.append(
                QueryResult(
                    text=full_text,
                    score=max([doc.get("score", 0.0) for doc in docs]) if docs else 0.0,
                    file_id=file_id,
                    file_path=file_path,
                    source=filename,
                    metadata={
                        "description": file_metadata.get("description", ""),
                        "file_metadata": file_metadata.get("metadata", {}),
                        "chunk_count": len(docs)
                    }
                )
            )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time_ms=query_time
        )
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")


async def _stream_query(request: QueryRequest) -> StreamingResponse:
    """
    Streams a query response.
    
    Args:
        request: The query request.
        
    Returns:
        StreamingResponse: The streaming response.
    """
    async def generate():
        start_time = time.time()
        
        try:
            # Validate and sanitize the query
            query = request.query
            logger.info(f"DEBUG - Original query: '{query}'")
            
            # Check if query already contains streaming data (from a previous error perhaps)
            if query.strip().startswith('data:'):
                try:
                    # Try to extract a clean query from the streaming data
                    # Look for JSON structures and extract the original query if possible
                    import re
                    
                    # First, try to find a JSON object
                    json_matches = re.findall(r'data: ({.*?})\n\n', query)
                    if json_matches:
                        for match in json_matches:
                            try:
                                data = json.loads(match)
                                if 'query' in data:
                                    query = data['query']
                                    logger.info(f"DEBUG - Extracted query: '{query}'")
                                    break
                            except json.JSONDecodeError:
                                pass
                    
                    # If we still have streaming data, use a simple fallback
                    if query.strip().startswith('data:'):
                        # Fallback: just use a generic query
                        query = "Please provide information"
                        logger.warning(f"DEBUG - Couldn't extract query, using fallback: '{query}'")
                except Exception as extraction_error:
                    logger.error(f"Error extracting query: {extraction_error}")
                    query = "Please provide information"  # Fallback
            
            # Retrieve context
            raw_context = []
            if request.use_rag:
                # Default filter if none provided
                filter_dict = request.filter or {}
                
                # Handle user_id if provided to get toggled files
                if request.user_id:
                    # Get pinecone_ids from user's toggled files
                    output_dict = await supabase_client.get_user_toggled_files(request.user_id)
                    pinecone_ids = output_dict[True]
                    file_ids = output_dict[False]
                    
                    # First get all vector IDs that match the prefixes
                    if pinecone_ids:
                        vector_ids = []
                        for pinecone_id in pinecone_ids:
                            ids = await pinecone_client.list_vectors(prefix=pinecone_id, namespace=request.namespace)
                            # Collect all IDs
                            vector_ids.extend(ids)
                        
                        # If we found matching vector IDs, create a proper filter
                        if vector_ids:
                            # Use the helper method to create a properly formatted filter
                            filter_dict = await pinecone_client.create_id_filter(vector_ids)
                    if file_ids:
                        file_content = await supabase_client.get_file_content(file_ids)
                        logger.info(f"DEBUG - Appending Raw File Content: {file_content}")
                        raw_context.extend(file_content)
                
                # Use the sanitized query
                raw_context_rag = await rag_service.retrieve_context(
                    query=query,
                    top_k=request.top_k,
                    namespace=request.namespace,
                    filter=filter_dict
                )
                raw_context.extend(raw_context_rag)
            
            # Direct pass-through of search results to the LLM
            # Format each result with minimal processing
            context = []
            for doc in raw_context:
                # Get the document metadata
                metadata = doc.get("metadata", {})
                
                # Create context entry directly from the search result
                context_entry = {
                    "text": metadata.get("text_chunk", doc.get("text", "")),
                    "score": doc.get("score", 0.0),
                    "file_id": metadata.get("file_id", doc.get("file_id", "")),
                    "file_path": metadata.get("file_path", doc.get("file_path", "")),
                    "source": doc.get("source", metadata.get("source", "Unknown")),
                    "metadata": metadata
                }
                
                # Add to context
                context.append(context_entry)            
            # Log the final context count
            logger.info(f"Final context count: {len(context)} documents")
            
            # Format sources for the response (using processed context)
            sources = [
                {
                    "text": doc.get("text", ""),
                    "score": doc.get("score", 0.0),
                    "file_id": doc.get("file_id", ""),
                    "file_path": doc.get("file_path", ""),
                    "source": doc.get("source", ""),
                    "metadata": doc.get("metadata", {})
                }
                for doc in context
            ]
            
            # Send sources as the first chunk
            sources_json = json.dumps({"type": "sources", "data": sources})
            yield f"data: {sources_json}\n\n"

            
            logger.info(f"DEBUG - Streaming answer with context: {context}")
            
            # Stream the answer
            answer_stream = await rag_service.generate_answer(
                query=query,
                context=context,
                model_name=request.model_name,
                stream=True
            )
            
            # Since answer_stream is an AsyncGenerator, we can directly iterate over it
            try:
                async for chunk in answer_stream:
                    logger.info(f"DEBUG - Received chunk: {chunk}")
                    chunk_json = json.dumps({"type": "token", "data": chunk})
                    yield f"data: {chunk_json}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
            except Exception as e:
                logger.error(f"Error streaming chunks: {e}")
                error_json = json.dumps({"type": "error", "error": f"Streaming error: {str(e)}"})
                yield f"data: {error_json}\n\n"
            
            # Send query time as the final chunk
            query_time = (time.time() - start_time) * 1000
            done_json = json.dumps({"type": "done", "query_time_ms": query_time})
            yield f"data: {done_json}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            error_json = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {error_json}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    ) 

async def _stream_query_coding(request: QueryRequest) -> StreamingResponse:
    """
    Streams a response for coding questions, bypassing RAG and going directly to the LLM.
    
    Args:
        request: The query request.
        
    Returns:
        StreamingResponse: The streaming response.
    """
    async def generate():
        start_time = time.time()
        
        try:
            # Get the query
            query = request.query
            logger.info(f"DEBUG - Original coding query: '{query}'")
            
            # Check if query already contains streaming data (from a previous error perhaps)
            if query.strip().startswith('data:'):
                try:
                    # Try to extract a clean query from the streaming data
                    import re
                    
                    # First, try to find a JSON object
                    json_matches = re.findall(r'data: ({.*?})\n\n', query)
                    if json_matches:
                        for match in json_matches:
                            try:
                                data = json.loads(match)
                                if 'query' in data:
                                    query = data['query']
                                    logger.info(f"DEBUG - Extracted coding query: '{query}'")
                                    break
                            except json.JSONDecodeError:
                                pass
                    
                    # If we still have streaming data, use a simple fallback
                    if query.strip().startswith('data:'):
                        query = "Please provide coding assistance"
                        logger.warning(f"DEBUG - Couldn't extract coding query, using fallback: '{query}'")
                except Exception as extraction_error:
                    logger.error(f"Error extracting coding query: {extraction_error}")
                    query = "Please provide coding assistance"  # Fallback
            
            # For coding questions, we don't use RAG, so we send empty sources
            sources = []
            sources_json = json.dumps({"type": "sources", "data": sources})
            yield f"data: {sources_json}\n\n"
            
            logger.info(f"DEBUG - Streaming coding answer directly to LLM")
            
            # Get the answer stream from the LLM service
            # The generate_answer_with_coding_question method returns a stream
            answer_stream = await llm_service.generate_answer_with_coding_question(
                query=query,
                model_name=request.model_name
            )
            
            # Process the stream
            if hasattr(answer_stream, '__aiter__'):
                # It's an async generator
                try:
                    async for chunk in answer_stream:
                        logger.info(f"DEBUG - Received coding chunk: {chunk}")
                        chunk_json = json.dumps({"type": "token", "data": chunk})
                        yield f"data: {chunk_json}\n\n"
                        await asyncio.sleep(0)  # Allow other tasks to run
                except Exception as e:
                    logger.error(f"Error streaming coding chunks: {e}")
                    error_json = json.dumps({"type": "error", "error": f"Streaming error: {str(e)}"})
                    yield f"data: {error_json}\n\n"
            else:
                # It's a string (non-streaming response)
                logger.info(f"DEBUG - Received non-streaming coding response")
                # Send the entire response as a single token
                chunk_json = json.dumps({"type": "token", "data": answer_stream})
                yield f"data: {chunk_json}\n\n"
            
            # Send query time as the final chunk
            query_time = (time.time() - start_time) * 1000
            done_json = json.dumps({"type": "done", "query_time_ms": query_time})
            yield f"data: {done_json}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming coding query: {e}")
            error_json = json.dumps({"type": "error", "error": str(e)})
            yield f"data: {error_json}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    ) 