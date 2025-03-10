"""
Pinecone client module for vector database operations.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import pinecone
from pinecone import Index, Pinecone

from app.core.config import settings
from app.core.logging import logger


class PineconeClient:
    """
    Singleton client for Pinecone vector database.
    """

    _instance: Optional["PineconeClient"] = None
    _client: Optional[Pinecone] = None
    _index: Optional[Index] = None

    def __new__(cls) -> "PineconeClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._client is None:
            try:
                # Initialize Pinecone client
                self._client = Pinecone(api_key=settings.PINECONE_API_KEY)
                
                # Connect to the index
                self._index = self._client.Index(
                    host=settings.PINECONE_HOST_LLAMA,
                    name=settings.PINECONE_INDEX_LLAMA,
                )
                
                logger.info(f"Pinecone client initialized successfully with index {settings.PINECONE_INDEX_LLAMA}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone client: {e}")
                raise

    @property
    def client(self) -> Pinecone:
        """
        Returns the Pinecone client.
        
        Returns:
            Pinecone: The Pinecone client.
        """
        if self._client is None:
            raise ValueError("Pinecone client not initialized")
        return self._client

    @property
    def index(self) -> Index:
        """
        Returns the Pinecone index.
        
        Returns:
            Index: The Pinecone index.
        """
        if self._index is None:
            raise ValueError("Pinecone index not initialized")
        return self._index

    async def upsert_vectors(
        self, vectors: List[Tuple[str, List[float], Dict[str, Any]]], namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upserts vectors into the Pinecone index.
        
        Args:
            vectors: List of tuples containing (id, vector, metadata)
            namespace: The namespace to upsert into
            
        Returns:
            Dict[str, Any]: The upsert response.
        """
        try:
            # Convert the list of tuples to the format expected by Pinecone
            vectors_to_upsert = []
            skipped_vectors = 0
            
            # Pinecone metadata limit is 40KB (40960 bytes)
            MAX_METADATA_BYTES = 40960
            
            for id, vector, metadata in vectors:
                # Validate and sanitize metadata for Pinecone compatibility
                # Pinecone only accepts strings, numbers, booleans, or arrays of strings as values
                sanitized_metadata = {}
                for key, value in metadata.items():
                    # Skip null values
                    if value is None:
                        continue
                        
                    # Handle different value types
                    if isinstance(value, (str, int, float, bool)):
                        # These primitive types are directly supported
                        sanitized_metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        # Lists of strings are supported
                        sanitized_metadata[key] = value
                    elif isinstance(value, list):
                        # Convert non-string lists to string
                        sanitized_metadata[key] = json.dumps(value)
                    elif isinstance(value, dict):
                        # Convert dictionaries to string
                        sanitized_metadata[key] = json.dumps(value)
                    else:
                        # For any other types, convert to string
                        sanitized_metadata[key] = str(value)
                
                # Check metadata size after sanitization
                metadata_json = json.dumps(sanitized_metadata)
                metadata_size = len(metadata_json.encode('utf-8'))
                
                if metadata_size > MAX_METADATA_BYTES:
                    logger.warning(f"Skipping vector {id} due to metadata size {metadata_size} bytes exceeding limit of {MAX_METADATA_BYTES} bytes")
                    skipped_vectors += 1
                    continue
                
                vectors_to_upsert.append({"id": id, "values": vector, "metadata": sanitized_metadata})
            
            if not vectors_to_upsert:
                logger.warning("No vectors to upsert after size filtering")
                return {"upserted_count": 0, "skipped_count": skipped_vectors, "results": []}
            
            # Batch upsert in chunks of 100 (Pinecone's recommendation)
            chunk_size = 100
            results = []
            
            for i in range(0, len(vectors_to_upsert), chunk_size):
                chunk = vectors_to_upsert[i:i + chunk_size]
                response = self.index.upsert(vectors=chunk, namespace=namespace)
                results.append(response)
            
            logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone (skipped {skipped_vectors} due to size limits)")
            return {"upserted_count": len(vectors_to_upsert), "skipped_count": skipped_vectors, "results": results}
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            raise

    async def delete_vectors(self, ids: List[str], namespace: str = "") -> Dict[str, Any]:
        """
        Deletes vectors from the Pinecone index.
        
        Args:
            ids: List of vector IDs to delete
            namespace: The namespace to delete from
            
        Returns:
            Dict[str, Any]: The deletion response.
        """
        try:
            response = self.index.delete(ids=ids, namespace=namespace)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return response
        except Exception as e:
            logger.error(f"Error deleting vectors from Pinecone: {e}")
            raise

    async def create_id_filter(self, ids: List[str]) -> Dict[str, Any]:
        """
        Creates a proper ID filter for Pinecone query.
        
        Args:
            ids: List of vector IDs to filter by
            
        Returns:
            Dict[str, Any]: The filter dictionary properly formatted for Pinecone.
        """
        # Ensure ids is a simple list of strings, not nested
        flat_ids = []
        for id_val in ids:
            if isinstance(id_val, list):
                # Recursively flatten nested lists
                for nested_id in id_val:
                    if isinstance(nested_id, list):
                        flat_ids.extend(nested_id)
                    else:
                        flat_ids.append(nested_id)
            else:
                flat_ids.append(id_val)
        
        # Additional validation to ensure all IDs are strings
        validated_ids = [str(id_val) for id_val in flat_ids if id_val]
        
        # Create the filter with the proper structure
        filter_dict = {"id": {"$in": validated_ids}}
        logger.info(f"Created ID filter with {len(validated_ids)} IDs: {filter_dict}")
        return filter_dict

    async def query_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries vectors from the Pinecone index.
        
        Args:
            query_vector: The query vector
            top_k: Number of results to return
            namespace: The namespace to query in
            filter: Optional filter to apply
            
        Returns:
            List[Dict[str, Any]]: The query results.
        """
        try:
            # Debug output of the filter
            if filter:
                logger.info(f"Querying with filter: {filter}")
            
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True
            )
            logger.info(f"DEBUG - Pinecone query response: {response}")
            return response.get("matches", [])
        except Exception as e:
            logger.error(f"Error querying vectors from Pinecone: {e}")
            raise

    async def list_vectors(self, prefix: str, namespace: str = "") -> List[str]:
        """
        Lists vector IDs with the given prefix from the Pinecone index.
        
        Args:
            prefix: The prefix to filter vector IDs by
            namespace: The namespace to list from
            
        Returns:
            List[str]: List of vector IDs.
        """
        try:
            # The list method returns a generator, not a dictionary
            vector_ids = []
            for vector_id in self.index.list(prefix=prefix, namespace=namespace):
                vector_ids.append(vector_id)
            
            logger.info(f"Listed {len(vector_ids)} vectors with prefix {prefix}")
            return vector_ids
        except Exception as e:
            logger.error(f"Error listing vectors from Pinecone: {e}")
            raise

    async def search_records(
        self, 
        query_text: str, 
        top_k: int = 5, 
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs a text search using Pinecone's search_records API.
        
        Args:
            query_text: The text query to search for.
            top_k: Number of results to return.
            namespace: The namespace to search in.
            filter: Optional filter to apply
            
        Returns:
            List[Dict[str, Any]]: The search results.
        """
        try:
            # Debug output of the filter
            if filter:
                logger.info(f"Searching records with filter: {filter}")
            
            logger.info(f"Performing text search with query: '{query_text}'")
            
            # Construct the search query according to Pinecone's search_records format
            if filter:
                search_query = {
                    "inputs": {"text": query_text},
                    "top_k": top_k,
                    "filter": filter,
                }
            else:
                search_query = {
                    "inputs": {"text": query_text},
                    "top_k": top_k,
                }
            
            if not namespace:
                namespace = ""
            # Perform search without applying filter at the Pinecone level
            response = self.index.search_records(
                namespace=namespace,
                query=search_query
            )
            
            logger.info(f"DEBUG - Pinecone search_records response: {response}")
            
            # Transform the response to match the expected format
            transformed_results = []
            if response and "result" in response and "hits" in response["result"]:
                for hit in response["result"]["hits"]:
                    # Extract the metadata and other relevant fields
                    fields = hit.get("fields", {})
                    
                    # Process the text chunk
                    text_chunk = fields.get("text_chunk", "")
                    
                    # Process additional info if it exists
                    additional_info = fields.get("additional_info", {})
                    
                    # Create the result dictionary
                    result = {
                        "id": hit.get("_id", ""),
                        "score": hit.get("_score", 0.0),
                        "text": text_chunk,
                        "file_id": fields.get("file_id", ""),
                        "file_path": fields.get("file_path", ""),
                        "source": fields.get("source", "Unknown"),
                        "metadata": {
                            "chunk_index": fields.get("chunk_index", 0),
                            "description": fields.get("description", ""),
                            "additional_info": additional_info,
                            "entities": fields.get("entities", []),
                            "key_points": fields.get("key_points", []),
                            "topics": fields.get("topics", []),
                            "total_chunks": fields.get("total_chunks", 1)
                        }
                    }
                    transformed_results.append(result)
            
            logger.info(f"Search returned {len(transformed_results)} results")
            return transformed_results
        except Exception as e:
            logger.error(f"Error searching records in Pinecone: {e}")
            raise


# Global instance of the Pinecone client
pinecone_client = PineconeClient() 