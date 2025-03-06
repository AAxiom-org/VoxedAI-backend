"""
RAG service module for retrieval augmented generation.
"""
import time
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

from app.core.logging import logger
from app.services.llm_service import llm_service
from app.db.pinecone import pinecone_client


class RAGService:
    """
    Service for retrieval augmented generation.
    """

    @staticmethod
    async def retrieve_context(
        query: str, 
        top_k: int = 5, 
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        optimize_query: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieves context documents for a query using direct text-based search.
        
        Args:
            query: The user query.
            top_k: Number of results to return.
            namespace: The namespace to search in.
            filter: Metadata filters to apply.
            optimize_query: Whether to optimize the query before searching.
            
        Returns:
            List[Dict[str, Any]]: The retrieved context documents.
        """
        try:
            # Check if query is already a formatted stream
            if query.strip().startswith('data:'):
                optimize_query = False
                
            # Optimize the query if requested
            search_query = query
            if optimize_query:
                try:
                    logger.info(f"DEBUG - About to optimize query: '{query}'")
                    optimized = await llm_service.optimize_query(query)
                    logger.info(f"DEBUG - Raw optimized query: '{optimized}'")
                    
                    # Only use the optimized query if it's a valid string and not empty
                    if isinstance(optimized, str) and optimized.strip():
                        search_query = optimized
                        logger.info(f"Optimized query: '{query}' -> '{search_query}'")
                    else:
                        logger.warning(f"Optimization returned invalid result, using original query")
                except Exception as e:
                    logger.error(f"Error during query optimization: {e}")
                    # Continue with the original query if optimization fails
            
            # Search directly using text-based search (no embedding generation needed)
            logger.info(f"DEBUG - Searching with query: '{search_query}'")
            results = await pinecone_client.search_records(
                query_text=search_query,
                top_k=top_k,
                namespace=namespace,
                filter=filter
            )
            
            logger.info(f"Retrieved {len(results)} context documents for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise

    @staticmethod
    async def generate_answer(
        query: str, 
        context: List[Dict[str, Any]], 
        model_name: str = "gemini",
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generates an answer to a query using the retrieved context.
        
        Args:
            query: The user query.
            context: The retrieved context documents.
            model_name: The name of the LLM model to use.
            stream: Whether to stream the response.
            
        Returns:
            Union[str, AsyncGenerator[str, None]]: The generated answer or a stream of tokens.
        """
        try:
            # Always await the call to llm_service.generate_answer
            # For streaming mode, this should return an async generator
            # For non-streaming mode, this should return a string
            logger.info(f"DEBUG - RAG generate_answer called with stream={stream}, model={model_name}")
            
            result = await llm_service.generate_answer(
                query=query,
                context=context,
                model_name=model_name,
                stream=stream
            )
            
            logger.info(f"DEBUG - RAG generate_answer result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


    @staticmethod
    async def query(
        query: str, 
        top_k: int = 5, 
        model_name: str = "gemini",
        use_rag: bool = True,
        stream: bool = False,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[Union[str, Any], List[Dict[str, Any]], float]:
        """
        Performs a complete RAG query.
        
        Args:
            query: The user query.
            top_k: Number of results to return.
            model_name: The name of the LLM model to use.
            use_rag: Whether to use RAG or just the LLM.
            stream: Whether to stream the response.
            namespace: The namespace to search in.
            filter: Metadata filters to apply.
            
        Returns:
            Tuple[Union[str, Any], List[Dict[str, Any]], float]: The answer, context documents, and query time.
        """
        start_time = time.time()
        
        try:
            context = []
            
            # Retrieve context if using RAG
            if use_rag:
                context = await RAGService.retrieve_context(
                    query=query,
                    top_k=top_k,
                    namespace=namespace,
                    filter=filter
                )
            
            # Generate answer
            answer = await RAGService.generate_answer(
                query=query,
                context=context,
                model_name=model_name,
                stream=stream
            )
            
            query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.info(f"RAG query completed in {query_time:.2f}ms")
            
            return answer, context, query_time
        except Exception as e:
            logger.error(f"Error performing RAG query: {e}")
            query_time = (time.time() - start_time) * 1000
            raise


# Global instance of the RAG service
rag_service = RAGService() 