"""
Service for executing code in a sandbox environment.
"""
import asyncio
import json
import os
import tempfile
import time
from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import logger


class CodeExecutionService:
    """Service for executing code in a secure Docker sandbox."""
    
    def __init__(self):
        """Initialize the code execution service."""
        # URL of the sandbox service
        self.sandbox_url = settings.CODE_SANDBOX_URL
        # Default timeout for code execution
        self.default_timeout = 30
        # Initialize httpx client for communicating with the sandbox
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def _health_check(self) -> bool:
        """Check if the sandbox service is healthy."""
        try:
            response = await self.client.get(f"{self.sandbox_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Sandbox health check failed: {str(e)}")
            logger.error(f"Sandbox URL: {self.sandbox_url}")
            return False
    
    async def execute_code(
        self,
        code: str,
        input_data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the provided code in the sandbox and return the results.
        
        Args:
            code: The code to execute.
            input_data: Optional data to make available to the code.
            timeout: Maximum execution time in seconds.
            
        Returns:
            Dict containing execution results.
        """
        # Set timeout to default if not provided
        timeout = timeout or self.default_timeout
        
        # Check if sandbox is running
        if not await self._health_check():
            logger.error(f"Sandbox health check failed before code execution")
            raise HTTPException(
                status_code=503,
                detail="Code execution sandbox is not available"
            )
        
        # Prepare the request payload
        payload = {
            "code": code,
            "input_data": input_data or {},
        }
        
        try:
            start_time = time.time()
            # Execute the code in the sandbox
            logger.info(f"Sending code to sandbox at {self.sandbox_url}/execute_code")
            response = await self.client.post(
                f"{self.sandbox_url}/execute_code",
                json=payload,
                timeout=float(timeout) + 5.0  # Add buffer to timeout
            )
            execution_time = time.time() - start_time
            
            if response.status_code != 200:
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                except:
                    error_detail = f"HTTP {response.status_code}: {response.text}"
                
                logger.error(f"Sandbox execution failed: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Code execution failed: {error_detail}"
                )
            
            # Parse and return the results
            result = response.json()
            result["execution_time"] = execution_time
            
            # Log if there was any stderr output
            if result.get("stderr"):
                logger.warning(f"Code execution produced stderr: {result.get('stderr')}")
                
            return result
            
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=408,
                detail=f"Code execution timed out after {timeout} seconds"
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with code sandbox: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Code execution failed: {str(e)}"
            )

# Create a singleton instance
code_execution_service = CodeExecutionService() 