"""
API endpoints for code execution operations.
"""
import json
from typing import Any, Dict
import traceback

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.logging import logger
from app.schemas.code import CodeExecutionRequest, CodeExecutionResponse
from app.services.code_execution_service import code_execution_service

router = APIRouter()


@router.post("/run", response_model=CodeExecutionResponse)
async def run_code(
    request: CodeExecutionRequest,
) -> CodeExecutionResponse:
    """
    Executes code snippets in a secure sandbox environment.
    
    This endpoint:
    1. Takes a code snippet and optional input data
    2. Executes it in an isolated Docker container sandbox
    3. Returns the execution results
    
    Args:
        request: The code execution request containing the code and optional data.
        
    Returns:
        CodeExecutionResponse: The execution results.
    """
    try:
        logger.info(f"Executing code snippet, length: {len(request.code)}")
        logger.info(f"Executing code snippet: {request.code}")
        
        # Execute the code in the sandbox
        result = await code_execution_service.execute_code(
            code=request.code,
            input_data=request.input_data,
            timeout=request.timeout
        )

        logger.info(f"Execution result: {result}")
        logger.info(f"Execution stdout: {result.get('stdout')}")
        logger.info(f"Execution result data: {result.get('result')}")
        
        # Check for errors in stderr
        stderr = result.get("stderr", "")
        if stderr and "Error" in stderr:
            logger.warning(f"Code execution produced errors: {stderr}")
        
        return CodeExecutionResponse(
            success=True,
            result=result.get("result"),
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
            execution_time=result.get("execution_time")
        )
        
    except HTTPException as e:
        logger.error(f"HTTP error executing code: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing code: {str(e)}"
        ) 