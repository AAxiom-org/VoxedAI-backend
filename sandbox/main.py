"""
Secure code execution sandbox API.

This service runs in a Docker container and provides an API for
executing Python code in an isolated environment.
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import json
import sys
import io
import traceback
import asyncio
import time
from typing import Dict, Any, Optional
from contextlib import redirect_stdout, redirect_stderr
import importlib
from pydantic import BaseModel, Field

# Instantiate FastAPI application
app = FastAPI(
    title="Code Execution Sandbox",
    description="API for securely executing Python code",
    version="1.0.0"
)

# Dictionary to store uploaded files and data
data_store = {}


class CodeExecutionRequest(BaseModel):
    """Schema for code execution request."""
    code: str = Field(..., description="The code to execute")
    input_data: Dict[str, Any] = Field(
        default={},
        description="Optional input data to be made available to the code"
    )


@app.get("/health")
async def health_check():
    """Check if the sandbox service is healthy."""
    try:
        # Test basic code execution
        test_code = "result = 1 + 1"
        test_result = await execute_code(CodeExecutionRequest(code=test_code))
        
        if test_result.get("result", {}).get("result") == 2:
            return JSONResponse(
                status_code=200,
                content={"status": "healthy", "message": "Sandbox is operational"}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"status": "unhealthy", "message": "Code execution test failed"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "message": f"Health check failed: {str(e)}"}
        )


@app.post("/execute_code")
async def execute_code(request: CodeExecutionRequest):
    """
    Execute Python code in a safe environment.
    
    This endpoint:
    1. Takes Python code and optional input data
    2. Executes the code in a safe environment
    3. Returns the results, stdout, and stderr
    
    Returns:
        Dict containing execution results, stdout, and stderr.
    """
    # Set up stdout and stderr capture
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    # Create a global scope with builtins
    global_scope = {"__builtins__": __builtins__}
    
    # Initialize local variables dict with input data
    local_vars = {}
    
    # Add pandas and numpy to local scope
    local_vars['pd'] = pd
    local_vars['np'] = np
    
    # Add the input data to the local scope
    local_vars.update(request.input_data)
    
    # Add __name__ = "__main__" to ensure main block execution
    global_scope["__name__"] = "__main__"
    
    # Record execution time
    start_time = time.time()
    
    try:
        # Execute code with captured stdout/stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(request.code, global_scope, local_vars)
        
        execution_time = time.time() - start_time
        
        # Combine global and local scopes for result extraction
        # (excluding built-ins and modules)
        result_dict = {}
        for scope in [global_scope, local_vars]:
            for k, v in scope.items():
                # Skip modules, built-in objects, and private attributes
                if k.startswith('__') or isinstance(v, type(pd)):
                    continue
                    
                if isinstance(v, pd.DataFrame):
                    # Convert DataFrames to JSON
                    result_dict[k] = json.loads(v.to_json(orient="records"))
                elif isinstance(v, np.ndarray):
                    # Convert NumPy arrays to lists
                    result_dict[k] = v.tolist()
                elif hasattr(v, '__dict__'):
                    # Convert custom objects to dict
                    result_dict[k] = vars(v)
                else:
                    # Try to include the value directly if it's JSON serializable
                    try:
                        json.dumps({k: v})
                        result_dict[k] = v
                    except (TypeError, OverflowError):
                        # If not serializable, convert to string
                        result_dict[k] = str(v)
        
        return {
            "result": result_dict,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "execution_time": execution_time
        }
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = str(e)
        traceback_str = traceback.format_exc()
        
        # Log detailed error information
        print(f"Code execution error: {error_message}", file=sys.stderr)
        print(f"Traceback: {traceback_str}", file=sys.stderr)
        
        return {
            "result": None,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue() + f"\n{traceback_str}",
            "error": error_message,
            "execution_time": execution_time
        }


@app.post("/upload_data/{key}")
async def upload_data(key: str, data: Dict[str, Any]):
    """Upload data to be used in code execution."""
    data_store[key] = data
    return {"message": f"Data for key '{key}' uploaded successfully"}


@app.get("/get_data/{key}")
async def get_data(key: str):
    """Retrieve data by key."""
    if key not in data_store:
        raise HTTPException(status_code=404, detail=f"No data found for key '{key}'")
    return data_store[key]


@app.get("/clear_data")
async def clear_data():
    """Clear all stored data."""
    data_store.clear()
    return {"message": "All data cleared"} 