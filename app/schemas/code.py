"""
Schema definitions for code execution operations.
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class CodeExecutionRequest(BaseModel):
    """Schema for code execution request."""
    
    code: str = Field(..., 
                      description="The code snippet to execute",
                      example="print('Hello, world!')")
                      
    input_data: Optional[Dict[str, Any]] = Field(
        default={},
        description="Optional input data to be made available to the code",
        example={"df": {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}}
    )
    
    timeout: int = Field(
        default=30,
        description="Maximum execution time in seconds",
        ge=1,
        le=300
    )
    
    @validator('code')
    def validate_code_length(cls, v):
        """Validate that code is not empty and not too long."""
        if not v.strip():
            raise ValueError("Code cannot be empty")
        if len(v) > 50000:
            raise ValueError("Code exceeds maximum length of 50000 characters")
        return v


class CodeExecutionResponse(BaseModel):
    """Schema for code execution response."""
    
    success: bool = Field(..., 
                         description="Whether the code execution was successful")
                         
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The result of the code execution"
    )
    
    stdout: Optional[str] = Field(
        default=None,
        description="Standard output from the code execution"
    )
    
    stderr: Optional[str] = Field(
        default=None,
        description="Standard error from the code execution"
    )
    
    execution_time: Optional[float] = Field(
        default=None,
        description="Execution time in seconds"
    ) 