import os
from typing import Dict, Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings.

    These settings are loaded from environment variables.
    """

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "VoxAI Backend"

    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Pinecone Configuration
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_REGION_LLAMA: str = Field(..., env="PINECONE_REGION_LLAMA")
    PINECONE_HOST_LLAMA: str = Field(..., env="PINECONE_HOST_LLAMA")
    PINECONE_INDEX_LLAMA: str = Field(..., env="PINECONE_INDEX_LLAMA")
    PINECONE_FIELD_LLAMA: str = Field(..., env="PINECONE_FIELD_LLAMA")

    # Supabase Configuration
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_ANON_KEY: str = Field(..., env="SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY: str = Field(..., env="SUPABASE_SERVICE_ROLE_KEY")

    # LLM API Keys
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(..., env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
    GEMINI_API_KEY: str = Field(..., env="GEMINI_API_KEY")
    
    # Code Execution Sandbox Configuration
    CODE_SANDBOX_URL: str = Field("http://localhost:8001", env="CODE_SANDBOX_URL")

    class Config:
        case_sensitive = True
        env_file = ".env"


# Create global settings object
settings = Settings() 