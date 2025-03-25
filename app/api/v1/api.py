"""
API router for v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import files, agent, code

api_router = APIRouter()

api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(code.router, prefix="/code", tags=["code"]) 