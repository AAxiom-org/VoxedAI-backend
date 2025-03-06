"""
API router for v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import files, query, code

api_router = APIRouter()

api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(code.router, prefix="/code", tags=["code"]) 