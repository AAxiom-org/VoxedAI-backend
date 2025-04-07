"""
API router for v1 endpoints.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import files, agent, code, graph, research

api_router = APIRouter()

api_router.include_router(
    files.router, prefix="/files", tags=["file"]
)
api_router.include_router(
    agent.router, prefix="/agent", tags=["agent", "chat"]
)
api_router.include_router(
    code.router, prefix="/code", tags=["code"]
)
api_router.include_router(
    graph.router, prefix="/graph", tags=["graph"]
)
api_router.include_router(
    research.router, prefix="/research", tags=["research", "digest"]
) 