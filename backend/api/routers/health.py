"""Health check router."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "alphamind-api", "version": "1.0.0"}


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {"status": "healthy", "service": "alphamind-api", "version": "1.0.0"}
