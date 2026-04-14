"""
Entry point for the AlphaMind backend.
Delegates to the FastAPI application defined in api/main.py.
"""

import os

from app.main import app  # noqa: F401 - re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_DEBUG", "false").lower() == "true",
        log_level="info",
    )
