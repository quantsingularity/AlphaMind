"""
Entry point for the AlphaMind backend.
Delegates to the FastAPI application defined in api/main.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.main import app  # noqa: F401 - re-exported for uvicorn

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_DEBUG", "false").lower() == "true",
        log_level="info",
    )
