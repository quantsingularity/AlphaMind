"""Research papers router.

Serves the research catalogue consumed by the mobile client's Research screen.
The catalogue is a static seed list; swap ``_PAPERS`` for a repository-backed
source when papers are persisted in the database.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_PAPERS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "title": "Deep Reinforcement Learning for Risk-Aware Portfolio Optimization",
        "authors": ["AlphaMind Research"],
        "abstract": (
            "A reinforcement-learning framework that balances expected return "
            "against drawdown and volatility constraints for portfolio allocation."
        ),
        "category": "Machine Learning",
        "year": 2025,
        "url": "",
    },
    {
        "id": "2",
        "title": "Explainable Volatility Forecasting with LSTM-Attention",
        "authors": ["AlphaMind Research"],
        "abstract": (
            "An attention-augmented LSTM with SHAP attributions for interpretable "
            "short-horizon volatility forecasting."
        ),
        "category": "Machine Learning",
        "year": 2025,
        "url": "",
    },
    {
        "id": "3",
        "title": "Detecting Algorithmic Spoofing with Graph Neural Networks",
        "authors": ["AlphaMind Research"],
        "abstract": (
            "High-frequency market-microstructure analysis combining temporal "
            "event networks and GNNs to flag spoofing patterns."
        ),
        "category": "Market Microstructure",
        "year": 2026,
        "url": "",
    },
    {
        "id": "4",
        "title": "Factor Investing with Alternative Data",
        "authors": ["AlphaMind Research"],
        "abstract": (
            "Construction of cross-sectional equity factors from alternative data "
            "sources and their incremental predictive power."
        ),
        "category": "Alternative Data",
        "year": 2026,
        "url": "",
    },
]


@router.get("/papers")
async def list_papers(
    category: Optional[str] = Query(default=None),
) -> List[Dict[str, Any]]:
    """Return research papers, optionally filtered by category."""
    if category:
        return [p for p in _PAPERS if p["category"].lower() == category.lower()]
    return _PAPERS


@router.get("/papers/{paper_id}")
async def get_paper(paper_id: str) -> Dict[str, Any]:
    """Return a single research paper by ID."""
    for paper in _PAPERS:
        if paper["id"] == paper_id:
            return paper
    raise HTTPException(status_code=404, detail="Paper not found")
