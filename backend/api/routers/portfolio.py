"""Portfolio management router."""

from datetime import datetime
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Position(BaseModel):
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    pnl: float


class Portfolio(BaseModel):
    total_value: float
    cash: float
    positions: List[Position]
    updated_at: datetime


@router.get("/", response_model=Portfolio)
async def get_portfolio():
    """Get current portfolio."""
    return {
        "total_value": 100000.0,
        "cash": 50000.0,
        "positions": [],
        "updated_at": datetime.now(),
    }


@router.get("/performance")
async def get_performance():
    """Get portfolio performance metrics."""
    return {
        "daily_return": 0.0,
        "total_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
    }
