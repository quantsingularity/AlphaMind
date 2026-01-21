"""Trading strategies router."""

from datetime import datetime
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Strategy(BaseModel):
    id: str
    name: str
    description: str
    status: str
    created_at: datetime


class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float


class BacktestResult(BaseModel):
    strategy_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades: int


@router.get("/", response_model=List[Strategy])
async def get_strategies():
    """Get all trading strategies."""
    return []


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run strategy backtest."""
    return {
        "strategy_id": request.strategy_id,
        "total_return": 0.15,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.08,
        "trades": 100,
    }
