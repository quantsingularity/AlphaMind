"""Portfolio management router."""

from typing import Any, Dict, List

from app.services import PortfolioService, get_portfolio_service
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class AssetAllocation(BaseModel):
    ticker: str
    value: float
    percentage: float


class Position(BaseModel):
    id: str
    ticker: str
    sector: str
    quantity: float
    entryPrice: float
    currentPrice: float
    unrealizedPnL: float
    realizedPnL: float
    weight: float
    beta: float
    sharpeContrib: float
    var95: float
    timestamp: str | None = None


class Portfolio(BaseModel):
    id: str
    name: str
    totalValue: float
    cash: float
    dailyPnL: float
    totalPnL: float
    allocation: List[AssetAllocation]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/", response_model=Portfolio)
async def get_portfolio(
    svc: PortfolioService = Depends(get_portfolio_service),
) -> Dict[str, Any]:
    """Return current portfolio summary."""
    return await svc.get_portfolio()


@router.get("/holdings")
async def get_holdings(
    svc: PortfolioService = Depends(get_portfolio_service),
) -> List[Dict[str, Any]]:
    """Return simplified holdings list (used by mobile)."""
    return await svc.get_holdings()


@router.get("/performance")
async def get_performance(
    timeframe: str = "1M",
    svc: PortfolioService = Depends(get_portfolio_service),
) -> Dict[str, Any]:
    """Return portfolio performance metrics and equity curve."""
    return await svc.get_performance(timeframe=timeframe)


@router.get("/positions", response_model=List[Position])
async def get_positions(
    svc: PortfolioService = Depends(get_portfolio_service),
) -> List[Dict[str, Any]]:
    """Return all open positions."""
    return await svc.get_positions()


@router.get("/positions/{position_id}", response_model=Position)
async def get_position(
    position_id: str,
    svc: PortfolioService = Depends(get_portfolio_service),
) -> Dict[str, Any]:
    """Return a single position by ID."""
    pos = await svc.get_position(position_id)
    if pos is None:
        raise HTTPException(status_code=404, detail="Position not found")
    return pos


@router.post("/positions/{position_id}/close")
async def close_position(
    position_id: str,
    svc: PortfolioService = Depends(get_portfolio_service),
) -> Dict[str, Any]:
    """Close an open position and realise its PnL."""
    result = await svc.close_position(position_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Position not found or already closed"
        )
    return result
