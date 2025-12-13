"""Market data router."""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime

router = APIRouter()


class Quote(BaseModel):
    symbol: str
    price: float
    volume: int
    timestamp: datetime


class HistoricalData(BaseModel):
    symbol: str
    data: List[dict]


@router.get("/quote/{symbol}", response_model=Quote)
async def get_quote(symbol: str):
    """Get current quote for symbol."""
    return {
        "symbol": symbol,
        "price": 100.0,
        "volume": 1000000,
        "timestamp": datetime.now(),
    }


@router.get("/historical/{symbol}", response_model=HistoricalData)
async def get_historical(symbol: str, days: int = Query(30, ge=1, le=365)):
    """Get historical data for symbol."""
    return {"symbol": symbol, "data": []}
