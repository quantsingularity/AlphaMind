"""
Market data service tests.

The service uses a live-connector waterfall (Yahoo Finance, then Polygon) with a
deterministic synthetic fallback. These tests verify both the response contract
and that the service stays resilient when the live source is available or not.
"""

from __future__ import annotations

import pytest
from app.services.market_data_service import MarketDataService


@pytest.mark.asyncio
async def test_quote_contract_and_source_flag():
    svc = MarketDataService()
    quote = await svc.get_quote("AAPL")
    for field in [
        "ticker",
        "last",
        "bid",
        "ask",
        "volume",
        "open",
        "high",
        "low",
        "close",
        "source",
    ]:
        assert field in quote, f"quote missing {field}"
    assert quote["source"] in {"live", "synthetic"}


@pytest.mark.asyncio
async def test_ohlcv_contract():
    svc = MarketDataService()
    bars = await svc.get_ohlcv("AAPL", days=30)
    assert isinstance(bars, list) and bars
    for field in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert field in bars[0], f"ohlcv missing {field}"


@pytest.mark.asyncio
async def test_live_price_is_used_when_available(monkeypatch):
    """When a live connector returns a price, the quote is flagged as live."""
    svc = MarketDataService()

    async def fake_live_price(ticker):
        return 123.45

    monkeypatch.setattr(svc, "_fetch_live_price", fake_live_price)
    quote = await svc.get_quote("AAPL")
    assert quote["source"] == "live"
    assert quote["last"] == 123.45


@pytest.mark.asyncio
async def test_synthetic_fallback_when_live_unavailable(monkeypatch):
    """When live connectors return nothing, the service still serves valid data."""
    svc = MarketDataService()

    async def no_live_price(ticker):
        return None

    monkeypatch.setattr(svc, "_fetch_live_price", no_live_price)
    quote = await svc.get_quote("AAPL")
    assert quote["source"] == "synthetic"
    assert quote["last"] > 0  # synthetic feed still produces a usable price


def test_synthetic_is_deterministic():
    """The synthetic fallback must be deterministic for reproducible demos/tests."""
    svc = MarketDataService()
    a = svc._synthetic_ohlcv("AAPL", 10)
    b = svc._synthetic_ohlcv("AAPL", 10)
    assert [x["close"] for x in a] == [x["close"] for x in b]
