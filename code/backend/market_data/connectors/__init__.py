"""
AlphaMind Market Data Connectors.

Unified API connector layer for Bloomberg, Refinitiv, Yahoo Finance,
Polygon, Tiingo, Quandl, Intrinio, IEX Cloud, FRED, Alpha Vantage.
"""

from market_data.connectors.alpha_vantage import AlphaVantageConnector
from market_data.connectors.base import (
    APIConnector,
    APICredentials,
    DataProvider,
    DataRequest,
    DataResponse,
    RateLimiter,
)
from market_data.connectors.bloomberg import BloombergConnector
from market_data.connectors.fred import FREDConnector
from market_data.connectors.iex_cloud import IEXCloudConnector
from market_data.connectors.intrinio import IntrinioConnector
from market_data.connectors.polygon import PolygonConnector
from market_data.connectors.quandl import QuandlConnector
from market_data.connectors.refinitiv import RefinitivConnector
from market_data.connectors.tiingo import TiingoConnector
from market_data.connectors.yahoo_finance import YahooFinanceConnector

__all__ = [
    "DataProvider",
    "APIConnector",
    "APICredentials",
    "DataRequest",
    "DataResponse",
    "RateLimiter",
    "AlphaVantageConnector",
    "BloombergConnector",
    "RefinitivConnector",
    "YahooFinanceConnector",
    "IEXCloudConnector",
    "QuandlConnector",
    "PolygonConnector",
    "TiingoConnector",
    "FREDConnector",
    "IntrinioConnector",
]
