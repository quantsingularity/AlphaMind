"""
AlphaMind API Connectors for Financial Data Providers

This package provides connectors for accessing financial market data
from various data providers, including real-time and historical data,
fundamental data, and alternative data sources.
"""

from market_data.api_connectors.alpha_vantage import AlphaVantageConnector
from market_data.api_connectors.base import (
    APIConnector,
    APICredentials,
    DataProvider,
    DataRequest,
    DataResponse,
    RateLimiter,
)
from market_data.api_connectors.bloomberg import BloombergConnector
from market_data.api_connectors.fred import FREDConnector
from market_data.api_connectors.iex_cloud import IEXCloudConnector
from market_data.api_connectors.intrinio import IntrinioConnector
from market_data.api_connectors.polygon import PolygonConnector
from market_data.api_connectors.quandl import QuandlConnector
from market_data.api_connectors.refinitiv import RefinitivConnector
from market_data.api_connectors.tiingo import TiingoConnector
from market_data.api_connectors.yahoo_finance import YahooFinanceConnector

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
