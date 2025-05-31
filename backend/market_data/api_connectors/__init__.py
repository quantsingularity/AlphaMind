"""
AlphaMind API Connectors for Financial Data Providers

This package provides connectors for accessing financial market data
from various data providers, including real-time and historical data,
fundamental data, and alternative data sources.
"""

from .base import (
    DataProvider,
    APIConnector,
    APICredentials,
    DataRequest,
    DataResponse,
    RateLimiter
)

from .alpha_vantage import AlphaVantageConnector
from .bloomberg import BloombergConnector
from .refinitiv import RefinitivConnector
from .yahoo_finance import YahooFinanceConnector
from .iex_cloud import IEXCloudConnector
from .quandl import QuandlConnector
from .polygon import PolygonConnector
from .tiingo import TiingoConnector
from .fred import FREDConnector
from .intrinio import IntrinioConnector

__all__ = [
    'DataProvider', 'APIConnector', 'APICredentials', 'DataRequest', 'DataResponse', 'RateLimiter',
    'AlphaVantageConnector', 'BloombergConnector', 'RefinitivConnector', 'YahooFinanceConnector',
    'IEXCloudConnector', 'QuandlConnector', 'PolygonConnector', 'TiingoConnector', 'FREDConnector',
    'IntrinioConnector'
]
