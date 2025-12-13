"""
Base classes for API connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, Optional
import time
import threading

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported data providers."""

    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    FRED = "fred"
    BLOOMBERG = "bloomberg"
    YAHOO_FINANCE = "yahoo_finance"


@dataclass
class APICredentials:
    """API credentials container."""

    api_key: str
    api_secret: Optional[str] = None
    additional_params: Optional[Dict[str, str]] = None


@dataclass
class DataRequest:
    """Data request parameters."""

    symbol: str
    data_type: str  # quote, historical, fundamentals, etc.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclass
class DataResponse:
    """Standardized data response."""

    provider: str
    symbol: str
    data: Any
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, calls_per_minute: int = 5):
        """
        Initialize rate limiter.

        Args:
            calls_per_minute: Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Wait if necessary to stay within rate limit."""
        with self.lock:
            now = time.time()
            # Remove calls older than 60 seconds
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]

            if len(self.calls) >= self.calls_per_minute:
                # Calculate wait time
                oldest_call = self.calls[0]
                wait_time = 60 - (now - oldest_call)
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    # Remove old calls after waiting
                    self.calls = [
                        call_time
                        for call_time in self.calls
                        if time.time() - call_time < 60
                    ]

            # Record this call
            self.calls.append(time.time())


class APIConnector(ABC):
    """Base class for API connectors."""

    def __init__(
        self,
        credentials: APICredentials,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize API connector.

        Args:
            credentials: API credentials
            rate_limiter: Rate limiter instance
        """
        self.credentials = credentials
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    @abstractmethod
    def get_quote(self, symbol: str) -> DataResponse:
        """Get real-time quote for symbol."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> DataResponse:
        """Get historical price data."""

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> DataResponse:
        """Get fundamental data for symbol."""

    def _make_request(self, request: DataRequest) -> DataResponse:
        """
        Make API request with rate limiting.

        Args:
            request: Data request

        Returns:
            Data response
        """
        self.rate_limiter.acquire()
        try:
            # Route to appropriate method based on data_type
            if request.data_type == "quote":
                return self.get_quote(request.symbol)
            elif request.data_type == "historical":
                return self.get_historical_data(
                    request.symbol,
                    request.start_date or "",
                    request.end_date or "",
                    request.interval or "1d",
                )
            elif request.data_type == "fundamentals":
                return self.get_fundamentals(request.symbol)
            else:
                return DataResponse(
                    provider=self.__class__.__name__,
                    symbol=request.symbol,
                    data=None,
                    timestamp=time.time(),
                    error=f"Unsupported data type: {request.data_type}",
                )
        except Exception as e:
            self.logger.error(f"Error making request: {e}")
            return DataResponse(
                provider=self.__class__.__name__,
                symbol=request.symbol,
                data=None,
                timestamp=time.time(),
                error=str(e),
            )

    def test_connection(self) -> bool:
        """
        Test API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to get a quote for a common symbol
            response = self.get_quote("AAPL")
            return response.error is None
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
