"""
Base classes for API connectors.
"""

import logging
import threading
import time
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported data providers."""

    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    FRED = "fred"
    BLOOMBERG = "bloomberg"
    YAHOO_FINANCE = "yahoo_finance"
    REFINITIV = "refinitiv"
    TIINGO = "tiingo"
    QUANDL = "quandl"
    INTRINIO = "intrinio"


class DataCategory(Enum):
    """Category of data being requested from a provider."""

    MARKET_DATA = "market_data"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    ECONOMIC = "economic"
    ALTERNATIVE = "alternative"
    REFERENCE = "reference"


class DataFormat(Enum):
    """Expected payload format of an API response."""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    TEXT = "text"


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

    provider: str = ""
    symbol: str = ""
    data: Any = None
    timestamp: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, Any]] = None
    category: Optional[DataCategory] = None
    format: Optional[DataFormat] = None
    endpoint: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    request: Optional[Any] = None

    def is_success(self) -> bool:
        """Return True if the response represents a successful call."""
        if self.error is not None:
            return False
        if self.status_code is not None:
            return 200 <= self.status_code < 300
        return self.data is not None


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(
        self,
        calls_per_minute: Optional[int] = None,
        requests_per_second: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        max_requests: Optional[int] = None,
        time_window: Optional[float] = None,
    ) -> None:
        """
        Initialize rate limiter.

        Accepts several equivalent ways of expressing a rate limit and
        normalizes them to a maximum number of calls allowed per 60s window.

        Args:
            calls_per_minute: Maximum calls allowed per minute.
            requests_per_second: Maximum requests per second.
            requests_per_minute: Maximum requests per minute.
            requests_per_hour: Maximum requests per hour.
            max_requests: Maximum requests within ``time_window`` seconds.
            time_window: Window length in seconds for ``max_requests``.
        """
        if requests_per_second is not None:
            limit = requests_per_second * 60
        elif requests_per_minute is not None:
            limit = requests_per_minute
        elif requests_per_hour is not None:
            limit = requests_per_hour / 60.0
        elif max_requests is not None and time_window:
            limit = max_requests * (60.0 / time_window)
        elif calls_per_minute is not None:
            limit = calls_per_minute
        else:
            limit = 5
        self.calls_per_minute = max(1, int(limit))
        self.calls: list[float] = []
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
        base_url: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        """
        Initialize API connector.

        Args:
            credentials: API credentials
            base_url: Base URL for the provider's REST API
            rate_limiter: Rate limiter instance
        """
        self.credentials = credentials
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.session = requests.Session()
        try:
            self.provider = self.provider_name
        except Exception:
            self.provider = self.__class__.__name__

    @property
    def provider_name(self) -> str:
        """Human-readable provider name. Subclasses may override."""
        return self.__class__.__name__

    def request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        category: Optional[DataCategory] = None,
        format: DataFormat = DataFormat.JSON,
        headers: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        symbol: str = "",
    ) -> DataResponse:
        """
        Perform a rate-limited HTTP request against the provider API.

        Args:
            endpoint: Path appended to ``base_url`` (leading slash optional).
            params: Query-string parameters.
            category: Logical category of the requested data.
            format: Expected response payload format.
            headers: Optional HTTP headers.
            method: HTTP method.
            symbol: Symbol associated with the request (for the response).

        Returns:
            A populated :class:`DataResponse`.
        """
        self.rate_limiter.acquire()
        base = (self.base_url or "").rstrip("/")
        url = f"{base}/{endpoint.lstrip('/')}" if base else endpoint
        try:
            resp = self.session.request(
                method, url, params=params, headers=headers, timeout=30
            )
            if format == DataFormat.JSON:
                try:
                    data = resp.json() if resp.content else None
                except ValueError:
                    data = resp.text
            else:
                data = resp.text
            error = None if resp.ok else f"HTTP {resp.status_code}: {resp.text[:200]}"
            return DataResponse(
                provider=self.provider,
                symbol=symbol,
                data=data,
                timestamp=time.time(),
                status_code=resp.status_code,
                headers=dict(resp.headers),
                category=category,
                format=format,
                endpoint=endpoint,
                params=params,
                error=error,
            )
        except Exception as e:
            self.logger.error(f"Request to {url} failed: {e}")
            return DataResponse(
                provider=self.provider,
                symbol=symbol,
                data=None,
                timestamp=time.time(),
                category=category,
                format=format,
                endpoint=endpoint,
                params=params,
                error=str(e),
            )

    def get_quote(self, symbol: str) -> DataResponse:
        """Get real-time quote for symbol. Override in provider subclass."""
        return DataResponse(
            provider=self.provider,
            symbol=symbol,
            timestamp=time.time(),
            error="get_quote not implemented for this provider",
        )

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> DataResponse:
        """Get historical price data. Override in provider subclass."""
        return DataResponse(
            provider=self.provider,
            symbol=symbol,
            timestamp=time.time(),
            error="get_historical_data not implemented for this provider",
        )

    def get_fundamentals(self, symbol: str) -> DataResponse:
        """Get fundamental data for symbol. Override in provider subclass."""
        return DataResponse(
            provider=self.provider,
            symbol=symbol,
            timestamp=time.time(),
            error="get_fundamentals not implemented for this provider",
        )

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
