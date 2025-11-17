"""
Base classes for API connectors to financial data providers.

This module provides abstract base classes and utility functions
for implementing API connectors to financial data providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import pandas as pd
import requests


class DataFrequency(Enum):
    """Enumeration of data frequencies."""

    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    THIRTY_MINUTE = "30m"
    HOUR = "1h"
    FOUR_HOUR = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"
    QUARTER = "3mo"
    YEAR = "1y"


class DataCategory(Enum):
    """Enumeration of data categories."""

    MARKET_DATA = "market_data"
    FUNDAMENTAL = "fundamental"
    ECONOMIC = "economic"
    ALTERNATIVE = "alternative"
    NEWS = "news"
    SOCIAL = "social"
    SENTIMENT = "sentiment"


class DataFormat(Enum):
    """Enumeration of data formats."""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    EXCEL = "excel"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class APICredentials:
    """
    Class for storing API credentials.

    This class provides a secure way to store and manage
    API credentials for financial data providers.

    Parameters
    ----------
    api_key : str, optional
        API key for authentication.
    api_secret : str, optional
        API secret for authentication.
    username : str, optional
        Username for authentication.
    password : str, optional
        Password for authentication.
    token : str, optional
        Authentication token.
    additional_params : dict, optional
        Additional authentication parameters.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.username = username
        self.password = password
        self.token = token
        self.additional_params = additional_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert credentials to dictionary.

        Returns
        -------
        credentials_dict : dict
            Dictionary representation of the credentials.
        """
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "username": self.username,
            "password": self.password,
            "token": self.token,
            "additional_params": self.additional_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APICredentials":
        """
        Create credentials from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the credentials.

        Returns
        -------
        credentials : APICredentials
            Credentials created from the dictionary.
        """
        return cls(
            api_key=data.get("api_key"),
            api_secret=data.get("api_secret"),
            username=data.get("username"),
            password=data.get("password"),
            token=data.get("token"),
            additional_params=data.get("additional_params", {}),
        )

    @classmethod
    def from_env(cls, prefix: str = "") -> "APICredentials":
        """
        Create credentials from environment variables.

        Parameters
        ----------
        prefix : str, default=""
            Prefix for environment variables.

        Returns
        -------
        credentials : APICredentials
            Credentials created from environment variables.
        """
        return cls(
            api_key=os.environ.get(f"{prefix}API_KEY"),
            api_secret=os.environ.get(f"{prefix}API_SECRET"),
            username=os.environ.get(f"{prefix}USERNAME"),
            password=os.environ.get(f"{prefix}PASSWORD"),
            token=os.environ.get(f"{prefix}TOKEN"),
            additional_params=json.loads(
                os.environ.get(f"{prefix}ADDITIONAL_PARAMS", "{}")
            ),
        )

    @classmethod
    def from_file(cls, filepath: str) -> "APICredentials":
        """
        Create credentials from a file.

        Parameters
        ----------
        filepath : str
            Path to the credentials file.

        Returns
        -------
        credentials : APICredentials
            Credentials created from the file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save(self, filepath: str) -> None:
        """
        Save credentials to a file.

        Parameters
        ----------
        filepath : str
            Path to save the credentials to.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class DataRequest:
    """
    Class for representing a data request.

    This class provides a common structure for data requests
    to financial data providers.

    Parameters
    ----------
    provider : str
        Name of the data provider.
    endpoint : str
        API endpoint to request data from.
    method : str, default="GET"
        HTTP method to use for the request.
    params : dict, optional
        Query parameters for the request.
    data : dict, optional
        Request body data.
    headers : dict, optional
        HTTP headers for the request.
    category : DataCategory, optional
        Category of data being requested.
    format : DataFormat, optional
        Format of the data being requested.
    """

    def __init__(
        self,
        provider: str,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        category: Optional[DataCategory] = None,
        format: Optional[DataFormat] = None,
    ):
        self.id = str(uuid.uuid4())
        self.provider = provider
        self.endpoint = endpoint
        self.method = method
        self.params = params or {}
        self.data = data or {}
        self.headers = headers or {}
        self.category = category
        self.format = format
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert request to dictionary.

        Returns
        -------
        request_dict : dict
            Dictionary representation of the request.
        """
        return {
            "id": self.id,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "method": self.method,
            "params": self.params,
            "data": self.data,
            "headers": self.headers,
            "category": self.category.value if self.category else None,
            "format": self.format.value if self.format else None,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataRequest":
        """
        Create request from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the request.

        Returns
        -------
        request : DataRequest
            Request created from the dictionary.
        """
        request = cls(
            provider=data["provider"],
            endpoint=data["endpoint"],
            method=data["method"],
            params=data.get("params", {}),
            data=data.get("data", {}),
            headers=data.get("headers", {}),
        )

        request.id = data["id"]

        if data.get("category"):
            request.category = DataCategory(data["category"])

        if data.get("format"):
            request.format = DataFormat(data["format"])

        request.timestamp = datetime.fromisoformat(data["timestamp"])

        return request


class DataResponse:
    """
    Class for representing a data response.

    This class provides a common structure for data responses
    from financial data providers.

    Parameters
    ----------
    request : DataRequest
        Request that generated this response.
    status_code : int
        HTTP status code of the response.
    data : any
        Response data.
    headers : dict, optional
        HTTP headers of the response.
    error : str, optional
        Error message if the request failed.
    """

    def __init__(
        self,
        request: DataRequest,
        status_code: int,
        data: Any,
        headers: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
    ):
        self.id = str(uuid.uuid4())
        self.request_id = request.id
        self.provider = request.provider
        self.endpoint = request.endpoint
        self.status_code = status_code
        self.data = data
        self.headers = headers or {}
        self.error = error
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary.

        Returns
        -------
        response_dict : dict
            Dictionary representation of the response.
        """
        return {
            "id": self.id,
            "request_id": self.request_id,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "data": self.data,
            "headers": self.headers,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], request: DataRequest) -> "DataResponse":
        """
        Create response from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the response.
        request : DataRequest
            Request that generated this response.

        Returns
        -------
        response : DataResponse
            Response created from the dictionary.
        """
        response = cls(
            request=request,
            status_code=data["status_code"],
            data=data["data"],
            headers=data.get("headers", {}),
            error=data.get("error"),
        )

        response.id = data["id"]
        response.timestamp = datetime.fromisoformat(data["timestamp"])

        return response

    def is_success(self) -> bool:
        """
        Check if the response was successful.

        Returns
        -------
        is_success : bool
            Whether the response was successful.
        """
        return 200 <= self.status_code < 300 and not self.error

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert response data to DataFrame.

        Returns
        -------
        df : DataFrame
            DataFrame created from the response data.
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data

        if isinstance(self.data, list):
            return pd.DataFrame(self.data)

        if isinstance(self.data, dict):
            # Try to extract data from common API response formats
            if "data" in self.data:
                data = self.data["data"]
                if isinstance(data, list):
                    return pd.DataFrame(data)

            if "results" in self.data:
                results = self.data["results"]
                if isinstance(results, list):
                    return pd.DataFrame(results)

            # Try to convert the entire dictionary to a DataFrame
            return pd.DataFrame([self.data])

        raise ValueError("Cannot convert response data to DataFrame")


class RateLimiter:
    """
    Class for rate limiting API requests.

    This class provides methods for enforcing rate limits
    on API requests to financial data providers.

    Parameters
    ----------
    requests_per_second : float, optional
        Maximum number of requests per second.
    requests_per_minute : int, optional
        Maximum number of requests per minute.
    requests_per_hour : int, optional
        Maximum number of requests per hour.
    requests_per_day : int, optional
        Maximum number of requests per day.
    """

    def __init__(
        self,
        requests_per_second: Optional[float] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None,
    ):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day

        self.second_timestamps = []
        self.minute_timestamps = []
        self.hour_timestamps = []
        self.day_timestamps = []

        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def wait(self) -> None:
        """
        Wait until a request can be made without exceeding rate limits.
        """
        with self.lock:
            now = time.time()

            # Check and enforce rate limits
            if self.requests_per_second:
                self._enforce_rate_limit(
                    self.second_timestamps, self.requests_per_second, now - 1.0
                )

            if self.requests_per_minute:
                self._enforce_rate_limit(
                    self.minute_timestamps, self.requests_per_minute, now - 60.0
                )

            if self.requests_per_hour:
                self._enforce_rate_limit(
                    self.hour_timestamps, self.requests_per_hour, now - 3600.0
                )

            if self.requests_per_day:
                self._enforce_rate_limit(
                    self.day_timestamps, self.requests_per_day, now - 86400.0
                )

            # Record the request
            now = time.time()

            if self.requests_per_second:
                self.second_timestamps.append(now)

            if self.requests_per_minute:
                self.minute_timestamps.append(now)

            if self.requests_per_hour:
                self.hour_timestamps.append(now)

            if self.requests_per_day:
                self.day_timestamps.append(now)

    def _enforce_rate_limit(
        self, timestamps: List[float], limit: Union[int, float], cutoff: float
    ) -> None:
        """
        Enforce a rate limit.

        Parameters
        ----------
        timestamps : list
            List of timestamps for previous requests.
        limit : int or float
            Maximum number of requests allowed.
        cutoff : float
            Cutoff time for considering requests.
        """
        # Remove old timestamps
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

        # Check if rate limit is exceeded
        if len(timestamps) >= limit:
            # Calculate wait time
            if timestamps:
                wait_time = cutoff - timestamps[0] + 0.1
                if wait_time > 0:
                    self.logger.debug(
                        f"Rate limit reached. Waiting {wait_time:.2f} seconds"
                    )
                    time.sleep(wait_time)

                    # Remove old timestamps again after waiting
                    now = time.time()
                    cutoff = now - (cutoff - timestamps[0])
                    while timestamps and timestamps[0] < cutoff:
                        timestamps.pop(0)


class DataProvider(Enum):
    """Enumeration of supported data providers."""

    ALPHA_VANTAGE = "alpha_vantage"
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    YAHOO_FINANCE = "yahoo_finance"
    IEX_CLOUD = "iex_cloud"
    QUANDL = "quandl"
    POLYGON = "polygon"
    TIINGO = "tiingo"
    FRED = "fred"
    INTRINIO = "intrinio"


class APIConnector(ABC):
    """
    Abstract base class for API connectors.

    This class provides a common interface for all API connectors
    to financial data providers.

    Parameters
    ----------
    credentials : APICredentials
        Credentials for authenticating with the API.
    base_url : str
        Base URL for the API.
    rate_limiter : RateLimiter, optional
        Rate limiter for enforcing API rate limits.
    """

    def __init__(
        self,
        credentials: APICredentials,
        base_url: str,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.credentials = credentials
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Get the name of the data provider.

        Returns
        -------
        provider_name : str
            Name of the data provider.
        """
        pass

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticate with the API.

        Returns
        -------
        success : bool
            Whether authentication was successful.
        """
        pass

    def request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        category: Optional[DataCategory] = None,
        format: Optional[DataFormat] = None,
    ) -> DataResponse:
        """
        Make a request to the API.

        Parameters
        ----------
        endpoint : str
            API endpoint to request data from.
        method : str, default="GET"
            HTTP method to use for the request.
        params : dict, optional
            Query parameters for the request.
        data : dict, optional
            Request body data.
        headers : dict, optional
            HTTP headers for the request.
        category : DataCategory, optional
            Category of data being requested.
        format : DataFormat, optional
            Format of the data being requested.

        Returns
        -------
        response : DataResponse
            Response from the API.
        """
        # Create request object
        request = DataRequest(
            provider=self.provider_name,
            endpoint=endpoint,
            method=method,
            params=params,
            data=data,
            headers=headers,
            category=category,
            format=format,
        )

        # Apply rate limiting if configured
        if self.rate_limiter:
            self.rate_limiter.wait()

        # Make the request
        try:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

            response = self.session.request(
                method=method, url=url, params=params, json=data, headers=headers
            )

            # Parse response data based on content type
            content_type = response.headers.get("Content-Type", "")

            if "application/json" in content_type:
                response_data = response.json()
            elif "text/csv" in content_type:
                response_data = pd.read_csv(pd.StringIO(response.text))
            elif "text/xml" in content_type or "application/xml" in content_type:
                response_data = response.text  # XML as string
            else:
                response_data = response.text

            # Create response object
            return DataResponse(
                request=request,
                status_code=response.status_code,
                data=response_data,
                headers=dict(response.headers),
            )

        except Exception as e:
            self.logger.error(f"Error making request: {e}")

            # Create error response
            return DataResponse(
                request=request, status_code=500, data=None, error=str(e)
            )

    def close(self) -> None:
        """Close the API connector."""
        self.session.close()

    def __enter__(self) -> "APIConnector":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()
