"""
IEX Cloud API connector for financial data.

This module provides a connector for accessing financial market data
from IEX Cloud, including stock prices, company data, market information,
and more.
"""

from datetime import datetime, timedelta
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .base import (
    APIConnector,
    APICredentials,
    DataCategory,
    DataFormat,
    DataProvider,
    DataResponse,
    RateLimiter,
)


class IEXCloudConnector(APIConnector):
    """
    Connector for IEX Cloud API.

    This class provides methods for accessing financial market data
    from IEX Cloud, including stock prices, company data, market information,
    and more.

    Parameters
    ----------
    api_key : str
        IEX Cloud API key (publishable token).
    api_secret : str, optional
        IEX Cloud API secret (secret token).
    sandbox : bool, default=False
        Whether to use the sandbox environment.
    version : str, default="v1"
        API version.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
        version: str = "v1",
    ):
        # Create credentials
        credentials = APICredentials(api_key=api_key, api_secret=api_secret)

        # Set base URL based on environment
        if sandbox:
            base_url = f"https://sandbox.iexapis.com/{version}"
        else:
            base_url = f"https://cloud.iexapis.com/{version}"

        # Create rate limiter
        # IEX Cloud has a limit of 100 requests per second per IP
        rate_limiter = RateLimiter(requests_per_second=100)

        super().__init__(
            credentials=credentials, base_url=base_url, rate_limiter=rate_limiter
        )

        self.sandbox = sandbox
        self.version = version
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def provider_name(self) -> str:
        """
        Get the name of the data provider.

        Returns
        -------
        provider_name : str
            Name of the data provider.
        """
        return DataProvider.IEX_CLOUD.value

    def authenticate(self) -> bool:
        """
        Authenticate with the IEX Cloud API.

        Returns
        -------
        success : bool
            Whether authentication was successful.
        """
        # IEX Cloud uses API key for authentication
        # Test authentication by making a simple request
        response = self.get_quote("AAPL")

        return response.is_success()

    def _add_token_param(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add token parameter to request parameters.

        Parameters
        ----------
        params : dict, optional
            Request parameters.

        Returns
        -------
        params : dict
            Request parameters with token added.
        """
        params = params or {}
        params["token"] = self.credentials.api_key
        return params

    def get_quote(self, symbol: str) -> DataResponse:
        """
        Get real-time stock quote.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the stock quote.
        """
        endpoint = f"stock/{symbol}/quote"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_historical_prices(
        self,
        symbol: str,
        range: str = "1m",
        chart_by_day: bool = True,
        chart_close_only: bool = False,
        chart_interval: Optional[int] = None,
        chart_reset: bool = False,
        chart_simplify: bool = False,
    ) -> DataResponse:
        """
        Get historical stock prices.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        range : str, default="1m"
            Time range. Options: "max", "5y", "2y", "1y", "ytd", "6m", "3m", "1m", "1mm", "5d", "5dm", "1d", "dynamic".
        chart_by_day : bool, default=True
            Whether to return data by day.
        chart_close_only : bool, default=False
            Whether to return only closing prices.
        chart_interval : int, optional
            Chart interval in minutes.
        chart_reset : bool, default=False
            Whether to reset the data.
        chart_simplify : bool, default=False
            Whether to simplify the data.

        Returns
        -------
        response : DataResponse
            Response containing the historical prices.
        """
        endpoint = f"stock/{symbol}/chart/{range}"

        params = self._add_token_param(
            {
                "chartByDay": str(chart_by_day).lower(),
                "chartCloseOnly": str(chart_close_only).lower(),
                "chartReset": str(chart_reset).lower(),
                "chartSimplify": str(chart_simplify).lower(),
            }
        )

        if chart_interval is not None:
            params["chartInterval"] = chart_interval

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_company(self, symbol: str) -> DataResponse:
        """
        Get company information.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the company information.
        """
        endpoint = f"stock/{symbol}/company"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_financials(
        self, symbol: str, period: str = "quarter", last: int = 1
    ) -> DataResponse:
        """
        Get financial statements.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        period : str, default="quarter"
            Period. Options: "annual", "quarter".
        last : int, default=1
            Number of periods to return.

        Returns
        -------
        response : DataResponse
            Response containing the financial statements.
        """
        endpoint = f"stock/{symbol}/financials"

        params = self._add_token_param({"period": period, "last": last})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_income_statement(
        self, symbol: str, period: str = "quarter", last: int = 1
    ) -> DataResponse:
        """
        Get income statement.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        period : str, default="quarter"
            Period. Options: "annual", "quarter".
        last : int, default=1
            Number of periods to return.

        Returns
        -------
        response : DataResponse
            Response containing the income statement.
        """
        endpoint = f"stock/{symbol}/income"

        params = self._add_token_param({"period": period, "last": last})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_balance_sheet(
        self, symbol: str, period: str = "quarter", last: int = 1
    ) -> DataResponse:
        """
        Get balance sheet.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        period : str, default="quarter"
            Period. Options: "annual", "quarter".
        last : int, default=1
            Number of periods to return.

        Returns
        -------
        response : DataResponse
            Response containing the balance sheet.
        """
        endpoint = f"stock/{symbol}/balance-sheet"

        params = self._add_token_param({"period": period, "last": last})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_cash_flow(
        self, symbol: str, period: str = "quarter", last: int = 1
    ) -> DataResponse:
        """
        Get cash flow statement.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        period : str, default="quarter"
            Period. Options: "annual", "quarter".
        last : int, default=1
            Number of periods to return.

        Returns
        -------
        response : DataResponse
            Response containing the cash flow statement.
        """
        endpoint = f"stock/{symbol}/cash-flow"

        params = self._add_token_param({"period": period, "last": last})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_earnings(self, symbol: str, last: int = 1) -> DataResponse:
        """
        Get earnings data.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        last : int, default=1
            Number of periods to return.

        Returns
        -------
        response : DataResponse
            Response containing the earnings data.
        """
        endpoint = f"stock/{symbol}/earnings"

        params = self._add_token_param({"last": last})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_dividends(self, symbol: str, range: str = "1m") -> DataResponse:
        """
        Get dividend data.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        range : str, default="1m"
            Time range. Options: "5y", "2y", "1y", "ytd", "6m", "3m", "1m", "next".

        Returns
        -------
        response : DataResponse
            Response containing the dividend data.
        """
        endpoint = f"stock/{symbol}/dividends/{range}"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_splits(self, symbol: str, range: str = "1m") -> DataResponse:
        """
        Get stock split data.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        range : str, default="1m"
            Time range. Options: "5y", "2y", "1y", "ytd", "6m", "3m", "1m", "next".

        Returns
        -------
        response : DataResponse
            Response containing the stock split data.
        """
        endpoint = f"stock/{symbol}/splits/{range}"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_news(self, symbol: str, last: int = 10) -> DataResponse:
        """
        Get news for a symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        last : int, default=10
            Number of news items to return.

        Returns
        -------
        response : DataResponse
            Response containing the news.
        """
        endpoint = f"stock/{symbol}/news/last/{last}"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.NEWS,
            format=DataFormat.JSON,
        )

    def get_peers(self, symbol: str) -> DataResponse:
        """
        Get peer symbols.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the peer symbols.
        """
        endpoint = f"stock/{symbol}/peers"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_stats(self, symbol: str) -> DataResponse:
        """
        Get key stats for a symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the key stats.
        """
        endpoint = f"stock/{symbol}/stats"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_largest_trades(self, symbol: str) -> DataResponse:
        """
        Get largest trades for a symbol.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the largest trades.
        """
        endpoint = f"stock/{symbol}/largest-trades"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_market_volume(self) -> DataResponse:
        """
        Get market volume.

        Returns
        -------
        response : DataResponse
            Response containing the market volume.
        """
        endpoint = "market/volume"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_market_list(
        self, list_type: str = "mostactive", list_size: int = 10
    ) -> DataResponse:
        """
        Get market list.

        Parameters
        ----------
        list_type : str, default="mostactive"
            List type. Options: "mostactive", "gainers", "losers", "iexvolume", "iexpercent", "infocus".
        list_size : int, default=10
            Number of items to return.

        Returns
        -------
        response : DataResponse
            Response containing the market list.
        """
        endpoint = f"stock/market/list/{list_type}"

        params = self._add_token_param({"listLimit": list_size})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_sector_performance(self) -> DataResponse:
        """
        Get sector performance.

        Returns
        -------
        response : DataResponse
            Response containing the sector performance.
        """
        endpoint = "stock/market/sector-performance"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_crypto_quote(self, symbol: str) -> DataResponse:
        """
        Get cryptocurrency quote.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol.

        Returns
        -------
        response : DataResponse
            Response containing the cryptocurrency quote.
        """
        endpoint = f"crypto/{symbol}/quote"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_forex_rates(self, symbols: Optional[List[str]] = None) -> DataResponse:
        """
        Get forex exchange rates.

        Parameters
        ----------
        symbols : list, optional
            List of forex symbols. If None, returns all available rates.

        Returns
        -------
        response : DataResponse
            Response containing the forex exchange rates.
        """
        endpoint = "fx/latest"

        params = self._add_token_param()

        if symbols:
            params["symbols"] = ",".join(symbols)

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_options(
        self, symbol: str, expiration_date: Optional[str] = None
    ) -> DataResponse:
        """
        Get options data.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        expiration_date : str, optional
            Expiration date in format YYYYMMDD. If None, returns all available dates.

        Returns
        -------
        response : DataResponse
            Response containing the options data.
        """
        if expiration_date:
            endpoint = f"stock/{symbol}/options/{expiration_date}"
        else:
            endpoint = f"stock/{symbol}/options"

        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_batch_quotes(self, symbols: List[str]) -> DataResponse:
        """
        Get batch quotes for multiple symbols.

        Parameters
        ----------
        symbols : list
            List of stock symbols.

        Returns
        -------
        response : DataResponse
            Response containing the batch quotes.
        """
        endpoint = "stock/market/batch"

        params = self._add_token_param({"symbols": ",".join(symbols), "types": "quote"})

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_batch_data(
        self,
        symbols: List[str],
        types: List[str],
        range: Optional[str] = None,
        last: Optional[int] = None,
    ) -> DataResponse:
        """
        Get batch data for multiple symbols and data types.

        Parameters
        ----------
        symbols : list
            List of stock symbols.
        types : list
            List of data types. Options: "quote", "news", "chart", "price", "company", "stats", "peers", "financials", "earnings", "dividends", "splits", "logo", "price".
        range : str, optional
            Time range for chart data. Options: "max", "5y", "2y", "1y", "ytd", "6m", "3m", "1m", "1mm", "5d", "5dm", "1d", "dynamic".
        last : int, optional
            Number of periods to return for financials and earnings.

        Returns
        -------
        response : DataResponse
            Response containing the batch data.
        """
        endpoint = "stock/market/batch"

        params = self._add_token_param(
            {"symbols": ",".join(symbols), "types": ",".join(types)}
        )

        if range:
            params["range"] = range

        if last:
            params["last"] = last

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def search(self, fragment: str) -> DataResponse:
        """
        Search for symbols.

        Parameters
        ----------
        fragment : str
            Search fragment.

        Returns
        -------
        response : DataResponse
            Response containing the search results.
        """
        endpoint = f"search/{fragment}"
        params = self._add_token_param()

        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )
