"""
Alpha Vantage API connector for financial data.

This module provides a connector for accessing financial market data
from typing import Any, Dict, List, Optional, Tuple, Union
from Alpha Vantage, including stock prices, technical indicators,
forex data, and more.
"""

import logging
from market_data.api_connectors.base import (
    APIConnector,
    APICredentials,
    DataCategory,
    DataFormat,
    DataProvider,
    DataResponse,
    RateLimiter,
)


class AlphaVantageConnector(APIConnector):
    """
    Connector for Alpha Vantage API.

    This class provides methods for accessing financial market data
    from Alpha Vantage, including stock prices, technical indicators,
    forex data, and more.

    Parameters
    ----------
    api_key : str
        Alpha Vantage API key.
    premium : bool, default=False
        Whether to use premium API features.
    """

    def __init__(self, api_key: str, premium: bool = False) -> Any:
        credentials = APICredentials(api_key=api_key)
        base_url = "https://www.alphavantage.co/query"
        if premium:
            rate_limiter = RateLimiter(requests_per_minute=75, requests_per_day=300)
        else:
            rate_limiter = RateLimiter(requests_per_minute=5, requests_per_day=500)
        super().__init__(
            credentials=credentials, base_url=base_url, rate_limiter=rate_limiter
        )
        self.premium = premium
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
        return DataProvider.ALPHA_VANTAGE.value

    def authenticate(self) -> bool:
        """
        Authenticate with the Alpha Vantage API.

        Returns
        -------
        success : bool
            Whether authentication was successful.
        """
        response = self.get_stock_quote("AAPL")
        return response.is_success()

    def get_stock_quote(self, symbol: str) -> DataResponse:
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
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_time_series(
        self,
        symbol: str,
        interval: str = "daily",
        outputsize: str = "compact",
        datatype: str = "json",
        adjusted: bool = True,
    ) -> DataResponse:
        """
        Get time series data for a stock.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        interval : str, default="daily"
            Time interval. Options: "intraday", "daily", "weekly", "monthly".
        outputsize : str, default="compact"
            Output size. Options: "compact" (last 100 data points), "full" (all data points).
        datatype : str, default="json"
            Data type. Options: "json", "csv".
        adjusted : bool, default=True
            Whether to return adjusted data.

        Returns
        -------
        response : DataResponse
            Response containing the time series data.
        """
        if interval == "intraday":
            function = "TIME_SERIES_INTRADAY"
        elif interval == "daily":
            function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        elif interval == "weekly":
            function = (
                "TIME_SERIES_WEEKLY_ADJUSTED" if adjusted else "TIME_SERIES_WEEKLY"
            )
        elif interval == "monthly":
            function = (
                "TIME_SERIES_MONTHLY_ADJUSTED" if adjusted else "TIME_SERIES_MONTHLY"
            )
        else:
            raise ValueError(f"Invalid interval: {interval}")
        params = {
            "function": function,
            "symbol": symbol,
            "outputsize": outputsize,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        if interval == "intraday":
            params["interval"] = "5min"
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
        datatype: str = "json",
    ) -> DataResponse:
        """
        Get technical indicator data for a stock.

        Parameters
        ----------
        symbol : str
            Stock symbol.
        indicator : str
            Technical indicator. Examples: "SMA", "EMA", "RSI", "MACD".
        interval : str, default="daily"
            Time interval. Options: "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly".
        time_period : int, default=14
            Time period for the indicator.
        series_type : str, default="close"
            Price series type. Options: "close", "open", "high", "low".
        datatype : str, default="json"
            Data type. Options: "json", "csv".

        Returns
        -------
        response : DataResponse
            Response containing the technical indicator data.
        """
        params = {
            "function": indicator,
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def get_forex_rate(self, from_currency: str, to_currency: str) -> DataResponse:
        """
        Get real-time forex exchange rate.

        Parameters
        ----------
        from_currency : str
            From currency code.
        to_currency : str
            To currency code.

        Returns
        -------
        response : DataResponse
            Response containing the forex exchange rate.
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_forex_time_series(
        self,
        from_currency: str,
        to_currency: str,
        interval: str = "daily",
        outputsize: str = "compact",
        datatype: str = "json",
    ) -> DataResponse:
        """
        Get forex time series data.

        Parameters
        ----------
        from_currency : str
            From currency code.
        to_currency : str
            To currency code.
        interval : str, default="daily"
            Time interval. Options: "intraday", "daily", "weekly", "monthly".
        outputsize : str, default="compact"
            Output size. Options: "compact" (last 100 data points), "full" (all data points).
        datatype : str, default="json"
            Data type. Options: "json", "csv".

        Returns
        -------
        response : DataResponse
            Response containing the forex time series data.
        """
        if interval == "intraday":
            function = "FX_INTRADAY"
        elif interval == "daily":
            function = "FX_DAILY"
        elif interval == "weekly":
            function = "FX_WEEKLY"
        elif interval == "monthly":
            function = "FX_MONTHLY"
        else:
            raise ValueError(f"Invalid interval: {interval}")
        params = {
            "function": function,
            "from_symbol": from_currency,
            "to_symbol": to_currency,
            "outputsize": outputsize,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        if interval == "intraday":
            params["interval"] = "5min"
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def get_crypto_rate(self, from_currency: str, to_currency: str) -> DataResponse:
        """
        Get real-time cryptocurrency exchange rate.

        Parameters
        ----------
        from_currency : str
            From currency code.
        to_currency : str
            To currency code.

        Returns
        -------
        response : DataResponse
            Response containing the cryptocurrency exchange rate.
        """
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_crypto_time_series(
        self,
        symbol: str,
        market: str = "USD",
        interval: str = "daily",
        outputsize: str = "compact",
        datatype: str = "json",
    ) -> DataResponse:
        """
        Get cryptocurrency time series data.

        Parameters
        ----------
        symbol : str
            Cryptocurrency symbol.
        market : str, default="USD"
            Market to convert to.
        interval : str, default="daily"
            Time interval. Options: "intraday", "daily", "weekly", "monthly".
        outputsize : str, default="compact"
            Output size. Options: "compact" (last 100 data points), "full" (all data points).
        datatype : str, default="json"
            Data type. Options: "json", "csv".

        Returns
        -------
        response : DataResponse
            Response containing the cryptocurrency time series data.
        """
        if interval == "intraday":
            function = "CRYPTO_INTRADAY"
        elif interval == "daily":
            function = "DIGITAL_CURRENCY_DAILY"
        elif interval == "weekly":
            function = "DIGITAL_CURRENCY_WEEKLY"
        elif interval == "monthly":
            function = "DIGITAL_CURRENCY_MONTHLY"
        else:
            raise ValueError(f"Invalid interval: {interval}")
        params = {
            "function": function,
            "symbol": symbol,
            "market": market,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        if interval == "intraday":
            params["interval"] = "5min"
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def get_sector_performance(self) -> DataResponse:
        """
        Get sector performance data.

        Returns
        -------
        response : DataResponse
            Response containing the sector performance data.
        """
        params = {"function": "SECTOR", "apikey": self.credentials.api_key}
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON,
        )

    def get_economic_indicator(
        self, indicator: str, interval: str = "annual", datatype: str = "json"
    ) -> DataResponse:
        """
        Get economic indicator data.

        Parameters
        ----------
        indicator : str
            Economic indicator. Examples: "REAL_GDP", "UNEMPLOYMENT", "INFLATION".
        interval : str, default="annual"
            Time interval. Options: "annual", "quarterly", "monthly", "daily".
        datatype : str, default="json"
            Data type. Options: "json", "csv".

        Returns
        -------
        response : DataResponse
            Response containing the economic indicator data.
        """
        params = {
            "function": indicator,
            "interval": interval,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def search_symbol(self, keywords: str, datatype: str = "json") -> DataResponse:
        """
        Search for symbols matching keywords.

        Parameters
        ----------
        keywords : str
            Keywords to search for.
        datatype : str, default="json"
            Data type. Options: "json", "csv".

        Returns
        -------
        response : DataResponse
            Response containing the search results.
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
            "datatype": datatype,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if datatype == "json" else DataFormat.CSV,
        )

    def get_company_overview(self, symbol: str) -> DataResponse:
        """
        Get company overview data.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the company overview data.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_income_statement(self, symbol: str) -> DataResponse:
        """
        Get income statement data.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the income statement data.
        """
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_balance_sheet(self, symbol: str) -> DataResponse:
        """
        Get balance sheet data.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the balance sheet data.
        """
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_cash_flow(self, symbol: str) -> DataResponse:
        """
        Get cash flow data.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the cash flow data.
        """
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )

    def get_earnings(self, symbol: str) -> DataResponse:
        """
        Get earnings data.

        Parameters
        ----------
        symbol : str
            Stock symbol.

        Returns
        -------
        response : DataResponse
            Response containing the earnings data.
        """
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": self.credentials.api_key,
        }
        return self.request(
            endpoint="",
            params=params,
            category=DataCategory.FUNDAMENTAL,
            format=DataFormat.JSON,
        )
