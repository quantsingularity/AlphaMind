"""Alpha Vantage API connector."""

import time

from market_data.api_connectors.base import APIConnector, DataResponse


class AlphaVantageConnector(APIConnector):
    """Connector for Alpha Vantage API."""

    def get_quote(self, symbol: str) -> DataResponse:
        """Get quote from Alpha Vantage."""
        return DataResponse(
            provider="AlphaVantage",
            symbol=symbol,
            data={"price": 100.0, "volume": 1000000},
            timestamp=time.time(),
        )

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataResponse:
        """Get historical data."""
        return DataResponse(
            provider="AlphaVantage", symbol=symbol, data=[], timestamp=time.time()
        )

    def get_fundamentals(self, symbol: str) -> DataResponse:
        """Get fundamentals."""
        return DataResponse(
            provider="AlphaVantage", symbol=symbol, data={}, timestamp=time.time()
        )
