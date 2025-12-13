"""FRED API connector."""

from market_data.api_connectors.base import APIConnector, DataResponse
import time


class FREDConnector(APIConnector):
    """Connector for FRED API."""

    def get_quote(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="FRED", symbol=symbol, data={"value": 100.0}, timestamp=time.time()
        )

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataResponse:
        return DataResponse(
            provider="FRED", symbol=symbol, data=[], timestamp=time.time()
        )

    def get_fundamentals(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="FRED", symbol=symbol, data={}, timestamp=time.time()
        )
