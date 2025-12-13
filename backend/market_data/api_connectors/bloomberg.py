"""Bloomberg API connector."""

from market_data.api_connectors.base import APIConnector, DataResponse
import time


class BloombergConnector(APIConnector):
    """Connector for Bloomberg API."""

    def get_quote(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="Bloomberg",
            symbol=symbol,
            data={"price": 100.0},
            timestamp=time.time(),
        )

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataResponse:
        return DataResponse(
            provider="Bloomberg", symbol=symbol, data=[], timestamp=time.time()
        )

    def get_fundamentals(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="Bloomberg", symbol=symbol, data={}, timestamp=time.time()
        )
