"""IEX Cloud API connector."""

import time

from market_data.api_connectors.base import APIConnector, DataResponse


class IEXCloudConnector(APIConnector):
    """Connector for IEX Cloud API."""

    def get_quote(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="IEXCloud",
            symbol=symbol,
            data={"price": 100.0},
            timestamp=time.time(),
        )

    def get_historical_data(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> DataResponse:
        return DataResponse(
            provider="IEXCloud", symbol=symbol, data=[], timestamp=time.time()
        )

    def get_fundamentals(self, symbol: str) -> DataResponse:
        return DataResponse(
            provider="IEXCloud", symbol=symbol, data={}, timestamp=time.time()
        )
