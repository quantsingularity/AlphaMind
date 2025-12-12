import abc
import datetime
import json
import logging
from typing import Any, Dict, Optional, Union
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("AlphaMind.Core")


class MarketData:
    """Standard container for processed market and alternative data."""

    def __init__(self, data: pd.DataFrame, source: str = "combined") -> Any:
        """
        Args:
            data: A Pandas DataFrame containing time series data (e.g., price, volume, sentiment).
            source: The source or type of the data (e.g., 'price', 'sentiment', 'geospatial').
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MarketData must be initialized with a Pandas DataFrame.")
        self.data = data
        self.source = source
        logger.debug(
            f"MarketData created from source: {source} with {len(data)} records."
        )

    def get_latest(self) -> Dict[str, Any]:
        """Returns the latest data point as a dictionary."""
        if self.data.empty:
            return {}
        return self.data.iloc[-1].to_dict()


class Signal:
    """Standard container for trading signals."""

    def __init__(
        self,
        ticker: str,
        position: int,
        confidence: float,
        timestamp: datetime.datetime,
    ) -> Any:
        """
        Args:
            ticker: The asset symbol (e.g., 'AAPL').
            position: The desired position (-1: Short, 0: Hold/Cash, 1: Long).
            confidence: A score indicating the strength of the signal (0.0 to 1.0).
            timestamp: The generation time of the signal.
        """
        self.ticker = ticker
        self.position = position
        self.confidence = confidence
        self.timestamp = timestamp

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        """Returns the signal as a dictionary for easy storage/transmission."""
        return {
            "ticker": self.ticker,
            "position": self.position,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


class ConfigManager:
    """Handles loading and accessing global configuration settings."""

    def __init__(self, config_path: str = "config.json") -> Any:
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from a JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(
                f"Configuration file not found at {self.config_path}. Using default settings."
            )
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON in {self.config_path}: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value by key, supporting nested keys (e.g., 'API.SENTINEL_KEY')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value


class BaseModule(abc.ABC):
    """
    Abstract base class for all functional modules in the AlphaMind system.
    Enforces a standard lifecycle and interface for configuration and execution.
    """

    def __init__(self, module_name: str, config_manager: ConfigManager) -> Any:
        self.module_name = module_name
        self.config = config_manager
        self.logger = logging.getLogger(f"AlphaMind.{module_name}")
        self.logger.info(f"Initializing {self.module_name}...")
        self.is_configured = False

    @abc.abstractmethod
    def configure(self) -> bool:
        """
        Performs necessary setup, loads external resources (e.g., models, databases),
        and validates configuration parameters. Must be called before run().

        Returns:
            True if configuration is successful, False otherwise.
        """
        self.is_configured = True
        self.logger.info(f"{self.module_name} configured.")
        return True

    @abc.abstractmethod
    def run(
        self, input_data: Optional[Union[MarketData, Signal]] = None
    ) -> Union[MarketData, Signal, None]:
        """
        Executes the primary logic of the module (e.g., fetch data, analyze sentiment, generate signals).

        Args:
            input_data: Data object passed from a previous module in the workflow.

        Returns:
            The processed data or generated signal object, or None on failure.
        """
        if not self.is_configured:
            self.logger.error(f"{self.module_name} must be configured before running.")
            return None
        self.logger.info(f"Running {self.module_name}.")
        return None


if __name__ == "__main__":
    temp_config = {
        "GLOBAL": {"TICKERS": ["AAPL", "MSFT"]},
        "API": {"SENTINEL_KEY": "dummy_key", "MAX_RETRIES": 5},
    }
    with open("config.json", "w") as f:
        json.dump(temp_config, f)
    config = ConfigManager()
    tickers = config.get("GLOBAL.TICKERS")
    logger.info(f"\nLoaded Tickers: {tickers}")

    class DataFetcher(BaseModule):

        def configure(self) -> Any:
            super().configure()
            self.logger.info(f"Using key: {self.config.get('API.SENTINEL_KEY', 'N/A')}")
            return True

        def run(self, input_data: Any = None) -> Any:
            super().run()
            data = pd.DataFrame(
                {"close": [100, 101, 102]},
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            )
            return MarketData(data, source="price_data")

    fetcher = DataFetcher("DataFetcher", config)
    fetcher.configure()
    data = fetcher.run()
    if data:
        logger.info(f"\nLatest Data Point: {data.get_latest()}")
    signal = Signal("AAPL", 1, 0.95, datetime.datetime.now())
    logger.info(f"Generated Signal: {signal.to_dict()}")
