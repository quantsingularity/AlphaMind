import os
import sys
from typing import Any

import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODELS_PATH = os.path.join(BASE_DIR, "..")
if AI_MODELS_PATH not in sys.path:
    sys.path.insert(0, AI_MODELS_PATH)


@pytest.fixture
def sample_market_data() -> Any:
    """Fixture providing sample time series data for AI model testing."""
    import numpy as np

    return {
        "prices": np.random.randn(100, 4),
        "timestamps": list(range(100)),
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    }
