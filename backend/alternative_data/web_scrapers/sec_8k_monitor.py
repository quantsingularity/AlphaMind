"""
SEC 8-K Filing Monitor for AlphaMind

This module monitors and processes SEC 8-K filings for specified tickers,
extracting relevant information and performing sentiment analysis.
"""

from datetime import datetime
import logging
import re
import time
from typing import Any, Dict, List

try:
    from sec_edgar_downloader import Downloader
    from transformers import pipeline as transformer_pipeline
except ImportError:

    class Downloader:

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "sec_edgar_downloader is required. Install with: pip install sec-edgar-downloader"
            )

        def get(self, *args, **kwargs) -> Any:
            raise NotImplementedError("This method is not yet implemented")

    def transformer_pipeline(*args, **kwargs) -> Any:
        """Stub for transformers pipeline"""

        class StubPipeline:
            def __call__(self, *args, **kwargs) -> List[Dict[str, Any]]:
                return [{"label": "unknown", "score": 0.0}]

        return StubPipeline()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SEC8KMonitor:
    """Monitor and process SEC 8-K filings."""

    def __init__(
        self, tickers: List[str], max_retries: int = 3, retry_delay: int = 5
    ) -> None:
        """
        Initialize SEC 8-K monitor.

        Args:
            tickers: List of ticker symbols to monitor
            max_retries: Maximum number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
        """
        self.dl = Downloader(
            company_name="AlphaMind", email_address="alphamind@example.com"
        )
        self.ticker_pattern = re.compile("\\b(?:" + "|".join(tickers) + ")\\b", re.I)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sentiment_model = None
        logger.info(f"Initialized SEC 8-K monitor for {len(tickers)} tickers")

    def download_filings(
        self, ticker: str, filing_type: str = "8-K", count: int = 10
    ) -> bool:
        """
        Download SEC filings for a ticker with retry logic.

        Args:
            ticker: Ticker symbol
            filing_type: SEC filing type ('8-K', '10-Q', etc.)
            count: Number of filings to download

        Returns:
            True if download successful, False otherwise
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                logger.info(f"Downloading {filing_type} filings for {ticker}")
                self.dl.get(filing_type, ticker, count)
                logger.info(
                    f"Successfully downloaded {filing_type} filings for {ticker}"
                )
                return True
            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Attempt {attempts}/{self.max_retries} failed for {ticker}: {e}"
                )
                if attempts < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to download {filing_type} filings for {ticker} after {self.max_retries} attempts"
                    )
                    return False
        return False

    def process_filing(self, filing: Any) -> Dict[str, Any]:
        """
        Process SEC filing with data validation.

        Args:
            filing: SEC filing object (should have 'text' and 'date' attributes)

        Returns:
            Processed filing data dictionary.
        """
        try:
            if not self._validate_filing(filing):
                logger.warning(f"Invalid filing encountered.")
                return {
                    "filing_date": getattr(filing, "date", datetime.now().isoformat()),
                    "mentioned_tickers": [],
                    "sentiment": "unknown",
                    "valid": False,
                    "error": "Invalid filing format or missing data",
                }
            text = self._clean_html(filing.text)
            matches = self.ticker_pattern.findall(text)
            sentiment = self._calculate_sentiment_with_retry(text)
            result = {
                "filing_date": filing.date,
                "mentioned_tickers": list(set(matches)),
                "sentiment": sentiment,
                "valid": True,
                "processed_at": datetime.now().isoformat(),
            }
            logger.info(
                f"Processed filing from {filing.date} with sentiment: {sentiment}. Tickers: {result['mentioned_tickers']}"
            )
            return result
        except Exception as e:
            logger.error(f"Critical error processing filing: {e}")
            return {
                "filing_date": getattr(filing, "date", datetime.now().isoformat()),
                "mentioned_tickers": [],
                "sentiment": "unknown",
                "valid": False,
                "error": str(e),
            }

    def _validate_filing(self, filing: Any) -> bool:
        """
        Validate filing data structure and content presence.

        Args:
            filing: SEC filing object

        Returns:
            True if filing is valid, False otherwise
        """
        if not hasattr(filing, "text") or not hasattr(filing, "date"):
            logger.warning("Filing missing required attributes (text or date)")
            return False
        if (
            not filing.text
            or not isinstance(filing.text, str)
            or len(filing.text.strip()) < 50
        ):
            logger.warning("Filing text is empty, not a string, or too short")
            return False
        try:
            if isinstance(filing.date, str):
                datetime.fromisoformat(filing.date.replace("Z", "+00:00"))
        except ValueError:
            logger.warning(f"Invalid filing date format: {filing.date}")
            return False
        return True

    def _clean_html(self, html: str) -> str:
        """
        Clean HTML content from filing by removing tags and normalizing whitespace.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned plain text
        """
        try:
            text = re.sub("<[^>]+>", " ", html)
            text = re.sub("\\s+", " ", text).strip()
            return text
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return html

    def _calculate_sentiment_with_retry(self, text: str) -> str:
        """
        Calculate sentiment with retry logic for API/Model reliability.

        Args:
            text: Text to analyze

        Returns:
            Sentiment label ('positive', 'negative', 'neutral', or 'unknown')
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                return self._calculate_sentiment(text)
            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Sentiment analysis attempt {attempts}/{self.max_retries} failed: {e}"
                )
                if attempts < self.max_retries:
                    logger.info(
                        f"Retrying sentiment analysis in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to calculate sentiment after {self.max_retries} attempts"
                    )
                    return "unknown"

    def _calculate_sentiment(self, text: str) -> str:
        """
        Calculate sentiment of text using the FinBERT model.

        Args:
            text: Text to analyze

        Returns:
            Sentiment label
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for sentiment analysis")
            return "unknown"
        MAX_LEN = 512
        if len(text) > MAX_LEN:
            logger.debug(
                f"Truncating text from {len(text)} to {MAX_LEN} characters for sentiment analysis"
            )
            text = text[:MAX_LEN]
        if self.sentiment_model is None:
            try:
                logger.info("Loading FinBERT sentiment model (ProsusAI/finbert)")
                self.sentiment_model = transformer_pipeline(
                    "text-classification", model="ProsusAI/finbert"
                )
                logger.info("FinBERT model loaded successfully")
            except Exception as e:
                logger.error(
                    f"Failed to load sentiment model: {e}. Check 'transformers' and 'torch' installation."
                )
                return "unknown"
        try:
            result = self.sentiment_model(text)[0]
            logger.debug(f"Sentiment analysis result: {result}")
            return result["label"]
        except Exception as e:
            logger.error(f"Error in sentiment analysis prediction: {e}")
            return "unknown"
