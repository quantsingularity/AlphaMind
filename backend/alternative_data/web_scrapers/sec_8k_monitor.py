# """"""
## SEC 8-K Filing Monitor for AlphaMind
#
## This module monitors and processes SEC 8-K filings for specified tickers,
## extracting relevant information and performing sentiment analysis.
# """"""

# from datetime import datetime
# import logging
# import re
# import time
# from typing import Any, Dict, List, Optional, Union

# import requests
# from sec_edgar_downloader import Downloader

# Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


# class SEC8KMonitor:
#    """Monitor and process SEC 8-K filings."""
#
##     def __init__(self, tickers: List[str], max_retries: int = 3, retry_delay: int = 5):
#        """"""
#         Initialize SEC 8-K monitor.

#         Args:
#             tickers: List of ticker symbols to monitor
#             max_retries: Maximum number of retry attempts for failed operations
#             retry_delay: Delay between retry attempts in seconds
#        """"""
#        # Updated to provide both company_name and email_address as required by SEC Edgar Downloader
##         self.dl = Downloader(
##             company_name="AlphaMind", email_address="alphamind@example.com"
#        )
##         self.ticker_pattern = re.compile(r"\b(?:" + "|".join(tickers) + r")\b", re.I)
##         self.max_retries = max_retries
##         self.retry_delay = retry_delay
##         self.sentiment_model = None
#
##         logger.info(f"Initialized SEC 8-K monitor for {len(tickers)} tickers")
#
##     def download_filings(
##         self, ticker: str, filing_type: str = "8-K", count: int = 10
##     ) -> bool:
#        """"""
#         Download SEC filings for a ticker with retry logic.

#         Args:
#             ticker: Ticker symbol
#             filing_type: SEC filing type
#             count: Number of filings to download

#         Returns:
#             True if download successful, False otherwise
#        """"""
##         attempts = 0
##         while attempts < self.max_retries:
##             try:
##                 logger.info(f"Downloading {filing_type} filings for {ticker}")
##                 self.dl.get(filing_type, ticker, count)
##                 logger.info(
##                     f"Successfully downloaded {filing_type} filings for {ticker}"
#                )
##                 return True
##             except Exception as e:
##                 attempts += 1
##                 logger.warning(
##                     f"Attempt {attempts}/{self.max_retries} failed for {ticker}: {e}"
#                )
##                 if attempts < self.max_retries:
##                     logger.info(f"Retrying in {self.retry_delay} seconds...")
##                     time.sleep(self.retry_delay)
##                 else:
##                     logger.error(
##                         f"Failed to download {filing_type} filings for {ticker} after {self.max_retries} attempts"
#                    )
##                     return False
#
##     def process_filing(self, filing) -> Dict[str, Any]:
#        """"""
#         Process SEC filing with data validation.

#         Args:
#             filing: SEC filing object

#         Returns:
#             Processed filing data
#        """"""
##         try:
#            # Validate filing
##             if not self._validate_filing(filing):
##                 logger.warning(f"Invalid filing: {filing}")
##                 return {
#                    "filing_date": datetime.now().isoformat(),
#                    "mentioned_tickers": [],
#                    "sentiment": "unknown",
#                    "valid": False,
#                    "error": "Invalid filing format",
#                }
#
#            # Clean and process text
##             text = self._clean_html(filing.text)
##             matches = self.ticker_pattern.findall(text)
#
#            # Calculate sentiment with retry logic
##             sentiment = self._calculate_sentiment_with_retry(text)
#
##             result = {
#                "filing_date": filing.date,
#                "mentioned_tickers": list(set(matches)),
#                "sentiment": sentiment,
#                "valid": True,
#                "processed_at": datetime.now().isoformat(),
#            }
#
##             logger.info(
##                 f"Processed filing from {filing.date} with sentiment: {sentiment}"
#            )
##             return result
#
##         except Exception as e:
##             logger.error(f"Error processing filing: {e}")
##             return {
#                "filing_date": getattr(filing, "date", datetime.now().isoformat()),
#                "mentioned_tickers": [],
#                "sentiment": "unknown",
#                "valid": False,
#                "error": str(e),
#            }
#
##     def _validate_filing(self, filing) -> bool:
#        """"""
#         Validate filing data.

#         Args:
#             filing: SEC filing object

#         Returns:
#             True if filing is valid, False otherwise
#        """"""
#        # Check if filing has required attributes
##         if not hasattr(filing, "text") or not hasattr(filing, "date"):
##             logger.warning("Filing missing required attributes")
##             return False
#
#        # Check if text is not empty
##         if not filing.text or not isinstance(filing.text, str):
##             logger.warning("Filing text is empty or not a string")
##             return False
#
#        # Check if date is valid
##         try:
##             if isinstance(filing.date, str):
##                 datetime.fromisoformat(filing.date.replace("Z", "+00:00"))
##         except ValueError:
##             logger.warning(f"Invalid filing date format: {filing.date}")
##             return False
#
##         return True
#
##     def _clean_html(self, html: str) -> str:
#        """"""
#         Clean HTML content from filing.

#         Args:
#             html: HTML content

#         Returns:
#             Cleaned text
#        """"""
##         try:
#            # Simple HTML tag removal
##             text = re.sub(r"<[^>]+>", " ", html)
#            # Remove extra whitespace
##             text = re.sub(r"\s+", " ", text).strip()
##             return text
##         except Exception as e:
##             logger.error(f"Error cleaning HTML: {e}")
##             return html
#
##     def _calculate_sentiment_with_retry(self, text: str) -> str:
#        """"""
#         Calculate sentiment with retry logic.

#         Args:
#             text: Text to analyze

#         Returns:
#             Sentiment label
#        """"""
##         attempts = 0
##         while attempts < self.max_retries:
##             try:
##                 return self._calculate_sentiment(text)
##             except Exception as e:
##                 attempts += 1
##                 logger.warning(
##                     f"Sentiment analysis attempt {attempts}/{self.max_retries} failed: {e}"
#                )
##                 if attempts < self.max_retries:
##                     logger.info(
##                         f"Retrying sentiment analysis in {self.retry_delay} seconds..."
#                    )
##                     time.sleep(self.retry_delay)
##                 else:
##                     logger.error(
##                         f"Failed to calculate sentiment after {self.max_retries} attempts"
#                    )
##                     return "unknown"
#
##     def _calculate_sentiment(self, text: str) -> str:
#        """"""
#         Calculate sentiment of text.

#         Args:
#             text: Text to analyze

#         Returns:
#             Sentiment label
#        """"""
#        # Validate input
##         if not text or not isinstance(text, str):
##             logger.warning("Invalid text for sentiment analysis")
##             return "unknown"
#
#        # Truncate text if too long
##         if len(text) > 512:
##             logger.debug(
##                 f"Truncating text from {len(text)} to 512 characters for sentiment analysis"
#            )
##             text = text[:512]
#
#        # Load model if not already loaded
##         if self.sentiment_model is None:
##             try:
##                 from transformers import pipeline
#
##                 logger.info("Loading FinBERT sentiment model")
##                 self.sentiment_model = pipeline(
#                    "text-classification", model="ProsusAI/finbert"
#                )
##                 logger.info("FinBERT model loaded successfully")
##             except Exception as e:
##                 logger.error(f"Failed to load sentiment model: {e}")
##                 return "unknown"
#
#        # Perform sentiment analysis
##         try:
##             result = self.sentiment_model(text)[0]
##             logger.debug(f"Sentiment analysis result: {result}")
##             return result["label"]
##         except Exception as e:
##             logger.error(f"Error in sentiment analysis: {e}")
##             return "unknown"
