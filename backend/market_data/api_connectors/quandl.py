"""
Quandl API connector for financial data.

This module provides a connector for accessing financial market data
from Quandl, including economic data, financial data, and alternative data.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
from datetime import datetime, timedelta, date

from .base import (
    APIConnector,
    APICredentials,
    DataResponse,
    DataCategory,
    DataFormat,
    RateLimiter,
    DataProvider
)


class QuandlConnector(APIConnector):
    """
    Connector for Quandl API.
    
    This class provides methods for accessing financial market data
    from Quandl, including economic data, financial data, and alternative data.
    
    Parameters
    ----------
    api_key : str
        Quandl API key.
    """
    
    def __init__(
        self,
        api_key: str
    ):
        # Create credentials
        credentials = APICredentials(api_key=api_key)
        
        # Set base URL
        base_url = "https://www.quandl.com/api/v3"
        
        # Create rate limiter
        # Quandl has a limit of 300 calls per 10 seconds, 2000 calls per 10 minutes
        rate_limiter = RateLimiter(
            requests_per_second=30
        )
        
        super().__init__(
            credentials=credentials,
            base_url=base_url,
            rate_limiter=rate_limiter
        )
        
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
        return DataProvider.QUANDL.value
    
    def authenticate(self) -> bool:
        """
        Authenticate with the Quandl API.
        
        Returns
        -------
        success : bool
            Whether authentication was successful.
        """
        # Quandl uses API key for authentication
        # Test authentication by making a simple request
        response = self.get_dataset("WIKI/AAPL", rows=1)
        
        return response.is_success()
    
    def _add_api_key(
        self,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add API key to request parameters.
        
        Parameters
        ----------
        params : dict, optional
            Request parameters.
            
        Returns
        -------
        params : dict
            Request parameters with API key added.
        """
        params = params or {}
        params["api_key"] = self.credentials.api_key
        return params
    
    def get_dataset(
        self,
        dataset_code: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        column_index: Optional[Union[int, List[int]]] = None,
        collapse: Optional[str] = None,
        transform: Optional[str] = None,
        order: str = "asc",
        rows: Optional[int] = None,
        format: str = "json"
    ) -> DataResponse:
        """
        Get dataset data.
        
        Parameters
        ----------
        dataset_code : str
            Dataset code in format "DATABASE_CODE/DATASET_CODE".
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
        column_index : int or list, optional
            Column index or list of column indices.
        collapse : str, optional
            Frequency to collapse data to. Options: "daily", "weekly", "monthly", "quarterly", "annual".
        transform : str, optional
            Transformation to apply to data. Options: "diff", "rdiff", "cumul", "normalize".
        order : str, default="asc"
            Sort order. Options: "asc", "desc".
        rows : int, optional
            Number of rows to return.
        format : str, default="json"
            Response format. Options: "json", "csv", "xml".
            
        Returns
        -------
        response : DataResponse
            Response containing the dataset data.
        """
        endpoint = f"datasets/{dataset_code}.{format}"
        
        params = self._add_api_key({
            "order": order
        })
        
        # Convert dates to strings if needed
        if start_date:
            if isinstance(start_date, (date, datetime)):
                start_date = start_date.strftime("%Y-%m-%d")
            params["start_date"] = start_date
        
        if end_date:
            if isinstance(end_date, (date, datetime)):
                end_date = end_date.strftime("%Y-%m-%d")
            params["end_date"] = end_date
        
        if column_index is not None:
            if isinstance(column_index, list):
                params["column_index"] = ",".join(map(str, column_index))
            else:
                params["column_index"] = str(column_index)
        
        if collapse:
            params["collapse"] = collapse
        
        if transform:
            params["transform"] = transform
        
        if rows:
            params["rows"] = rows
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if format == "json" else (DataFormat.CSV if format == "csv" else DataFormat.XML)
        )
    
    def get_dataset_metadata(
        self,
        dataset_code: str
    ) -> DataResponse:
        """
        Get dataset metadata.
        
        Parameters
        ----------
        dataset_code : str
            Dataset code in format "DATABASE_CODE/DATASET_CODE".
            
        Returns
        -------
        response : DataResponse
            Response containing the dataset metadata.
        """
        endpoint = f"datasets/{dataset_code}/metadata"
        params = self._add_api_key()
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON
        )
    
    def get_database_metadata(
        self,
        database_code: str
    ) -> DataResponse:
        """
        Get database metadata.
        
        Parameters
        ----------
        database_code : str
            Database code.
            
        Returns
        -------
        response : DataResponse
            Response containing the database metadata.
        """
        endpoint = f"databases/{database_code}"
        params = self._add_api_key()
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON
        )
    
    def search_datasets(
        self,
        query: str,
        database_code: Optional[str] = None,
        per_page: int = 100,
        page: int = 1
    ) -> DataResponse:
        """
        Search for datasets.
        
        Parameters
        ----------
        query : str
            Search query.
        database_code : str, optional
            Database code to filter by.
        per_page : int, default=100
            Number of results per page.
        page : int, default=1
            Page number.
            
        Returns
        -------
        response : DataResponse
            Response containing the search results.
        """
        endpoint = "datasets"
        
        params = self._add_api_key({
            "query": query,
            "per_page": per_page,
            "page": page
        })
        
        if database_code:
            params["database_code"] = database_code
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON
        )
    
    def list_databases(
        self,
        per_page: int = 100,
        page: int = 1
    ) -> DataResponse:
        """
        List databases.
        
        Parameters
        ----------
        per_page : int, default=100
            Number of results per page.
        page : int, default=1
            Page number.
            
        Returns
        -------
        response : DataResponse
            Response containing the list of databases.
        """
        endpoint = "databases"
        
        params = self._add_api_key({
            "per_page": per_page,
            "page": page
        })
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON
        )
    
    def get_datatable(
        self,
        datatable_code: str,
        format: str = "json"
    ) -> DataResponse:
        """
        Get datatable.
        
        Parameters
        ----------
        datatable_code : str
            Datatable code.
        format : str, default="json"
            Response format. Options: "json", "csv".
            
        Returns
        -------
        response : DataResponse
            Response containing the datatable.
        """
        endpoint = f"datatables/{datatable_code}.{format}"
        params = self._add_api_key()
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if format == "json" else DataFormat.CSV
        )
    
    def get_datatable_with_filters(
        self,
        datatable_code: str,
        filters: Dict[str, Any],
        format: str = "json"
    ) -> DataResponse:
        """
        Get datatable with filters.
        
        Parameters
        ----------
        datatable_code : str
            Datatable code.
        filters : dict
            Filters to apply to the datatable.
        format : str, default="json"
            Response format. Options: "json", "csv".
            
        Returns
        -------
        response : DataResponse
            Response containing the filtered datatable.
        """
        endpoint = f"datatables/{datatable_code}.{format}"
        
        params = self._add_api_key()
        
        # Add filters to params
        for key, value in filters.items():
            if isinstance(value, list):
                params[key] = ",".join(map(str, value))
            else:
                params[key] = value
        
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.MARKET_DATA,
            format=DataFormat.JSON if format == "json" else DataFormat.CSV
        )
    
    def get_time_series(
        self,
        database_code: str,
        dataset_code: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        collapse: Optional[str] = None,
        transform: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get time series data as a pandas DataFrame.
        
        Parameters
        ----------
        database_code : str
            Database code.
        dataset_code : str
            Dataset code.
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
        collapse : str, optional
            Frequency to collapse data to. Options: "daily", "weekly", "monthly", "quarterly", "annual".
        transform : str, optional
            Transformation to apply to data. Options: "diff", "rdiff", "cumul", "normalize".
        limit : int, optional
            Number of rows to return.
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the time series data.
        """
        dataset_code_full = f"{database_code}/{dataset_code}"
        
        response = self.get_dataset(
            dataset_code=dataset_code_full,
            start_date=start_date,
            end_date=end_date,
            collapse=collapse,
            transform=transform,
            rows=limit
        )
        
        if not response.is_success():
            self.logger.error(f"Failed to get time series data: {response.error}")
            return pd.DataFrame()
        
        # Extract data from response
        dataset = response.data.get("dataset", {})
        data = dataset.get("data", [])
        column_names = dataset.get("column_names", [])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert date column to datetime
        if len(column_names) > 0 and column_names[0].lower() == "date":
            df[column_names[0]] = pd.to_datetime(df[column_names[0]])
        
        return df
    
    def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        collapse: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get stock data as a pandas DataFrame.
        
        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
        collapse : str, optional
            Frequency to collapse data to. Options: "daily", "weekly", "monthly", "quarterly", "annual".
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the stock data.
        """
        # Try to get data from WIKI database (historical)
        try:
            df = self.get_time_series(
                database_code="WIKI",
                dataset_code=ticker,
                start_date=start_date,
                end_date=end_date,
                collapse=collapse
            )
            
            if not df.empty:
                return df
        except Exception as e:
            self.logger.warning(f"Failed to get stock data from WIKI database: {e}")
        
        # Try to get data from EOD database
        try:
            df = self.get_time_series(
                database_code="EOD",
                dataset_code=ticker,
                start_date=start_date,
                end_date=end_date,
                collapse=collapse
            )
            
            if not df.empty:
                return df
        except Exception as e:
            self.logger.warning(f"Failed to get stock data from EOD database: {e}")
        
        # Try to get data from EURONEXT database
        try:
            df = self.get_time_series(
                database_code="EURONEXT",
                dataset_code=ticker,
                start_date=start_date,
                end_date=end_date,
                collapse=collapse
            )
            
            if not df.empty:
                return df
        except Exception as e:
            self.logger.warning(f"Failed to get stock data from EURONEXT database: {e}")
        
        # Return empty DataFrame if all attempts fail
        self.logger.error(f"Failed to get stock data for {ticker}")
        return pd.DataFrame()
    
    def get_economic_data(
        self,
        indicator_code: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get economic data as a pandas DataFrame.
        
        Parameters
        ----------
        indicator_code : str
            Indicator code in format "DATABASE_CODE/DATASET_CODE".
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the economic data.
        """
        response = self.get_dataset(
            dataset_code=indicator_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if not response.is_success():
            self.logger.error(f"Failed to get economic data: {response.error}")
            return pd.DataFrame()
        
        # Extract data from response
        dataset = response.data.get("dataset", {})
        data = dataset.get("data", [])
        column_names = dataset.get("column_names", [])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert date column to datetime
        if len(column_names) > 0 and column_names[0].lower() == "date":
            df[column_names[0]] = pd.to_datetime(df[column_names[0]])
        
        return df
    
    def get_futures_data(
        self,
        symbol: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get futures data as a pandas DataFrame.
        
        Parameters
        ----------
        symbol : str
            Futures symbol in format "DATABASE_CODE/DATASET_CODE".
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the futures data.
        """
        response = self.get_dataset(
            dataset_code=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if not response.is_success():
            self.logger.error(f"Failed to get futures data: {response.error}")
            return pd.DataFrame()
        
        # Extract data from response
        dataset = response.data.get("dataset", {})
        data = dataset.get("data", [])
        column_names = dataset.get("column_names", [])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert date column to datetime
        if len(column_names) > 0 and column_names[0].lower() == "date":
            df[column_names[0]] = pd.to_datetime(df[column_names[0]])
        
        return df
    
    def get_forex_data(
        self,
        currency_pair: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get forex data as a pandas DataFrame.
        
        Parameters
        ----------
        currency_pair : str
            Currency pair in format "FROM/TO".
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the forex data.
        """
        # Try to get data from CURRFX database
        try:
            dataset_code = f"CURRFX/{currency_pair}"
            
            response = self.get_dataset(
                dataset_code=dataset_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if response.is_success():
                # Extract data from response
                dataset = response.data.get("dataset", {})
                data = dataset.get("data", [])
                column_names = dataset.get("column_names", [])
                
                # Create DataFrame
                df = pd.DataFrame(data, columns=column_names)
                
                # Convert date column to datetime
                if len(column_names) > 0 and column_names[0].lower() == "date":
                    df[column_names[0]] = pd.to_datetime(df[column_names[0]])
                
                return df
        except Exception as e:
            self.logger.warning(f"Failed to get forex data from CURRFX database: {e}")
        
        # Return empty DataFrame if all attempts fail
        self.logger.error(f"Failed to get forex data for {currency_pair}")
        return pd.DataFrame()
    
    def get_commodity_data(
        self,
        commodity_code: str,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None
    ) -> pd.DataFrame:
        """
        Get commodity data as a pandas DataFrame.
        
        Parameters
        ----------
        commodity_code : str
            Commodity code in format "DATABASE_CODE/DATASET_CODE".
        start_date : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        end_date : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
            
        Returns
        -------
        df : DataFrame
            DataFrame containing the commodity data.
        """
        response = self.get_dataset(
            dataset_code=commodity_code,
            start_date=start_date,
            end_date=end_date
        )
        
        if not response.is_success():
            self.logger.error(f"Failed to get commodity data: {response.error}")
            return pd.DataFrame()
        
        # Extract data from response
        dataset = response.data.get("dataset", {})
        data = dataset.get("data", [])
        column_names = dataset.get("column_names", [])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert date column to datetime
        if len(column_names) > 0 and column_names[0].lower() == "date":
            df[column_names[0]] = pd.to_datetime(df[column_names[0]])
        
        return df
