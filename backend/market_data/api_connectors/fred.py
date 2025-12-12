"""
FRED (Federal Reserve Economic Data) API connector.

This module provides a connector for accessing economic data
from the Federal Reserve Economic Data (FRED) API.
"""

from datetime import date, datetime
import logging
from typing import Any, Dict, List, Optional, Union
from market_data.api_connectors.base import (
    APIConnector,
    APICredentials,
    DataCategory,
    DataFormat,
    DataProvider,
    DataResponse,
    RateLimiter,
)


class FREDConnector(APIConnector):
    """
    Connector for FRED (Federal Reserve Economic Data) API.

    This class provides methods for accessing economic data
    from the Federal Reserve Economic Data (FRED) API.

    Parameters
    ----------
    api_key : str
        FRED API key.
    """

    def __init__(self, api_key: str) -> Any:
        credentials = APICredentials(api_key=api_key)
        base_url = "https://api.stlouisfed.org/fred"
        rate_limiter = RateLimiter(requests_per_minute=120)
        super().__init__(
            credentials=credentials, base_url=base_url, rate_limiter=rate_limiter
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
        return DataProvider.FRED.value

    def authenticate(self) -> bool:
        """
        Authenticate with the FRED API.

        Returns
        -------
        success : bool
            Whether authentication was successful.
        """
        response = self.get_series("GDP")
        return response.is_success()

    def _add_api_key(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        params["file_type"] = "json"
        return params

    def get_series(
        self,
        series_id: str,
        observation_start: Optional[Union[str, date, datetime]] = None,
        observation_end: Optional[Union[str, date, datetime]] = None,
        units: Optional[str] = None,
        frequency: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        output_type: Optional[int] = None,
        vintage_dates: Optional[Union[str, List[str]]] = None,
    ) -> DataResponse:
        """
        Get series data.

        Parameters
        ----------
        series_id : str
            Series ID.
        observation_start : str or date or datetime, optional
            Start date (format: YYYY-MM-DD).
        observation_end : str or date or datetime, optional
            End date (format: YYYY-MM-DD).
        units : str, optional
            Units transformation. Options: "lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log".
        frequency : str, optional
            Frequency. Options: "d", "w", "bw", "m", "q", "sa", "a", "wef", "weth", "wew", "wetu", "wem", "wesu", "wesa", "bwew", "bwem".
        aggregation_method : str, optional
            Aggregation method. Options: "avg", "sum", "eop".
        output_type : int, optional
            Output type. Options: 1 (observations by observation date), 2 (observations by vintage date), 3 (vintage dates).
        vintage_dates : str or list, optional
            Vintage dates (format: YYYY-MM-DD).

        Returns
        -------
        response : DataResponse
            Response containing the series data.
        """
        endpoint = "series/observations"
        params = self._add_api_key({"series_id": series_id})
        if observation_start:
            if isinstance(observation_start, (date, datetime)):
                observation_start = observation_start.strftime("%Y-%m-%d")
            params["observation_start"] = observation_start
        if observation_end:
            if isinstance(observation_end, (date, datetime)):
                observation_end = observation_end.strftime("%Y-%m-%d")
            params["observation_end"] = observation_end
        if units:
            params["units"] = units
        if frequency:
            params["frequency"] = frequency
        if aggregation_method:
            params["aggregation_method"] = aggregation_method
        if output_type:
            params["output_type"] = output_type
        if vintage_dates:
            if isinstance(vintage_dates, list):
                vintage_dates = ",".join(vintage_dates)
            params["vintage_dates"] = vintage_dates
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_info(self, series_id: str) -> DataResponse:
        """
        Get series information.

        Parameters
        ----------
        series_id : str
            Series ID.

        Returns
        -------
        response : DataResponse
            Response containing the series information.
        """
        endpoint = "series"
        params = self._add_api_key({"series_id": series_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def search_series(
        self,
        search_text: str,
        search_type: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filter_variable: Optional[str] = None,
        filter_value: Optional[str] = None,
        tag_names: Optional[Union[str, List[str]]] = None,
        exclude_tag_names: Optional[Union[str, List[str]]] = None,
    ) -> DataResponse:
        """
        Search for series.

        Parameters
        ----------
        search_text : str
            Search text.
        search_type : str, optional
            Search type. Options: "full_text", "series_id", "series_id_full".
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "search_rank", "series_id", "title", "units", "frequency", "seasonal_adjustment", "realtime_start", "realtime_end", "last_updated", "observation_start", "observation_end", "popularity", "group_popularity".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        filter_variable : str, optional
            Filter variable. Options: "frequency", "units", "seasonal_adjustment".
        filter_value : str, optional
            Filter value.
        tag_names : str or list, optional
            Tag names.
        exclude_tag_names : str or list, optional
            Exclude tag names.

        Returns
        -------
        response : DataResponse
            Response containing the search results.
        """
        endpoint = "series/search"
        params = self._add_api_key(
            {"search_text": search_text, "limit": limit, "offset": offset}
        )
        if search_type:
            params["search_type"] = search_type
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if filter_variable:
            params["filter_variable"] = filter_variable
        if filter_value:
            params["filter_value"] = filter_value
        if tag_names:
            if isinstance(tag_names, list):
                tag_names = ";".join(tag_names)
            params["tag_names"] = tag_names
        if exclude_tag_names:
            if isinstance(exclude_tag_names, list):
                exclude_tag_names = ";".join(exclude_tag_names)
            params["exclude_tag_names"] = exclude_tag_names
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_categories(self, series_id: str) -> DataResponse:
        """
        Get categories for a series.

        Parameters
        ----------
        series_id : str
            Series ID.

        Returns
        -------
        response : DataResponse
            Response containing the series categories.
        """
        endpoint = "series/categories"
        params = self._add_api_key({"series_id": series_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_release(self, series_id: str) -> DataResponse:
        """
        Get release for a series.

        Parameters
        ----------
        series_id : str
            Series ID.

        Returns
        -------
        response : DataResponse
            Response containing the series release.
        """
        endpoint = "series/release"
        params = self._add_api_key({"series_id": series_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_tags(self, series_id: str) -> DataResponse:
        """
        Get tags for a series.

        Parameters
        ----------
        series_id : str
            Series ID.

        Returns
        -------
        response : DataResponse
            Response containing the series tags.
        """
        endpoint = "series/tags"
        params = self._add_api_key({"series_id": series_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_updates(
        self,
        limit: int = 1000,
        offset: int = 0,
        filter_value: Optional[str] = None,
        start_time: Optional[Union[str, datetime]] = None,
        end_time: Optional[Union[str, datetime]] = None,
    ) -> DataResponse:
        """
        Get series updates.

        Parameters
        ----------
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        filter_value : str, optional
            Filter value.
        start_time : str or datetime, optional
            Start time (format: YYYY-MM-DD HH:MM:SS).
        end_time : str or datetime, optional
            End time (format: YYYY-MM-DD HH:MM:SS).

        Returns
        -------
        response : DataResponse
            Response containing the series updates.
        """
        endpoint = "series/updates"
        params = self._add_api_key({"limit": limit, "offset": offset})
        if filter_value:
            params["filter_value"] = filter_value
        if start_time:
            if isinstance(start_time, datetime):
                start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            params["start_time"] = start_time
        if end_time:
            if isinstance(end_time, datetime):
                end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
            params["end_time"] = end_time
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_series_vintagedates(
        self,
        series_id: str,
        limit: int = 1000,
        offset: int = 0,
        sort_order: Optional[str] = None,
    ) -> DataResponse:
        """
        Get vintage dates for a series.

        Parameters
        ----------
        series_id : str
            Series ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        sort_order : str, optional
            Sort order. Options: "asc", "desc".

        Returns
        -------
        response : DataResponse
            Response containing the vintage dates.
        """
        endpoint = "series/vintagedates"
        params = self._add_api_key(
            {"series_id": series_id, "limit": limit, "offset": offset}
        )
        if sort_order:
            params["sort_order"] = sort_order
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_category(self, category_id: int) -> DataResponse:
        """
        Get category.

        Parameters
        ----------
        category_id : int
            Category ID.

        Returns
        -------
        response : DataResponse
            Response containing the category.
        """
        endpoint = "category"
        params = self._add_api_key({"category_id": category_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_category_children(self, category_id: int) -> DataResponse:
        """
        Get children of a category.

        Parameters
        ----------
        category_id : int
            Category ID.

        Returns
        -------
        response : DataResponse
            Response containing the category children.
        """
        endpoint = "category/children"
        params = self._add_api_key({"category_id": category_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_category_related(self, category_id: int) -> DataResponse:
        """
        Get related categories.

        Parameters
        ----------
        category_id : int
            Category ID.

        Returns
        -------
        response : DataResponse
            Response containing the related categories.
        """
        endpoint = "category/related"
        params = self._add_api_key({"category_id": category_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_category_series(
        self,
        category_id: int,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filter_variable: Optional[str] = None,
        filter_value: Optional[str] = None,
        tag_names: Optional[Union[str, List[str]]] = None,
        exclude_tag_names: Optional[Union[str, List[str]]] = None,
    ) -> DataResponse:
        """
        Get series in a category.

        Parameters
        ----------
        category_id : int
            Category ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "series_id", "title", "units", "frequency", "seasonal_adjustment", "realtime_start", "realtime_end", "last_updated", "observation_start", "observation_end", "popularity", "group_popularity".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        filter_variable : str, optional
            Filter variable. Options: "frequency", "units", "seasonal_adjustment".
        filter_value : str, optional
            Filter value.
        tag_names : str or list, optional
            Tag names.
        exclude_tag_names : str or list, optional
            Exclude tag names.

        Returns
        -------
        response : DataResponse
            Response containing the category series.
        """
        endpoint = "category/series"
        params = self._add_api_key(
            {"category_id": category_id, "limit": limit, "offset": offset}
        )
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if filter_variable:
            params["filter_variable"] = filter_variable
        if filter_value:
            params["filter_value"] = filter_value
        if tag_names:
            if isinstance(tag_names, list):
                tag_names = ";".join(tag_names)
            params["tag_names"] = tag_names
        if exclude_tag_names:
            if isinstance(exclude_tag_names, list):
                exclude_tag_names = ";".join(exclude_tag_names)
            params["exclude_tag_names"] = exclude_tag_names
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_category_tags(
        self,
        category_id: int,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        tag_names: Optional[Union[str, List[str]]] = None,
        tag_group_id: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> DataResponse:
        """
        Get tags for a category.

        Parameters
        ----------
        category_id : int
            Category ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "series_count", "popularity", "created", "name", "group_id".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        tag_names : str or list, optional
            Tag names.
        tag_group_id : str, optional
            Tag group ID.
        search_text : str, optional
            Search text.

        Returns
        -------
        response : DataResponse
            Response containing the category tags.
        """
        endpoint = "category/tags"
        params = self._add_api_key(
            {"category_id": category_id, "limit": limit, "offset": offset}
        )
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if tag_names:
            if isinstance(tag_names, list):
                tag_names = ";".join(tag_names)
            params["tag_names"] = tag_names
        if tag_group_id:
            params["tag_group_id"] = tag_group_id
        if search_text:
            params["search_text"] = search_text
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_releases(
        self,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> DataResponse:
        """
        Get releases.

        Parameters
        ----------
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "release_id", "name", "press_release", "realtime_start", "realtime_end".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".

        Returns
        -------
        response : DataResponse
            Response containing the releases.
        """
        endpoint = "releases"
        params = self._add_api_key({"limit": limit, "offset": offset})
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release(self, release_id: int) -> DataResponse:
        """
        Get release.

        Parameters
        ----------
        release_id : int
            Release ID.

        Returns
        -------
        response : DataResponse
            Response containing the release.
        """
        endpoint = "release"
        params = self._add_api_key({"release_id": release_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release_dates(
        self,
        release_id: int,
        limit: int = 1000,
        offset: int = 0,
        sort_order: Optional[str] = None,
        include_release_date_with_no_data: bool = False,
    ) -> DataResponse:
        """
        Get release dates.

        Parameters
        ----------
        release_id : int
            Release ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        include_release_date_with_no_data : bool, default=False
            Whether to include release dates with no data.

        Returns
        -------
        response : DataResponse
            Response containing the release dates.
        """
        endpoint = "release/dates"
        params = self._add_api_key(
            {
                "release_id": release_id,
                "limit": limit,
                "offset": offset,
                "include_release_date_with_no_data": str(
                    include_release_date_with_no_data
                ).lower(),
            }
        )
        if sort_order:
            params["sort_order"] = sort_order
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release_series(
        self,
        release_id: int,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        filter_variable: Optional[str] = None,
        filter_value: Optional[str] = None,
        tag_names: Optional[Union[str, List[str]]] = None,
        exclude_tag_names: Optional[Union[str, List[str]]] = None,
    ) -> DataResponse:
        """
        Get series in a release.

        Parameters
        ----------
        release_id : int
            Release ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "series_id", "title", "units", "frequency", "seasonal_adjustment", "realtime_start", "realtime_end", "last_updated", "observation_start", "observation_end", "popularity", "group_popularity".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        filter_variable : str, optional
            Filter variable. Options: "frequency", "units", "seasonal_adjustment".
        filter_value : str, optional
            Filter value.
        tag_names : str or list, optional
            Tag names.
        exclude_tag_names : str or list, optional
            Exclude tag names.

        Returns
        -------
        response : DataResponse
            Response containing the release series.
        """
        endpoint = "release/series"
        params = self._add_api_key(
            {"release_id": release_id, "limit": limit, "offset": offset}
        )
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if filter_variable:
            params["filter_variable"] = filter_variable
        if filter_value:
            params["filter_value"] = filter_value
        if tag_names:
            if isinstance(tag_names, list):
                tag_names = ";".join(tag_names)
            params["tag_names"] = tag_names
        if exclude_tag_names:
            if isinstance(exclude_tag_names, list):
                exclude_tag_names = ";".join(exclude_tag_names)
            params["exclude_tag_names"] = exclude_tag_names
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release_sources(self, release_id: int) -> DataResponse:
        """
        Get sources for a release.

        Parameters
        ----------
        release_id : int
            Release ID.

        Returns
        -------
        response : DataResponse
            Response containing the release sources.
        """
        endpoint = "release/sources"
        params = self._add_api_key({"release_id": release_id})
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release_tags(
        self,
        release_id: int,
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        tag_names: Optional[Union[str, List[str]]] = None,
        tag_group_id: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> DataResponse:
        """
        Get tags for a release.

        Parameters
        ----------
        release_id : int
            Release ID.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "series_count", "popularity", "created", "name", "group_id".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        tag_names : str or list, optional
            Tag names.
        tag_group_id : str, optional
            Tag group ID.
        search_text : str, optional
            Search text.

        Returns
        -------
        response : DataResponse
            Response containing the release tags.
        """
        endpoint = "release/tags"
        params = self._add_api_key(
            {"release_id": release_id, "limit": limit, "offset": offset}
        )
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if tag_names:
            if isinstance(tag_names, list):
                tag_names = ";".join(tag_names)
            params["tag_names"] = tag_names
        if tag_group_id:
            params["tag_group_id"] = tag_group_id
        if search_text:
            params["search_text"] = search_text
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )

    def get_release_related_tags(
        self,
        release_id: int,
        tag_names: Union[str, List[str]],
        limit: int = 1000,
        offset: int = 0,
        order_by: Optional[str] = None,
        sort_order: Optional[str] = None,
        tag_group_id: Optional[str] = None,
        search_text: Optional[str] = None,
        exclude_tag_names: Optional[Union[str, List[str]]] = None,
    ) -> DataResponse:
        """
        Get related tags for a release.

        Parameters
        ----------
        release_id : int
            Release ID.
        tag_names : str or list
            Tag names.
        limit : int, default=1000
            Number of results to return.
        offset : int, default=0
            Offset for pagination.
        order_by : str, optional
            Field to order by. Options: "series_count", "popularity", "created", "name", "group_id".
        sort_order : str, optional
            Sort order. Options: "asc", "desc".
        tag_group_id : str, optional
            Tag group ID.
        search_text : str, optional
            Search text.
        exclude_tag_names : str or list, optional
            Exclude tag names.

        Returns
        -------
        response : DataResponse
            Response containing the related tags.
        """
        endpoint = "release/related_tags"
        if isinstance(tag_names, list):
            tag_names = ";".join(tag_names)
        params = self._add_api_key(
            {
                "release_id": release_id,
                "tag_names": tag_names,
                "limit": limit,
                "offset": offset,
            }
        )
        if order_by:
            params["order_by"] = order_by
        if sort_order:
            params["sort_order"] = sort_order
        if tag_group_id:
            params["tag_group_id"] = tag_group_id
        if search_text:
            params["search_text"] = search_text
        if exclude_tag_names:
            if isinstance(exclude_tag_names, list):
                exclude_tag_names = ";".join(exclude_tag_names)
            params["exclude_tag_names"] = exclude_tag_names
        return self.request(
            endpoint=endpoint,
            params=params,
            category=DataCategory.ECONOMIC,
            format=DataFormat.JSON,
        )
