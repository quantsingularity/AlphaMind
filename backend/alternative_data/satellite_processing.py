import numpy as np
import pandas as pd
from sentinelhub import WmsRequest, DataCollection, MimeType, SHConfig, CRS
from tensorflow.keras.models import load_model
from typing import List, Dict, Union, Any
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SatelliteFeatureExtractor:
    """
    Extracts physical indicators (e.g., parking lot occupancy) from
    Sentinel satellite imagery and converts them into financial time series.
    """

    def __init__(self, client_id: str, client_secret: str, instance_id: str) -> Any:
        """
        Initializes the Sentinel Hub configuration and API request parameters.

        Args:
            client_id: Sentinel Hub OAuth Client ID.
            client_secret: Sentinel Hub OAuth Client Secret.
            instance_id: Sentinel Hub Instance ID for OGC services.
        """
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        self.config.instance_id = instance_id
        logger.info("Sentinel Hub configuration loaded.")
        try:
            self.model = load_model("efficientnet_parking.h5")
            logger.info("Occupancy detection model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load CNN model: {e}. Occupancy detection will fail."
            )
            self.model = None

    def process_geospatial(
        self, coordinates: List[float], time_range: str
    ) -> pd.Series:
        """
        Extract parking lot occupancy time series from satellite images for a single BBOX.

        Args:
            coordinates: Bounding box [min_lon, min_lat, max_lon, max_lat].
            time_range: Time interval e.g., "YYYY-MM-DD/YYYY-MM-DD" or "YYYY-MM/YYYY-MM".

        Returns:
            A Pandas Series of occupancy metrics over the time range.
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot process geospatial data.")
            return pd.Series()
        bbox = BBox(coordinates, crs=CRS.WGS84)
        wms_request = WmsRequest(
            data_collection=DataCollection.SENTINEL2_L2A,
            layer=f"TRUE-COLOR-S2L2A",
            bbox=bbox,
            time=time_range,
            width=512,
            height=512,
            image_format=MimeType.TIFF,
            config=self.config,
        )
        try:
            logger.info(
                f"Fetching satellite images for BBOX: {coordinates} in range {time_range}..."
            )
            images = wms_request.get_data()
            dates = wms_request.get_dates()
            logger.info(f"Successfully fetched {len(images)} images.")
        except Exception as e:
            logger.error(f"Sentinel Hub request failed: {e}")
            return pd.Series()
        processed_images = self._preprocess_images(images)
        occupancy_scores = self.model.predict(processed_images).flatten()
        time_series = pd.Series(occupancy_scores, index=dates)
        return time_series

    def create_timeseries(self, ticker: str) -> pd.DataFrame:
        """
        Aggregates physical indicators across all company facilities to form a single time series.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA').

        Returns:
            A Pandas DataFrame where columns are facility names and values are occupancy metrics.
        """
        facilities = self._get_company_facilities(ticker)
        if not facilities:
            logger.warning(f"No facilities found for ticker {ticker}.")
            return pd.DataFrame()
        all_timeseries = {}
        for facil in facilities:
            time_range = "2020-01-01/2023-06-30"
            ts = self.process_geospatial(facil["coordinates"], time_range)
            if not ts.empty:
                all_timeseries[facil["name"]] = ts
        final_df = pd.DataFrame(all_timeseries)
        if final_df.empty:
            logger.info(f"No valid time series data was generated for {ticker}.")
        else:
            logger.info(
                f"Successfully created time series for {ticker} across {len(facilities)} facilities."
            )
        return final_df

    def _preprocess_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess satellite images for model input (normalize and resize/reshape).

        Args:
            images: A list of raw image NumPy arrays from WmsRequest.

        Returns:
            A single NumPy array ready for model prediction (Batch_Size, Height, Width, Channels).
        """
        processed_images = []
        for img in images:
            normalized = img.astype(np.float32) / 255.0
            processed_images.append(normalized)
        return np.array(processed_images)

    def _get_company_facilities(
        self, ticker: str
    ) -> List[Dict[str, Union[str, List[float]]]]:
        """
        Get company facility locations from database.

        NOTE: In a real system, this would query a dedicated geospatial database (e.g., PostGIS)
        or an alternative data source (e.g., OpenStreetMap, commercial POI data).
        The coordinates are provided as a bounding box: [min_lon, min_lat, max_lon, max_lat].

        Returns:
            List of facility dictionaries.
        """
        if ticker == "AAPL":
            return [
                {
                    "name": "Apple Park",
                    "coordinates": [-122.009, 37.3346, -121.988, 37.3456],
                },
                {
                    "name": "Foxconn Zhengzhou",
                    "coordinates": [113.625, 34.75, 113.655, 34.77],
                },
            ]
        elif ticker == "TSLA":
            return [
                {
                    "name": "Fremont Factory",
                    "coordinates": [-121.9465, 37.492, -121.9365, 37.497],
                },
                {
                    "name": "Gigafactory Nevada",
                    "coordinates": [-119.441, 39.538, -119.431, 39.548],
                },
            ]
        else:
            return []
