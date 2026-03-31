"""
Base Market Connectivity Module.

Provides base classes and data structures for market venue connectivity.
"""

import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Status of a market venue connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class VenueConfig:
    """Configuration for a market venue connection."""

    venue_id: str
    name: str
    host: str
    port: int
    enabled: bool = True
    timeout: float = 30.0
    max_reconnects: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataUpdate:
    """Market data update from a venue."""

    venue_id: str
    instrument_id: str
    timestamp: datetime.datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    depth: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VenueAdapter:
    """Base adapter for connecting to a market venue."""

    def __init__(self, config: VenueConfig) -> None:
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.connected_at: Optional[datetime.datetime] = None
        self.last_error: Optional[str] = None

    def _connect_impl(self) -> bool:
        """Override in subclasses to implement the actual connection logic."""
        logger.info(f"Base connect_impl called for {self.config.venue_id}")
        return True

    def connect(self) -> bool:
        """Connect to the venue."""
        self.status = ConnectionStatus.CONNECTING
        try:
            success = self._connect_impl()
            if success:
                self.status = ConnectionStatus.CONNECTED
                self.connected_at = datetime.datetime.now()
            else:
                self.status = ConnectionStatus.ERROR
            return success
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Connection error for {self.config.venue_id}: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from the venue."""
        self.status = ConnectionStatus.DISCONNECTED
        self.connected_at = None
        logger.info(f"Disconnected from {self.config.venue_id}")
        return True

    def submit_order(self, order_data: Dict) -> Tuple[bool, str, Dict]:
        """Submit an order to the venue. Override in subclasses."""
        raise NotImplementedError("submit_order must be implemented by subclass")

    def get_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "venue_id": self.config.venue_id,
            "name": self.config.name,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "connected_at": (
                self.connected_at.isoformat() if self.connected_at else None
            ),
            "last_error": self.last_error,
        }


class MarketConnectivityManager:
    """Manages connections to multiple market venues."""

    def __init__(self) -> None:
        self.venues: Dict[str, VenueAdapter] = {}

    def add_venue(self, config: VenueConfig, adapter_class: Any = VenueAdapter) -> None:
        """Add a new venue."""
        adapter = adapter_class(config)
        self.venues[config.venue_id] = adapter
        logger.info(f"Added venue {config.venue_id}")

    def connect_venue(self, venue_id: str) -> bool:
        """Connect to a specific venue."""
        if venue_id not in self.venues:
            logger.warning(f"Venue {venue_id} not found")
            return False
        return self.venues[venue_id].connect()

    def disconnect_venue(self, venue_id: str) -> bool:
        """Disconnect from a specific venue."""
        if venue_id not in self.venues:
            logger.warning(f"Venue {venue_id} not found")
            return False
        return self.venues[venue_id].disconnect()

    def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled venues."""
        results: Dict[str, bool] = {}
        for venue_id, venue in self.venues.items():
            if venue.config.enabled:
                results[venue_id] = venue.connect()
        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all venues."""
        results: Dict[str, bool] = {}
        for venue_id, venue in self.venues.items():
            results[venue_id] = venue.disconnect()
        return results

    def get_venue_status(self, venue_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific venue."""
        if venue_id not in self.venues:
            return None
        return self.venues[venue_id].get_status()

    def get_all_venue_statuses(self) -> Dict[str, Any]:
        """Get status of all venues."""
        return {venue_id: venue.get_status() for venue_id, venue in self.venues.items()}
