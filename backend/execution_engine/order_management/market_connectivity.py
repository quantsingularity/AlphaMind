"""
Market Connectivity Module.

This module provides adapters for connecting to various market venues,
handling order routing, and processing market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import datetime
import uuid
from enum import Enum
import json
import threading
import time
import queue

# Configure logging
logger = logging.getLogger(__name__)

class VenueType(Enum):
    """Types of market venues."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "ecn"
    BROKER = "broker"
    MARKET_MAKER = "market_maker"
    INTERNAL = "internal"


class ConnectionStatus(Enum):
    """Status of market venue connections."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class VenueConfig:
    """Configuration for a market venue connection."""
    venue_id: str
    venue_type: VenueType
    name: str
    priority: int = 0
    enabled: bool = True
    connection_params: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
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
    depth: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VenueAdapter:
    """Base class for market venue adapters."""
    
    def __init__(self, config: VenueConfig):
        """Initialize venue adapter.
        
        Args:
            config: Venue configuration
        """
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.last_error = None
        self.connected_at = None
        self.order_callbacks = {}
        self.market_data_callbacks = {}
        
    def connect(self) -> bool:
        """Connect to the venue.
        
        Returns:
            True if connection was successful, False otherwise
        """
        self.status = ConnectionStatus.CONNECTING
        logger.info(f"Connecting to venue {self.config.venue_id} ({self.config.name})")
        
        try:
            # Implement venue-specific connection logic in subclasses
            success = self._connect_impl()
            
            if success:
                self.status = ConnectionStatus.CONNECTED
                self.connected_at = datetime.datetime.now()
                logger.info(f"Connected to venue {self.config.venue_id}")
            else:
                self.status = ConnectionStatus.ERROR
                logger.error(f"Failed to connect to venue {self.config.venue_id}")
                
            return success
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.last_error = str(e)
            logger.error(f"Error connecting to venue {self.config.venue_id}: {str(e)}")
            return False
            
    def _connect_impl(self) -> bool:
        """Implement venue-specific connection logic.
        
        Returns:
            True if connection was successful, False otherwise
        """
        # Default implementation for testing
        return True
        
    def disconnect(self) -> bool:
        """Disconnect from the venue.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.warning(f"Venue {self.config.venue_id} is not connected")
            return True
            
        try:
            # Implement venue-specific disconnection logic in subclasses
            success = self._disconnect_impl()
            
            if success:
                self.status = ConnectionStatus.DISCONNECTED
                logger.info(f"Disconnected from venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to disconnect from venue {self.config.venue_id}")
                
            return success
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error disconnecting from venue {self.config.venue_id}: {str(e)}")
            return False
            
    def _disconnect_impl(self) -> bool:
        """Implement venue-specific disconnection logic.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        # Default implementation for testing
        return True
        
    def submit_order(self, order_data: Dict) -> Tuple[bool, str, Dict]:
        """Submit an order to the venue.
        
        Args:
            order_data: Order data
            
        Returns:
            Tuple of (success, order_id, response_data)
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.error(f"Cannot submit order to venue {self.config.venue_id}: not connected")
            return False, "", {"error": "Not connected"}
            
        try:
            # Implement venue-specific order submission logic in subclasses
            success, order_id, response = self._submit_order_impl(order_data)
            
            if success:
                logger.info(f"Submitted order {order_id} to venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to submit order to venue {self.config.venue_id}: {response.get('error', 'Unknown error')}")
                
            return success, order_id, response
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error submitting order to venue {self.config.venue_id}: {str(e)}")
            return False, "", {"error": str(e)}
            
    def _submit_order_impl(self, order_data: Dict) -> Tuple[bool, str, Dict]:
        """Implement venue-specific order submission logic.
        
        Args:
            order_data: Order data
            
        Returns:
            Tuple of (success, order_id, response_data)
        """
        # Default implementation for testing
        order_id = str(uuid.uuid4())
        return True, order_id, {"status": "accepted"}
        
    def cancel_order(self, order_id: str) -> Tuple[bool, Dict]:
        """Cancel an order at the venue.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, response_data)
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.error(f"Cannot cancel order at venue {self.config.venue_id}: not connected")
            return False, {"error": "Not connected"}
            
        try:
            # Implement venue-specific order cancellation logic in subclasses
            success, response = self._cancel_order_impl(order_id)
            
            if success:
                logger.info(f"Cancelled order {order_id} at venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to cancel order {order_id} at venue {self.config.venue_id}: {response.get('error', 'Unknown error')}")
                
            return success, response
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error cancelling order {order_id} at venue {self.config.venue_id}: {str(e)}")
            return False, {"error": str(e)}
            
    def _cancel_order_impl(self, order_id: str) -> Tuple[bool, Dict]:
        """Implement venue-specific order cancellation logic.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, response_data)
        """
        # Default implementation for testing
        return True, {"status": "cancelled"}
        
    def modify_order(self, order_id: str, modifications: Dict) -> Tuple[bool, Dict]:
        """Modify an order at the venue.
        
        Args:
            order_id: ID of the order to modify
            modifications: Modifications to apply
            
        Returns:
            Tuple of (success, response_data)
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.error(f"Cannot modify order at venue {self.config.venue_id}: not connected")
            return False, {"error": "Not connected"}
            
        try:
            # Implement venue-specific order modification logic in subclasses
            success, response = self._modify_order_impl(order_id, modifications)
            
            if success:
                logger.info(f"Modified order {order_id} at venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to modify order {order_id} at venue {self.config.venue_id}: {response.get('error', 'Unknown error')}")
                
            return success, response
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error modifying order {order_id} at venue {self.config.venue_id}: {str(e)}")
            return False, {"error": str(e)}
            
    def _modify_order_impl(self, order_id: str, modifications: Dict) -> Tuple[bool, Dict]:
        """Implement venue-specific order modification logic.
        
        Args:
            order_id: ID of the order to modify
            modifications: Modifications to apply
            
        Returns:
            Tuple of (success, response_data)
        """
        # Default implementation for testing
        return True, {"status": "modified"}
        
    def subscribe_market_data(self, instrument_id: str) -> bool:
        """Subscribe to market data for an instrument.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            True if subscription was successful, False otherwise
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.error(f"Cannot subscribe to market data at venue {self.config.venue_id}: not connected")
            return False
            
        try:
            # Implement venue-specific market data subscription logic in subclasses
            success = self._subscribe_market_data_impl(instrument_id)
            
            if success:
                logger.info(f"Subscribed to market data for {instrument_id} at venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to subscribe to market data for {instrument_id} at venue {self.config.venue_id}")
                
            return success
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error subscribing to market data for {instrument_id} at venue {self.config.venue_id}: {str(e)}")
            return False
            
    def _subscribe_market_data_impl(self, instrument_id: str) -> bool:
        """Implement venue-specific market data subscription logic.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            True if subscription was successful, False otherwise
        """
        # Default implementation for testing
        return True
        
    def unsubscribe_market_data(self, instrument_id: str) -> bool:
        """Unsubscribe from market data for an instrument.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        if self.status != ConnectionStatus.CONNECTED:
            logger.error(f"Cannot unsubscribe from market data at venue {self.config.venue_id}: not connected")
            return False
            
        try:
            # Implement venue-specific market data unsubscription logic in subclasses
            success = self._unsubscribe_market_data_impl(instrument_id)
            
            if success:
                logger.info(f"Unsubscribed from market data for {instrument_id} at venue {self.config.venue_id}")
            else:
                logger.error(f"Failed to unsubscribe from market data for {instrument_id} at venue {self.config.venue_id}")
                
            return success
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error unsubscribing from market data for {instrument_id} at venue {self.config.venue_id}: {str(e)}")
            return False
            
    def _unsubscribe_market_data_impl(self, instrument_id: str) -> bool:
        """Implement venue-specific market data unsubscription logic.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        # Default implementation for testing
        return True
        
    def register_order_callback(self, callback_id: str, callback: Callable) -> None:
        """Register a callback for order updates.
        
        Args:
            callback_id: Unique identifier for the callback
            callback: Function to call when an order update is received
        """
        self.order_callbacks[callback_id] = callback
        logger.info(f"Registered order callback {callback_id} for venue {self.config.venue_id}")
        
    def unregister_order_callback(self, callback_id: str) -> bool:
        """Unregister an order callback.
        
        Args:
            callback_id: ID of the callback to unregister
            
        Returns:
            True if callback was unregistered, False if it didn't exist
        """
        if callback_id in self.order_callbacks:
            del self.order_callbacks[callback_id]
            logger.info(f"Unregistered order callback {callback_id} for venue {self.config.venue_id}")
            return True
        return False
        
    def register_market_data_callback(self, callback_id: str, callback: Callable) -> None:
        """Register a callback for market data updates.
        
        Args:
            callback_id: Unique identifier for the callback
            callback: Function to call when a market data update is received
        """
        self.market_data_callbacks[callback_id] = callback
        logger.info(f"Registered market data callback {callback_id} for venue {self.config.venue_id}")
        
    def unregister_market_data_callback(self, callback_id: str) -> bool:
        """Unregister a market data callback.
        
        Args:
            callback_id: ID of the callback to unregister
            
        Returns:
            True if callback was unregistered, False if it didn't exist
        """
        if callback_id in self.market_data_callbacks:
            del self.market_data_callbacks[callback_id]
            logger.info(f"Unregistered market data callback {callback_id} for venue {self.config.venue_id}")
            return True
        return False
        
    def get_status(self) -> Dict:
        """Get the current status of the venue connection.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "venue_id": self.config.venue_id,
            "name": self.config.name,
            "type": self.config.venue_type.value,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_error": self.last_error,
            "enabled": self.config.enabled
        }


class MarketConnectivityManager:
    """Manages connections to multiple market venues."""
    
    def __init__(self):
        """Initialize market connectivity manager."""
        self.venues: Dict[str, VenueAdapter] = {}
        self.venue_configs: Dict[str, VenueConfig] = {}
        self.market_data_queue = queue.Queue()
        self.processing_thread = None
        self.processing_active = False
        
    def add_venue(self, config: VenueConfig, adapter_class=VenueAdapter) -> None:
        """Add a new venue.
        
        Args:
            config: Venue configuration
            adapter_class: Class to use for the venue adapter (default: VenueAdapter)
        """
        if config.venue_id in self.venues:
            logger.warning(f"Overwriting existing venue {config.venue_id}")
            
        adapter = adapter_class(config)
        self.venues[config.venue_id] = adapter
        self.venue_configs[config.venue_id] = config
        logger.info(f"Added venue {config.venue_id} ({config.name})")
        
    def remove_venue(self, venue_id: str) -> bool:
        """Remove a venue.
        
        Args:
            venue_id: ID of the venue to remove
            
        Returns:
            True if venue was removed, False if it didn't exist
        """
        if venue_id not in self.venues:
            logger.warning(f"Attempted to remove non-existent venue {venue_id}")
            return False
            
        # Disconnect if connected
        venue = self.venues[venue_id]
        if venue.status == ConnectionStatus.CONNECTED:
            venue.disconnect()
            
        del self.venues[venue_id]
        del self.venue_configs[venue_id]
        logger.info(f"Removed venue {venue_id}")
        return True
        
    def connect_venue(self, venue_id: str) -> bool:
        """Connect to a specific venue.
        
        Args:
            venue_id: ID of the venue to connect to
            
        Returns:
            True if connection was successful, False otherwise
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.connect()
        
    def disconnect_venue(self, venue_id: str) -> bool:
        """Disconnect from a specific venue.
        
        Args:
            venue_id: ID of the venue to disconnect from
            
        Returns:
            True if disconnection was successful, False otherwise
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.disconnect()
        
    def connect_all_venues(self) -> Dict[str, bool]:
        """Connect to all enabled venues.
        
        Returns:
            Dictionary mapping venue IDs to connection success
        """
        results = {}
        
        for venue_id, venue in self.venues.items():
            if venue.config.enabled:
                results[venue_id] = venue.connect()
                
        return results
        
    def disconnect_all_venues(self) -> Dict[str, bool]:
        """Disconnect from all connected venues.
        
        Returns:
            Dictionary mapping venue IDs to disconnection success
        """
        results = {}
        
        for venue_id, venue in self.venues.items():
            if venue.status == ConnectionStatus.CONNECTED:
                results[venue_id] = venue.disconnect()
                
        return results
        
    def get_venue_status(self, venue_id: str) -> Dict:
        """Get the status of a specific venue.
        
        Args:
            venue_id: ID of the venue
            
        Returns:
            Dictionary containing status information
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.get_status()
        
    def get_all_venue_statuses(self) -> Dict[str, Dict]:
        """Get the status of all venues.
        
        Returns:
            Dictionary mapping venue IDs to status dictionaries
        """
        return {venue_id: venue.get_status() for venue_id, venue in self.venues.items()}
        
    def submit_order(self, venue_id: str, order_data: Dict) -> Tuple[bool, str, Dict]:
        """Submit an order to a specific venue.
        
        Args:
            venue_id: ID of the venue
            order_data: Order data
            
        Returns:
            Tuple of (success, order_id, response_data)
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.submit_order(order_data)
        
    def cancel_order(self, venue_id: str, order_id: str) -> Tuple[bool, Dict]:
        """Cancel an order at a specific venue.
        
        Args:
            venue_id: ID of the venue
            order_id: ID of the order to cancel
            
        Returns:
            Tuple of (success, response_data)
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.cancel_order(order_id)
        
    def modify_order(self, venue_id: str, order_id: str, modifications: Dict) -> Tuple[bool, Dict]:
        """Modify an order at a specific venue.
        
        Args:
            venue_id: ID of the venue
            order_id: ID of the order to modify
            modifications: Modifications to apply
            
        Returns:
            Tuple of (success, response_data)
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.modify_order(order_id, modifications)
        
    def subscribe_market_data(self, venue_id: str, instrument_id: str) -> bool:
        """Subscribe to market data for an instrument at a specific venue.
        
        Args:
            venue_id: ID of the venue
            instrument_id: ID of the instrument
            
        Returns:
            True if subscription was successful, False otherwise
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.subscribe_market_data(instrument_id)
        
    def subscribe_market_data_all_venues(self, instrument_id: str) -> Dict[str, bool]:
        """Subscribe to market data for an instrument at all connected venues.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            Dictionary mapping venue IDs to subscription success
        """
        results = {}
        
        for venue_id, venue in self.venues.items():
            if venue.status == ConnectionStatus.CONNECTED:
                results[venue_id] = venue.subscribe_market_data(instrument_id)
                
        return results
        
    def unsubscribe_market_data(self, venue_id: str, instrument_id: str) -> bool:
        """Unsubscribe from market data for an instrument at a specific venue.
        
        Args:
            venue_id: ID of the venue
            instrument_id: ID of the instrument
            
        Returns:
            True if unsubscription was successful, False otherwise
            
        Raises:
            KeyError: If the venue_id doesn't exist
        """
        if venue_id not in self.venues:
            raise KeyError(f"Venue {venue_id} not found")
            
        venue = self.venues[venue_id]
        return venue.unsubscribe_market_data(instrument_id)
        
    def register_market_data_callback(self, callback: Callable) -> str:
        """Register a callback for all market data updates.
        
        Args:
            callback: Function to call when a market data update is received
            
        Returns:
            Callback ID
        """
        callback_id = str(uuid.uuid4())
        
        for venue_id, venue in self.venues.items():
            venue.register_market_data_callback(callback_id, callback)
            
        return callback_id
        
    def unregister_market_data_callback(self, callback_id: str) -> None:
        """Unregister a market data callback from all venues.
        
        Args:
            callback_id: ID of the callback to unregister
        """
        for venue_id, venue in self.venues.items():
            venue.unregister_market_data_callback(callback_id)
            
    def start_market_data_processing(self) -> None:
        """Start the market data processing thread."""
        if self.processing_active:
            logger.warning("Market data processing is already active")
            return
            
        self.processing_active = True
        self.processing_thread = threading.Thread(target=self._market_data_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Started market data processing")
        
    def stop_market_data_processing(self) -> None:
        """Stop the market data processing thread."""
        if not self.processing_active:
            logger.warning("Market data processing is not active")
            return
            
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Stopped market data processing")
        
    def _market_data_processing_loop(self) -> None:
        """Main loop for the market data processing thread."""
        while self.processing_active:
            try:
                # Process market data updates from the queue
                try:
                    update = self.market_data_queue.get(timeout=0.1)
                    self._process_market_data_update(update)
                    self.market_data_queue.task_done()
                except queue.Empty:
                    pass
            except Exception as e:
                logger.error(f"Error in market data processing loop: {str(e)}")
                
    def _process_market_data_update(self, update: MarketDataUpdate) -> None:
        """Process a market data update.
        
        Args:
            update: Market data update
        """
        # Implement market data processing logic here
        pass
        
    def get_venue_by_priority(self, instrument_id: str) -> Optional[str]:
        """Get the highest priority venue for an instrument.
        
        Args:
            instrument_id: ID of the instrument
            
        Returns:
            Venue ID or None if no suitable venue is found
        """
        # Filter connected venues
        connected_venues = [
            venue_id for venue_id, venue in self.venues.items()
            if venue.status == ConnectionStatus.CONNECTED
        ]
        
        if not connected_venues:
            return None
            
        # Sort by priority (higher is better)
        sorted_venues = sorted(
            connected_venues,
            key=lambda venue_id: self.venue_configs[venue_id].priority,
            reverse=True
        )
        
        return sorted_venues[0] if sorted_venues else None
