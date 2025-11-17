"""
Tests for the enhanced market connectivity module.

This module contains tests for the enhanced market connectivity functionality,
including reconnection logic, failure simulation, and data feed stability.
"""

import datetime
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from order_management.enhanced_market_connectivity import (
    EnhancedMarketConnectivityManager,
    EnhancedVenueAdapter,
    FailureMode,
)
from order_management.market_connectivity import (
    ConnectionStatus,
    VenueConfig,
    VenueType,
)


class TestEnhancedVenueAdapter(unittest.TestCase):
    """Test cases for the EnhancedVenueAdapter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a venue configuration
        self.config = VenueConfig(
            venue_id="test_venue",
            venue_type=VenueType.EXCHANGE,
            name="Test Exchange",
            priority=1,
            enabled=True,
            connection_params={
                "api_key": "test_key",
                "api_secret": "test_secret",
                "url": "wss://test.exchange.com/ws",
            },
            capabilities=["market_data", "order_execution"],
        )

        # Create an enhanced venue adapter
        self.adapter = EnhancedVenueAdapter(self.config)

        # Mock the _connect_impl and _disconnect_impl methods
        self.adapter._connect_impl = MagicMock(return_value=True)
        self.adapter._disconnect_impl = MagicMock(return_value=True)

    def test_initialization(self):
        """Test initialization of enhanced venue adapter."""
        self.assertEqual(self.adapter.config.venue_id, "test_venue")
        self.assertEqual(self.adapter.status, ConnectionStatus.DISCONNECTED)
        self.assertFalse(self.adapter.failure_simulation_enabled)
        self.assertEqual(self.adapter.messages_sent, 0)
        self.assertEqual(self.adapter.messages_received, 0)

    def test_connect_disconnect(self):
        """Test connection and disconnection."""
        # Connect
        success = self.adapter.connect()
        self.assertTrue(success)
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)
        self.adapter._connect_impl.assert_called_once()

        # Disconnect
        success = self.adapter.disconnect()
        self.assertTrue(success)
        self.assertEqual(self.adapter.status, ConnectionStatus.DISCONNECTED)
        self.adapter._disconnect_impl.assert_called_once()

    def test_connection_failure(self):
        """Test handling of connection failure."""
        # Make connection fail
        self.adapter._connect_impl.return_value = False

        # Connect
        success = self.adapter.connect()
        self.assertFalse(success)
        self.assertEqual(self.adapter.status, ConnectionStatus.ERROR)
        self.assertEqual(self.adapter.connection_failures, 1)

    def test_failure_simulation_disconnect(self):
        """Test simulation of connection failure."""
        # Enable failure simulation
        self.adapter.enable_failure_simulation(True)
        self.adapter.configure_failure_mode(FailureMode.DISCONNECT, 1.0)

        # Connect
        success = self.adapter.connect()
        self.assertFalse(success)
        self.assertEqual(self.adapter.status, ConnectionStatus.ERROR)
        self.assertEqual(self.adapter.last_error, "Simulated connection failure")

    def test_health_check(self):
        """Test health check functionality."""
        # Connect
        self.adapter.connect()
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)

        # Perform health check
        is_healthy = self.adapter._check_connection_health()
        self.assertTrue(is_healthy)

        # Simulate timeout
        self.adapter.last_activity_time = datetime.datetime.now() - datetime.timedelta(
            seconds=120
        )
        is_healthy = self.adapter._check_connection_health()
        self.assertFalse(is_healthy)

    def test_failure_simulation_timeout(self):
        """Test simulation of timeout failure."""
        # Connect
        self.adapter.connect()
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)

        # Enable failure simulation
        self.adapter.enable_failure_simulation(True)
        self.adapter.configure_failure_mode(FailureMode.TIMEOUT, 1.0)

        # Perform health check
        is_healthy = self.adapter._check_connection_health()
        self.assertFalse(is_healthy)

    def test_order_submission_with_failure_simulation(self):
        """Test order submission with failure simulation."""
        # Connect
        self.adapter.connect()
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)

        # Submit order normally
        order_data = {"symbol": "AAPL", "quantity": 100, "price": 150.0}
        success, order_id, response = self.adapter.submit_order(order_data)
        self.assertTrue(success)
        self.assertNotEqual(order_id, "")

        # Enable failure simulation
        self.adapter.enable_failure_simulation(True)
        self.adapter.configure_failure_mode(FailureMode.TIMEOUT, 1.0)

        # Submit order with simulated timeout
        success, order_id, response = self.adapter.submit_order(order_data)
        self.assertFalse(success)
        self.assertEqual(order_id, "")
        self.assertEqual(response["error"], "Simulated timeout")

    def test_market_data_processing_with_failure_simulation(self):
        """Test market data processing with failure simulation."""
        # Connect
        self.adapter.connect()
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)

        # Process market data normally
        data = {
            "venue_id": "test_venue",
            "instrument_id": "AAPL",
            "timestamp": datetime.datetime.now(),
            "bid": 150.0,
            "ask": 150.1,
            "last": 150.05,
            "volume": 1000,
        }
        update = self.adapter.process_market_data(data)
        self.assertIsNotNone(update)
        self.assertEqual(update.venue_id, "test_venue")
        self.assertEqual(update.instrument_id, "AAPL")
        self.assertEqual(update.bid, 150.0)
        self.assertEqual(update.ask, 150.1)

        # Enable failure simulation for data corruption
        self.adapter.enable_failure_simulation(True)
        self.adapter.configure_failure_mode(FailureMode.DATA_CORRUPTION, 1.0)

        # Process market data with simulated corruption
        update = self.adapter.process_market_data(data)
        self.assertIsNone(update)

        # Enable failure simulation for partial data
        self.adapter.configure_failure_mode(FailureMode.DATA_CORRUPTION, 0.0)
        self.adapter.configure_failure_mode(FailureMode.PARTIAL_DATA, 1.0)

        # Process market data with simulated partial data
        update = self.adapter.process_market_data(data)
        self.assertIsNotNone(update)
        self.assertIsNone(update.bid)
        self.assertIsNone(update.ask)

    def test_get_enhanced_status(self):
        """Test getting enhanced status information."""
        # Connect
        self.adapter.connect()
        self.assertEqual(self.adapter.status, ConnectionStatus.CONNECTED)

        # Get status
        status = self.adapter.get_status()

        # Check enhanced status fields
        self.assertEqual(status["venue_id"], "test_venue")
        self.assertEqual(status["status"], "connected")
        self.assertIn("messages_sent", status)
        self.assertIn("messages_received", status)
        self.assertIn("connection_attempts", status)
        self.assertIn("failure_simulation_enabled", status)


class TestEnhancedMarketConnectivityManager(unittest.TestCase):
    """Test cases for the EnhancedMarketConnectivityManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a market connectivity manager
        self.manager = EnhancedMarketConnectivityManager()

        # Create venue configurations
        self.venue1_config = VenueConfig(
            venue_id="venue1",
            venue_type=VenueType.EXCHANGE,
            name="Exchange 1",
            priority=1,
            enabled=True,
        )

        self.venue2_config = VenueConfig(
            venue_id="venue2",
            venue_type=VenueType.EXCHANGE,
            name="Exchange 2",
            priority=2,
            enabled=True,
        )

        # Add venues
        self.manager.add_venue(self.venue1_config)
        self.manager.add_venue(self.venue2_config)

        # Mock connect methods
        self.manager.venues["venue1"]._connect_impl = MagicMock(return_value=True)
        self.manager.venues["venue2"]._connect_impl = MagicMock(return_value=True)

    def test_initialization(self):
        """Test initialization of enhanced market connectivity manager."""
        self.assertEqual(len(self.manager.venues), 2)
        self.assertIn("venue1", self.manager.venues)
        self.assertIn("venue2", self.manager.venues)
        self.assertFalse(self.manager.global_failure_simulation_enabled)

    def test_connect_all_venues(self):
        """Test connecting all venues."""
        # Connect all venues
        results = self.manager.connect_all_venues()

        # Check results
        self.assertEqual(len(results), 2)
        self.assertTrue(results["venue1"])
        self.assertTrue(results["venue2"])

        # Check venue statuses
        self.assertEqual(
            self.manager.venues["venue1"].status, ConnectionStatus.CONNECTED
        )
        self.assertEqual(
            self.manager.venues["venue2"].status, ConnectionStatus.CONNECTED
        )

    def test_global_failure_simulation(self):
        """Test global failure simulation."""
        # Enable global failure simulation
        self.manager.enable_global_failure_simulation(True)
        self.assertTrue(self.manager.global_failure_simulation_enabled)

        # Check that all venues have failure simulation enabled
        self.assertTrue(self.manager.venues["venue1"].failure_simulation_enabled)
        self.assertTrue(self.manager.venues["venue2"].failure_simulation_enabled)

        # Disable global failure simulation
        self.manager.enable_global_failure_simulation(False)
        self.assertFalse(self.manager.global_failure_simulation_enabled)

        # Check that all venues have failure simulation disabled
        self.assertFalse(self.manager.venues["venue1"].failure_simulation_enabled)
        self.assertFalse(self.manager.venues["venue2"].failure_simulation_enabled)

    def test_venue_specific_failure_configuration(self):
        """Test configuring failure for a specific venue."""
        # Configure failure for venue1
        success = self.manager.configure_venue_failure(
            venue_id="venue1", mode=FailureMode.DISCONNECT, probability=0.5
        )
        self.assertTrue(success)

        # Check that venue1 has the failure mode configured
        self.assertEqual(
            self.manager.venues["venue1"].failure_probabilities[FailureMode.DISCONNECT],
            0.5,
        )

        # Check that venue2 does not have the failure mode configured
        self.assertNotIn(
            FailureMode.DISCONNECT, self.manager.venues["venue2"].failure_probabilities
        )

    def test_simulate_venue_failure(self):
        """Test simulating a complete venue failure."""
        # Connect all venues
        self.manager.connect_all_venues()

        # Ensure venue1 is connected before simulating failure
        self.assertEqual(
            self.manager.venues["venue1"].status, ConnectionStatus.CONNECTED
        )

        # Mock the reconnection_manager.connection_lost method to prevent async behavior
        self.manager.venues["venue1"].reconnection_manager.connection_lost = (
            lambda: None
        )

        # Simulate failure for venue1
        success = self.manager.simulate_venue_failure("venue1")
        self.assertTrue(success)

        # Check that venue1 is in error state
        self.assertEqual(self.manager.venues["venue1"].status, ConnectionStatus.ERROR)
        self.assertEqual(
            self.manager.venues["venue1"].last_error, "Simulated complete failure"
        )

        # Check that venue2 is still connected
        self.assertEqual(
            self.manager.venues["venue2"].status, ConnectionStatus.CONNECTED
        )

    def test_simulate_market_data_issue(self):
        """Test simulating a market data issue."""
        # Connect all venues
        self.manager.connect_all_venues()

        # Simulate market data issue for venue1
        success = self.manager.simulate_market_data_issue(
            venue_id="venue1", issue_type=FailureMode.DATA_CORRUPTION
        )
        self.assertTrue(success)

        # Check that venue1 has the issue configured
        self.assertTrue(self.manager.venues["venue1"].failure_simulation_enabled)
        self.assertEqual(
            self.manager.venues["venue1"].failure_probabilities[
                FailureMode.DATA_CORRUPTION
            ],
            1.0,
        )

    def test_get_connection_health_report(self):
        """Test getting a connection health report."""
        # Connect all venues
        self.manager.connect_all_venues()

        # Get health report
        report = self.manager.get_connection_health_report()

        # Check report
        self.assertEqual(report["total_venues"], 2)
        self.assertEqual(report["connected_venues"], 2)
        self.assertEqual(report["overall_status"], "healthy")
        self.assertIn("venue1", report["venues"])
        self.assertIn("venue2", report["venues"])
        self.assertEqual(report["venues"]["venue1"]["status"], "connected")
        self.assertEqual(report["venues"]["venue2"]["status"], "connected")

        # Mock the reconnection_manager.connection_lost method to prevent async behavior
        self.manager.venues["venue1"].reconnection_manager.connection_lost = (
            lambda: None
        )

        # Simulate failure for venue1
        self.manager.venues["venue1"].status = ConnectionStatus.ERROR
        self.manager.venues["venue1"].last_error = "Simulated complete failure"

        # Get updated health report
        report = self.manager.get_connection_health_report()

        # Check updated report
        self.assertEqual(report["connected_venues"], 1)
        self.assertEqual(report["overall_status"], "degraded")
        self.assertEqual(report["venues"]["venue1"]["status"], "error")
        self.assertEqual(report["venues"]["venue2"]["status"], "connected")

    def test_reset_failure_simulations(self):
        """Test resetting all failure simulations."""
        # Enable global failure simulation
        self.manager.enable_global_failure_simulation(True)

        # Configure specific failures
        self.manager.configure_venue_failure(
            venue_id="venue1", mode=FailureMode.DISCONNECT, probability=0.5
        )

        # Reset all failure simulations
        self.manager.reset_failure_simulations()

        # Check that global simulation is disabled
        self.assertFalse(self.manager.global_failure_simulation_enabled)

        # Check that venue-specific simulations are disabled
        self.assertFalse(self.manager.venues["venue1"].failure_simulation_enabled)
        self.assertFalse(self.manager.venues["venue2"].failure_simulation_enabled)


if __name__ == "__main__":
    unittest.main()
