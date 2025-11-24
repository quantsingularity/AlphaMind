# """"""
## Tests for the reconnection manager module.
#
## This module contains tests for the reconnection manager functionality,
## including reconnection attempts, backoff behavior, and circuit breaking.
# """"""

# import datetime
# import os
# import sys
# import time
# import unittest
# from unittest.mock import MagicMock, patch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# from order_management.reconnection_manager import (
#     ReconnectionConfig,
#     ReconnectionManager,
#     ReconnectionState,


# class TestReconnectionManager(unittest.TestCase):
#    """Test cases for the ReconnectionManager class."""
#
##     def setUp(self):
#        """Set up test fixtures."""
# Create a configuration with shorter timeouts for testing
#         self.config = ReconnectionConfig(
#             initial_delay=0.1,
#             max_delay=0.5,
#             backoff_factor=2.0,
#             jitter=0.1,
#             max_attempts=5,
#             circuit_break_threshold=3,
#             circuit_break_timeout=0.5,
#             health_check_interval=0.2,

#         self.reconnection_manager = ReconnectionManager(self.config)

# Create mock callbacks
#         self.connect_callback = MagicMock(return_value=True)
#         self.health_check_callback = MagicMock(return_value=True)
#         self.on_reconnect_callback = MagicMock()
#         self.on_give_up_callback = MagicMock()

#     def test_initialization(self):
#        """Test initialization of reconnection manager."""
##         self.assertEqual(self.reconnection_manager.state, ReconnectionState.IDLE)
##         self.assertEqual(self.reconnection_manager.stats.total_attempts, 0)
##         self.assertEqual(self.reconnection_manager.stats.successful_reconnects, 0)
##         self.assertEqual(self.reconnection_manager.stats.failed_attempts, 0)
##         self.assertFalse(self.reconnection_manager.is_running)
#
##     def test_start_stop(self):
#        """Test starting and stopping the reconnection manager."""
# Start the manager
#         self.reconnection_manager.start(
#             connection_id="test_connection",
#             connect_callback=self.connect_callback,
#             health_check_callback=self.health_check_callback,
#             on_reconnect_callback=self.on_reconnect_callback,
#             on_give_up_callback=self.on_give_up_callback,

#         self.assertTrue(self.reconnection_manager.is_running)
#         self.assertEqual(self.reconnection_manager.connection_id, "test_connection")

# Stop the manager
#         self.reconnection_manager.stop()
#         self.assertFalse(self.reconnection_manager.is_running)

#     def test_successful_reconnection(self):
#        """Test successful reconnection attempt."""
#        # Start the manager
##         self.reconnection_manager.start(
##             connection_id="test_connection",
##             connect_callback=self.connect_callback,
##             on_reconnect_callback=self.on_reconnect_callback,
#        )
#
#        # Trigger reconnection
##         self.reconnection_manager.connection_lost()
#
#        # Wait for reconnection attempt
##         time.sleep(0.2)
#
#        # Check that connect callback was called
##         self.connect_callback.assert_called_once()
#
#        # Check that on_reconnect callback was called with success=True
##         self.on_reconnect_callback.assert_called_once_with(True)
#
#        # Check stats
##         self.assertEqual(self.reconnection_manager.stats.total_attempts, 1)
##         self.assertEqual(self.reconnection_manager.stats.successful_reconnects, 1)
##         self.assertEqual(self.reconnection_manager.stats.failed_attempts, 0)
#
##     def test_failed_reconnection(self):
#        """Test failed reconnection attempt with backoff."""
# Configure connect callback to fail
#         self.connect_callback.return_value = False

# Create a manager with more predictable timing for testing
#         config = ReconnectionConfig(
#             initial_delay=0.1,
#             max_delay=0.5,
#             backoff_factor=2.0,
#             jitter=0.0,  # No jitter for predictable timing
#             max_attempts=3,
#             circuit_break_threshold=5,
#         reconnection_manager = ReconnectionManager(config)

# Start the manager with a limited number of attempts
#         reconnection_manager.start(
#             connection_id="test_connection",
#             connect_callback=self.connect_callback,
#             on_reconnect_callback=self.on_reconnect_callback,

# Trigger reconnection
#         reconnection_manager.connection_lost()

# Wait for initial attempt and first backoff
#         time.sleep(0.25)

# Check that connect callback was called twice (initial + first backoff)
#         self.assertEqual(self.connect_callback.call_count, 2)

# Check that on_reconnect callback was called with success=False twice
#         self.assertEqual(self.on_reconnect_callback.call_count, 2)
#         self.on_reconnect_callback.assert_called_with(False)

# Check stats
#         self.assertEqual(reconnection_manager.stats.total_attempts, 2)
#         self.assertEqual(reconnection_manager.stats.successful_reconnects, 0)
#         self.assertEqual(reconnection_manager.stats.failed_attempts, 2)
#         self.assertEqual(reconnection_manager.stats.consecutive_failures, 2)

#     def test_circuit_breaker(self):
#        """Test circuit breaker activation after consecutive failures."""
#        # Configure connect callback to fail
##         self.connect_callback.return_value = False
#
#        # Start the manager
##         self.reconnection_manager.start(
##             connection_id="test_connection", connect_callback=self.connect_callback
#        )
#
#        # Trigger reconnection
##         self.reconnection_manager.connection_lost()
#
#        # Wait for enough failures to trigger circuit breaker
#        # (3 failures with short delays)
##         time.sleep(0.5)
#
#        # Check that circuit breaker was triggered
##         self.assertEqual(
##             self.reconnection_manager.state, ReconnectionState.CIRCUIT_OPEN
#        )
#
#        # Check that connect callback was called 3 times (threshold)
##         self.assertEqual(self.connect_callback.call_count, 3)
#
#        # Wait for circuit breaker timeout
##         time.sleep(0.6)
#
#        # Manually trigger reconnection after circuit breaker timeout
##         self.connect_callback.reset_mock()
##         self.reconnection_manager.connection_lost()
##         time.sleep(0.2)
#
#        # Check that connect callback was called again after timeout
##         self.assertEqual(self.connect_callback.call_count, 1)
#
##     def test_max_attempts(self):
#        """Test stopping after maximum number of attempts."""
# Configure connect callback to fail
#         self.connect_callback.return_value = False

# Start the manager with max_attempts=2
#         config = ReconnectionConfig(
#             initial_delay=0.1,
#             max_delay=0.2,
#             max_attempts=2,
#             circuit_break_threshold=10,  # Set high to avoid circuit breaking
#         reconnection_manager = ReconnectionManager(config)
#         reconnection_manager.start(
#             connection_id="test_connection",
#             connect_callback=self.connect_callback,
#             on_give_up_callback=self.on_give_up_callback,

# Trigger reconnection
#         reconnection_manager.connection_lost()

# Wait for max attempts
#         time.sleep(0.5)

# Check that connect callback was called exactly twice
#         self.assertEqual(self.connect_callback.call_count, 2)

# Check that on_give_up callback was called
#         self.on_give_up_callback.assert_called_once()

#     def test_health_check_failure(self):
#        """Test reconnection triggered by health check failure."""
#        # Start the manager with health check
##         self.reconnection_manager.start(
##             connection_id="test_connection",
##             connect_callback=self.connect_callback,
##             health_check_callback=self.health_check_callback,
#        )
#
#        # Simulate connection established
##         self.reconnection_manager.connection_established()
#
#        # Make health check fail
##         self.health_check_callback.return_value = False
#
#        # Manually trigger health check instead of waiting
##         self.reconnection_manager._perform_health_check()
#
#        # Wait a short time for reconnection to be triggered
##         time.sleep(0.2)
#
#        # Check that connect callback was called after health check failure
##         self.connect_callback.assert_called_once()
#
##     def test_exponential_backoff(self):
#        """Test exponential backoff behavior."""
# Configure connect callback to fail
#         self.connect_callback.return_value = False

# Create a manager with predictable backoff (no jitter)
#         config = ReconnectionConfig(
#             initial_delay=0.1,
#             max_delay=1.0,
#             backoff_factor=2.0,
#             jitter=0.0,
#             max_attempts=0,
#             circuit_break_threshold=10,  # Set high to avoid circuit breaking
#         reconnection_manager = ReconnectionManager(config)

# Start the manager
#         reconnection_manager.start(
#             connection_id="test_connection", connect_callback=self.connect_callback

# Store the initial delay for verification
#         initial_delay = reconnection_manager.current_delay

# Trigger reconnection
#         reconnection_manager.connection_lost()

# Check initial delay
#         self.assertEqual(initial_delay, 0.1)

# Wait for first attempt and backoff
#         time.sleep(0.2)

# Check that delay increased (should be doubled after first failure)
# The current_delay might be 0.2 or 0.4 depending on how many attempts occurred
# so we just verify it increased from initial value
#         self.assertGreater(reconnection_manager.current_delay, initial_delay)

# Store the current delay for next comparison
#         first_backoff_delay = reconnection_manager.current_delay

# Wait for next attempt
#         time.sleep(0.3)

# Check that delay increased again or stayed at max
#         self.assertGreaterEqual(reconnection_manager.current_delay, first_backoff_delay)

#     def test_connection_established_resets_state(self):
#        """Test that connection_established resets reconnection state."""
#        # Configure connect callback to fail
##         self.connect_callback.return_value = False
#
#        # Start the manager
##         self.reconnection_manager.start(
##             connection_id="test_connection", connect_callback=self.connect_callback
#        )
#
#        # Trigger reconnection
##         self.reconnection_manager.connection_lost()
#
#        # Wait for attempt
##         time.sleep(0.2)
#
#        # Check that state is not IDLE
##         self.assertNotEqual(self.reconnection_manager.state, ReconnectionState.IDLE)
#
#        # Simulate connection established
##         self.reconnection_manager.connection_established()
#
#        # Check that state is reset to IDLE
##         self.assertEqual(self.reconnection_manager.state, ReconnectionState.IDLE)
##         self.assertEqual(
##             self.reconnection_manager.current_delay, self.config.initial_delay
#        )
##         self.assertEqual(self.reconnection_manager.stats.consecutive_failures, 0)
#
##     def test_get_stats(self):
#        """Test getting reconnection statistics."""
# Start the manager
#         self.reconnection_manager.start(
#             connection_id="test_connection", connect_callback=self.connect_callback

# Trigger reconnection
#         self.reconnection_manager.connection_lost()

# Wait for attempt
#         time.sleep(0.2)

# Get stats
#         stats = self.reconnection_manager.get_stats()

# Check stats
#         self.assertEqual(stats["connection_id"], "test_connection")
#         self.assertEqual(stats["total_attempts"], 1)
#         self.assertEqual(stats["successful_reconnects"], 1)
#         self.assertEqual(stats["failed_attempts"], 0)


# if __name__ == "__main__":
#     unittest.main()
