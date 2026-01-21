"""
Real-time Risk Monitoring Module.

This module provides functionality for real-time monitoring of risk metrics,
alerting, and dashboard integration for risk visualization.
"""

import datetime
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

# Optional libs (only needed if metrics use them)
# import numpy as np
# import pandas as pd

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for risk alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Channels for delivering risk alerts."""

    DASHBOARD = "dashboard"
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class RiskAlert:
    """Risk alert configuration and data."""

    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime.datetime
    value: float
    threshold: float
    channels: List[AlertChannel]
    metadata: Dict = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime.datetime] = None


class RiskMonitor:
    """Real-time risk monitoring system."""

    def __init__(self, update_interval: int = 60) -> None:
        """
        Initialize risk monitor.

        Args:
            update_interval: Interval in seconds for risk metric updates.
        """
        self.update_interval = update_interval
        self.metrics: Dict[str, Dict] = {}
        self.thresholds: Dict[str, Dict] = {}
        self.alerts: List[RiskAlert] = []
        self.alert_callbacks: Dict[AlertChannel, Callable] = {}
        self.monitoring_active = False
        self.monitor_thread = None

    # -------------------------------------------------------------------------
    # Metric registration
    # -------------------------------------------------------------------------
    def register_metric(
        self,
        metric_id: str,
        name: str,
        calculation_func: Callable,
        description: str = "",
    ) -> None:
        """
        Register a risk metric for monitoring.

        Args:
            metric_id: Unique identifier for the metric.
            name: Human-readable name of the metric.
            calculation_func: Callable returning the metric’s current value.
            description: Optional description.
        """
        self.metrics[metric_id] = {
            "name": name,
            "description": description,
            "calculation_func": calculation_func,
            "last_value": None,
            "last_updated": None,
            "history": [],
        }
        logger.info(f"Registered risk metric {metric_id}: {name}")

    def set_threshold(
        self,
        metric_id: str,
        warning_level: float,
        critical_level: float,
        emergency_level: Optional[float] = None,
        channels: Optional[List[AlertChannel]] = None,
    ) -> None:
        """
        Set alert thresholds for a metric.

        Args:
            metric_id: ID of the metric.
            warning_level: Warning threshold.
            critical_level: Critical threshold.
            emergency_level: Emergency threshold (optional).
            channels: Channels to deliver alerts through.
        """
        if metric_id not in self.metrics:
            raise KeyError(f"Metric {metric_id} not registered")

        if channels is None:
            channels = [AlertChannel.DASHBOARD]

        self.thresholds[metric_id] = {
            "warning": warning_level,
            "critical": critical_level,
            "emergency": emergency_level,
            "channels": channels,
        }

        logger.info(f"Set thresholds for metric {metric_id}")

    def register_alert_callback(
        self, channel: AlertChannel, callback: Callable
    ) -> None:
        """
        Register a callback for alert delivery on a specific channel.

        Args:
            channel: Alert delivery channel.
            callback: Function to call when an alert triggers.
        """
        self.alert_callbacks[channel] = callback
        logger.info(f"Registered callback for {channel.value} alerts")

    # -------------------------------------------------------------------------
    # Metric updating
    # -------------------------------------------------------------------------
    def update_metric(self, metric_id: str) -> float:
        """
        Update a single risk metric.

        Returns:
            The updated metric value.
        """
        if metric_id not in self.metrics:
            raise KeyError(f"Metric {metric_id} not registered")

        metric = self.metrics[metric_id]

        try:
            value = metric["calculation_func"]()
            timestamp = datetime.datetime.now()

            metric["last_value"] = value
            metric["last_updated"] = timestamp
            metric["history"].append((timestamp, value))

            if len(metric["history"]) > 1000:
                metric["history"] = metric["history"][-1000:]

            if metric_id in self.thresholds:
                self._check_thresholds(metric_id, value, timestamp)

            return value

        except Exception as e:
            logger.error(f"Error updating metric {metric_id}: {e}")
            raise

    def _check_thresholds(
        self, metric_id: str, value: float, timestamp: datetime.datetime
    ) -> None:
        """Check threshold rules and generate alerts."""
        thresholds = self.thresholds[metric_id]
        metric_name = self.metrics[metric_id]["name"]

        if thresholds["emergency"] is not None and value >= thresholds["emergency"]:
            severity = AlertSeverity.EMERGENCY
            threshold = thresholds["emergency"]
        elif value >= thresholds["critical"]:
            severity = AlertSeverity.CRITICAL
            threshold = thresholds["critical"]
        elif value >= thresholds["warning"]:
            severity = AlertSeverity.WARNING
            threshold = thresholds["warning"]
        else:
            return

        self._create_alert(
            metric_id=metric_id,
            metric_name=metric_name,
            severity=severity,
            value=value,
            threshold=threshold,
            timestamp=timestamp,
            channels=thresholds["channels"],
        )

    def _create_alert(
        self,
        metric_id: str,
        metric_name: str,
        severity: AlertSeverity,
        value: float,
        threshold: float,
        timestamp: datetime.datetime,
        channels: List[AlertChannel],
    ) -> None:
        """Create an alert and dispatch it through channels."""
        alert_id = f"{metric_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        message = (
            f"{metric_name} exceeded {severity.value} threshold: "
            f"{value:.4f} >= {threshold:.4f}"
        )

        alert = RiskAlert(
            alert_id=alert_id,
            metric_name=metric_name,
            severity=severity,
            message=message,
            timestamp=timestamp,
            value=value,
            threshold=threshold,
            channels=channels,
            metadata={"metric_id": metric_id},
        )

        self.alerts.append(alert)
        logger.warning(f"Risk alert: {message}")

        # Send alert
        for channel in channels:
            if channel in self.alert_callbacks:
                try:
                    self.alert_callbacks[channel](alert)
                except Exception as e:
                    logger.error(f"Error sending alert via {channel.value}: {e}")

    # -------------------------------------------------------------------------
    # Alert operations
    # -------------------------------------------------------------------------
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert by ID."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.datetime.now()
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False

    # -------------------------------------------------------------------------
    # Monitoring loop
    # -------------------------------------------------------------------------
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started real-time monitoring")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return

        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped monitoring")

    def _monitor_loop(self) -> None:
        """Internal thread loop."""
        while self.monitoring_active:
            try:
                for metric_id in self.metrics:
                    self.update_metric(metric_id)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            time.sleep(self.update_interval)

    # -------------------------------------------------------------------------
    # Dashboard and history utilities
    # -------------------------------------------------------------------------
    def get_metric_history(
        self,
        metric_id: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> List[Tuple]:
        """Return filtered historical metric values."""
        if metric_id not in self.metrics:
            raise KeyError(f"Metric {metric_id} not registered")

        history = self.metrics[metric_id]["history"]

        if not start_time and not end_time:
            return history

        return [
            (ts, val)
            for ts, val in history
            if (not start_time or ts >= start_time) and (not end_time or ts <= end_time)
        ]

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[RiskAlert]:
        """Return unacknowledged alerts, optionally filtered by severity."""
        alerts = [a for a in self.alerts if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def generate_risk_dashboard_data(self) -> Dict:
        """Return structured data for a dashboard."""
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {
                metric_id: {
                    "name": m["name"],
                    "description": m["description"],
                    "current_value": m["last_value"],
                    "last_updated": (
                        m["last_updated"].isoformat() if m["last_updated"] else None
                    ),
                    "thresholds": self.thresholds.get(metric_id),
                }
                for metric_id, m in self.metrics.items()
            },
            "active_alerts": len(self.get_active_alerts()),
            "alerts_by_severity": {
                sev.value: len(self.get_active_alerts(sev)) for sev in AlertSeverity
            },
            "recent_alerts": [
                {
                    "id": a.alert_id,
                    "metric": a.metric_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "acknowledged": a.acknowledged,
                }
                for a in sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[
                    :10
                ]
            ],
        }
