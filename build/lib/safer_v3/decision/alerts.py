"""
Alert Management for SAFER v3.0.

This module implements the alert system for RUL-based maintenance
warnings and safety notifications.

Alert Levels (aligned with aerospace standards):
- INFO: Informational, no action required
- ADVISORY: Monitor closely, schedule inspection
- CAUTION: Near-term maintenance recommended
- WARNING: Urgent maintenance required
- CRITICAL: Immediate action necessary

Design Principles:
- Configurable thresholds for different operational contexts
- Hysteresis to prevent alert flickering
- Rate limiting to avoid alert fatigue
- Comprehensive logging for audit trails
- Integration with SAFER decision fabric
"""

import time
import json
import logging
from enum import IntEnum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime
import threading


logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Alert severity levels (aerospace-aligned)."""
    INFO = 0
    ADVISORY = 1
    CAUTION = 2
    WARNING = 3
    CRITICAL = 4
    
    @property
    def color(self) -> str:
        """Color code for display."""
        colors = {
            AlertLevel.INFO: 'green',
            AlertLevel.ADVISORY: 'blue',
            AlertLevel.CAUTION: 'yellow',
            AlertLevel.WARNING: 'orange',
            AlertLevel.CRITICAL: 'red',
        }
        return colors.get(self, 'white')
    
    @property
    def requires_action(self) -> bool:
        """Whether this level requires operator action."""
        return self >= AlertLevel.CAUTION


@dataclass
class Alert:
    """Individual alert instance.
    
    Attributes:
        level: Alert severity level
        message: Human-readable alert message
        source: Component that raised the alert
        timestamp: Unix timestamp when alert was created
        rul_value: RUL value that triggered the alert (if applicable)
        confidence: Confidence interval or uncertainty
        metadata: Additional context
        acknowledged: Whether alert has been acknowledged
        resolved: Whether alert condition has resolved
    """
    level: AlertLevel
    message: str
    source: str
    timestamp: float = field(default_factory=time.time)
    rul_value: Optional[float] = None
    confidence: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    @property
    def alert_id(self) -> str:
        """Unique alert identifier."""
        return f"{self.source}_{int(self.timestamp * 1000)}"
    
    @property
    def age_seconds(self) -> float:
        """Age of alert in seconds."""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['level'] = self.level.name
        d['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        d['alert_id'] = self.alert_id
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        logger.info(f"Alert acknowledged: {self.alert_id}")
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        logger.info(f"Alert resolved: {self.alert_id}")


@dataclass
class AlertRule:
    """Rule for generating alerts based on conditions.
    
    Attributes:
        name: Rule identifier
        level: Alert level to generate
        condition: Function (rul, context) -> bool
        message_template: Message template with {rul}, {threshold} placeholders
        threshold: Threshold value for condition
        hysteresis: Hysteresis value to prevent flickering
        cooldown: Minimum seconds between alerts from this rule
        enabled: Whether rule is active
    """
    name: str
    level: AlertLevel
    condition: Callable[[float, Dict[str, Any]], bool]
    message_template: str
    threshold: float = 0.0
    hysteresis: float = 0.0
    cooldown: float = 60.0
    enabled: bool = True
    
    _last_trigger: float = field(default=0.0, repr=False)
    _state: bool = field(default=False, repr=False)
    
    def evaluate(
        self,
        rul: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """Evaluate rule and optionally generate alert.
        
        Args:
            rul: Current RUL value
            context: Additional context for condition
            
        Returns:
            Alert if triggered, None otherwise
        """
        if not self.enabled:
            return None
        
        context = context or {}
        context['threshold'] = self.threshold
        
        current_time = time.time()
        
        # Evaluate condition
        triggered = self.condition(rul, context)
        
        # Apply hysteresis
        if self._state:
            # Already triggered - use hysteresis to clear
            if not triggered and rul > self.threshold + self.hysteresis:
                self._state = False
        else:
            # Not triggered - check for new trigger
            if triggered:
                self._state = True
        
        # Check cooldown
        if self._state and (current_time - self._last_trigger) >= self.cooldown:
            self._last_trigger = current_time
            
            # Prepare format kwargs (avoid duplicate threshold)
            format_kwargs = {'rul': rul, 'threshold': self.threshold}
            # Add context items that don't conflict
            for key, value in context.items():
                if key not in format_kwargs:
                    format_kwargs[key] = value
            
            message = self.message_template.format(**format_kwargs)
            
            return Alert(
                level=self.level,
                message=message,
                source=self.name,
                rul_value=rul,
                metadata={'threshold': self.threshold, **context},
            )
        
        return None


def create_rul_alert_rules(
    critical_threshold: float = 10,
    warning_threshold: float = 25,
    caution_threshold: float = 50,
    advisory_threshold: float = 100,
) -> List[AlertRule]:
    """Create standard RUL-based alert rules.
    
    Args:
        critical_threshold: RUL cycles for CRITICAL alerts
        warning_threshold: RUL cycles for WARNING alerts
        caution_threshold: RUL cycles for CAUTION alerts
        advisory_threshold: RUL cycles for ADVISORY alerts
        
    Returns:
        List of AlertRule instances
    """
    rules = [
        AlertRule(
            name='rul_critical',
            level=AlertLevel.CRITICAL,
            condition=lambda rul, ctx: rul <= ctx['threshold'],
            message_template=(
                "CRITICAL: RUL = {rul:.1f} cycles. "
                "Immediate maintenance action required."
            ),
            threshold=critical_threshold,
            hysteresis=2,
            cooldown=30,
        ),
        AlertRule(
            name='rul_warning',
            level=AlertLevel.WARNING,
            condition=lambda rul, ctx: rul <= ctx['threshold'],
            message_template=(
                "WARNING: RUL = {rul:.1f} cycles. "
                "Schedule urgent maintenance within {rul:.0f} cycles."
            ),
            threshold=warning_threshold,
            hysteresis=5,
            cooldown=60,
        ),
        AlertRule(
            name='rul_caution',
            level=AlertLevel.CAUTION,
            condition=lambda rul, ctx: rul <= ctx['threshold'],
            message_template=(
                "CAUTION: RUL = {rul:.1f} cycles. "
                "Plan maintenance inspection."
            ),
            threshold=caution_threshold,
            hysteresis=5,
            cooldown=120,
        ),
        AlertRule(
            name='rul_advisory',
            level=AlertLevel.ADVISORY,
            condition=lambda rul, ctx: rul <= ctx['threshold'],
            message_template=(
                "ADVISORY: RUL = {rul:.1f} cycles. "
                "Monitor degradation trend."
            ),
            threshold=advisory_threshold,
            hysteresis=10,
            cooldown=300,
        ),
    ]
    
    return rules


class AlertManager:
    """Central alert management system.
    
    Manages alert rules, tracks active alerts, handles
    acknowledgment/resolution, and provides alert history.
    
    Thread-safe for multi-process SAFER deployment.
    
    Usage:
        manager = AlertManager()
        manager.add_rules(create_rul_alert_rules())
        
        # Process RUL prediction
        alerts = manager.process(rul_value=45.2)
        
        # Get active alerts
        active = manager.get_active_alerts()
        
        # Acknowledge alert
        manager.acknowledge_alert(alert_id)
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        auto_resolve_timeout: float = 3600.0,
    ):
        """Initialize alert manager.
        
        Args:
            max_history: Maximum alerts to keep in history
            auto_resolve_timeout: Auto-resolve alerts after this many seconds
        """
        self.max_history = max_history
        self.auto_resolve_timeout = auto_resolve_timeout
        
        self._rules: List[AlertRule] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._history: deque = deque(maxlen=max_history)
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        
        # Statistics
        self._total_alerts = 0
        self._alerts_by_level: Dict[AlertLevel, int] = {
            level: 0 for level in AlertLevel
        }
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.
        
        Args:
            rule: AlertRule to add
        """
        with self._lock:
            self._rules.append(rule)
            logger.debug(f"Added alert rule: {rule.name}")
    
    def add_rules(self, rules: List[AlertRule]) -> None:
        """Add multiple alert rules.
        
        Args:
            rules: List of AlertRules
        """
        for rule in rules:
            self.add_rule(rule)
    
    def register_callback(
        self,
        callback: Callable[[Alert], None],
    ) -> None:
        """Register callback for new alerts.
        
        Args:
            callback: Function to call with new alerts
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def process(
        self,
        rul_value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Process RUL value against all rules.
        
        Args:
            rul_value: Current RUL prediction
            context: Additional context (confidence, etc.)
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        context = context or {}
        
        with self._lock:
            for rule in self._rules:
                alert = rule.evaluate(rul_value, context)
                if alert is not None:
                    self._handle_new_alert(alert)
                    triggered_alerts.append(alert)
            
            # Check for auto-resolution
            self._check_auto_resolve()
        
        return triggered_alerts
    
    def _handle_new_alert(self, alert: Alert) -> None:
        """Handle new alert internally.
        
        Args:
            alert: New alert instance
        """
        # Add to active alerts
        self._active_alerts[alert.alert_id] = alert
        
        # Add to history
        self._history.append(alert)
        
        # Update statistics
        self._total_alerts += 1
        self._alerts_by_level[alert.level] += 1
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.ADVISORY: logging.INFO,
            AlertLevel.CAUTION: logging.WARNING,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
        }.get(alert.level, logging.INFO)
        
        logger.log(log_level, f"Alert: [{alert.level.name}] {alert.message}")
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _check_auto_resolve(self) -> None:
        """Check for alerts to auto-resolve."""
        current_time = time.time()
        
        for alert_id, alert in list(self._active_alerts.items()):
            if alert.age_seconds > self.auto_resolve_timeout:
                alert.resolved = True
                del self._active_alerts[alert_id]
                logger.info(f"Auto-resolved alert: {alert_id}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if found and acknowledged
        """
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].acknowledge()
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if found and resolved
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolve()
                del self._active_alerts[alert_id]
                return True
            return False
    
    def get_active_alerts(
        self,
        min_level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get currently active alerts.
        
        Args:
            min_level: Minimum alert level to include
            
        Returns:
            List of active alerts, sorted by level (highest first)
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
            
            if min_level is not None:
                alerts = [a for a in alerts if a.level >= min_level]
            
            # Sort by level (descending) then timestamp (ascending)
            alerts.sort(key=lambda a: (-a.level, a.timestamp))
            
            return alerts
    
    def get_highest_alert(self) -> Optional[Alert]:
        """Get highest severity active alert.
        
        Returns:
            Highest severity alert or None
        """
        active = self.get_active_alerts()
        return active[0] if active else None
    
    def get_alert_history(
        self,
        limit: int = 100,
        level_filter: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """Get alert history.
        
        Args:
            limit: Maximum alerts to return
            level_filter: Filter by specific level
            
        Returns:
            List of historical alerts (most recent first)
        """
        with self._lock:
            history = list(self._history)
            
            if level_filter is not None:
                history = [a for a in history if a.level == level_filter]
            
            # Reverse for most recent first
            history = history[::-1][:limit]
            
            return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        with self._lock:
            return {
                'total_alerts': self._total_alerts,
                'active_alerts': len(self._active_alerts),
                'alerts_by_level': {
                    level.name: count
                    for level, count in self._alerts_by_level.items()
                },
                'unacknowledged': sum(
                    1 for a in self._active_alerts.values()
                    if not a.acknowledged
                ),
            }
    
    def clear_all(self) -> None:
        """Clear all active alerts."""
        with self._lock:
            for alert in self._active_alerts.values():
                alert.resolved = True
            self._active_alerts.clear()
            logger.info("All alerts cleared")
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alert rule.
        
        Args:
            rule_name: Name of rule to enable
            
        Returns:
            True if found and enabled
        """
        with self._lock:
            for rule in self._rules:
                if rule.name == rule_name:
                    rule.enabled = True
                    return True
            return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alert rule.
        
        Args:
            rule_name: Name of rule to disable
            
        Returns:
            True if found and disabled
        """
        with self._lock:
            for rule in self._rules:
                if rule.name == rule_name:
                    rule.enabled = False
                    return True
            return False


class AlertAggregator:
    """Aggregates alerts across multiple sources/engines.
    
    For fleet-level monitoring where multiple engines
    report alerts simultaneously.
    """
    
    def __init__(
        self,
        sources: List[str],
        max_alerts_per_source: int = 100,
    ):
        """Initialize alert aggregator.
        
        Args:
            sources: List of source identifiers (e.g., engine IDs)
            max_alerts_per_source: Max alerts to track per source
        """
        self.sources = sources
        self._managers: Dict[str, AlertManager] = {
            source: AlertManager(max_history=max_alerts_per_source)
            for source in sources
        }
        self._lock = threading.Lock()
    
    def get_manager(self, source: str) -> AlertManager:
        """Get manager for specific source.
        
        Args:
            source: Source identifier
            
        Returns:
            AlertManager for source
        """
        if source not in self._managers:
            raise KeyError(f"Unknown source: {source}")
        return self._managers[source]
    
    def process(
        self,
        source: str,
        rul_value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Alert]:
        """Process RUL for specific source.
        
        Args:
            source: Source identifier
            rul_value: RUL prediction
            context: Additional context
            
        Returns:
            List of triggered alerts
        """
        return self.get_manager(source).process(rul_value, context)
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get aggregated fleet alert status.
        
        Returns:
            Dictionary with fleet-wide statistics
        """
        with self._lock:
            status = {
                'total_sources': len(self.sources),
                'sources_with_alerts': 0,
                'critical_sources': [],
                'total_active_alerts': 0,
                'by_level': {level.name: 0 for level in AlertLevel},
            }
            
            for source, manager in self._managers.items():
                active = manager.get_active_alerts()
                
                if active:
                    status['sources_with_alerts'] += 1
                    status['total_active_alerts'] += len(active)
                    
                    for alert in active:
                        status['by_level'][alert.level.name] += 1
                    
                    # Check for critical
                    if any(a.level == AlertLevel.CRITICAL for a in active):
                        status['critical_sources'].append(source)
            
            return status
    
    def get_all_active_alerts(self) -> Dict[str, List[Alert]]:
        """Get all active alerts across fleet.
        
        Returns:
            Dictionary mapping source to active alerts
        """
        return {
            source: manager.get_active_alerts()
            for source, manager in self._managers.items()
        }
