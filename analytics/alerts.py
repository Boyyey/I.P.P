"""
Alerts Module
Handles alert generation, notification management, and threshold monitoring.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import threading
import time
from enum import Enum
from dataclasses import dataclass
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import our modules
from ml.inference import RealTimePredictor
from ml.feature_engineering import FeatureEngineer


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    BURNOUT_RISK = "burnout_risk"
    PERFORMANCE_DECLINE = "performance_decline"
    FATIGUE_DETECTED = "fatigue_detected"
    SLEEP_DEPRIVATION = "sleep_deprivation"
    ERROR_RATE_SPIKE = "error_rate_spike"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    recommendations: List[str]
    acknowledged: bool = False
    resolved: bool = False


class AlertThresholds:
    """Manages alert thresholds and rules."""
    
    def __init__(self):
        self.thresholds = self._default_thresholds()
    
    def _default_thresholds(self) -> Dict:
        """Default alert thresholds."""
        return {
            'burnout_risk': {
                'critical': 0.8,
                'high': 0.6,
                'medium': 0.4,
                'low': 0.2
            },
            'performance_decline_rate': {
                'critical': -0.05,  # 5% decline per hour
                'high': -0.03,
                'medium': -0.02,
                'low': -0.01
            },
            'error_rate': {
                'critical': 0.3,
                'high': 0.2,
                'medium': 0.15,
                'low': 0.1
            },
            'cognitive_load': {
                'critical': 0.9,
                'high': 0.8,
                'medium': 0.7,
                'low': 0.6
            },
            'sleep_debt': {
                'critical': 8,  # hours
                'high': 5,
                'medium': 3,
                'low': 1
            },
            'typing_speed_decline': {
                'critical': -0.3,  # 30% decline
                'high': -0.2,
                'medium': -0.15,
                'low': -0.1
            },
            'focus_ratio': {
                'critical': 0.3,
                'high': 0.4,
                'medium': 0.5,
                'low': 0.6
            }
        }
    
    def get_threshold(self, metric: str, severity: AlertSeverity) -> Optional[float]:
        """Get threshold for a specific metric and severity."""
        if metric in self.thresholds and severity.value in self.thresholds[metric]:
            return self.thresholds[metric][severity.value]
        return None
    
    def update_threshold(self, metric: str, severity: AlertSeverity, value: float):
        """Update threshold value."""
        if metric not in self.thresholds:
            self.thresholds[metric] = {}
        self.thresholds[metric][severity.value] = value
    
    def load_from_file(self, file_path: str):
        """Load thresholds from JSON file."""
        try:
            with open(file_path, 'r') as f:
                self.thresholds = json.load(f)
        except Exception as e:
            print(f"Failed to load thresholds from {file_path}: {e}")
    
    def save_to_file(self, file_path: str):
        """Save thresholds to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
        except Exception as e:
            print(f"Failed to save thresholds to {file_path}: {e}")


class AlertGenerator:
    """Generates alerts based on real-time metrics and predictions."""
    
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer(db_path)
        self.predictor = RealTimePredictor(db_path)
        self.thresholds = AlertThresholds()
        
        # Alert history
        self.alert_history = []
        self.active_alerts = []
        
        # Alert configuration
        self.alert_cooldown = {
            AlertType.BURNOUT_RISK: timedelta(hours=1),
            AlertType.PERFORMANCE_DECLINE: timedelta(hours=2),
            AlertType.FATIGUE_DETECTED: timedelta(hours=1),
            AlertType.SLEEP_DEPRIVATION: timedelta(hours=6),
            AlertType.ERROR_RATE_SPIKE: timedelta(minutes=30),
            AlertType.COGNITIVE_OVERLOAD: timedelta(minutes=45),
            AlertType.ANOMALY_DETECTED: timedelta(hours=1)
        }
        
        # Load custom thresholds if available
        thresholds_file = "alert_thresholds.json"
        if os.path.exists(thresholds_file):
            self.thresholds.load_from_file(thresholds_file)
    
    def check_all_alerts(self) -> List[Alert]:
        """Check all alert conditions and generate alerts."""
        alerts = []
        
        try:
            # Get current features and predictions
            current_features = self.predictor.extract_current_features()
            
            if not current_features:
                return []
            
            features_dict = current_features['features_dict']
            
            # Check burnout risk
            burnout_alert = self._check_burnout_risk(features_dict)
            if burnout_alert:
                alerts.append(burnout_alert)
            
            # Check performance decline
            performance_alert = self._check_performance_decline(features_dict)
            if performance_alert:
                alerts.append(performance_alert)
            
            # Check fatigue indicators
            fatigue_alert = self._check_fatigue(features_dict)
            if fatigue_alert:
                alerts.append(fatigue_alert)
            
            # Check sleep deprivation
            sleep_alert = self._check_sleep_deprivation(features_dict)
            if sleep_alert:
                alerts.append(sleep_alert)
            
            # Check error rate spike
            error_alert = self._check_error_rate(features_dict)
            if error_alert:
                alerts.append(error_alert)
            
            # Check cognitive overload
            cognitive_alert = self._check_cognitive_overload(features_dict)
            if cognitive_alert:
                alerts.append(cognitive_alert)
            
            # Check anomalies
            anomaly_alert = self._check_anomalies(features_dict)
            if anomaly_alert:
                alerts.append(anomaly_alert)
            
        except Exception as e:
            print(f"Alert checking failed: {e}")
        
        # Filter alerts based on cooldown
        filtered_alerts = self._filter_alerts_by_cooldown(alerts)
        
        # Add to history and active alerts
        for alert in filtered_alerts:
            self.alert_history.append(alert)
            self.active_alerts.append(alert)
        
        return filtered_alerts
    
    def _check_burnout_risk(self, features: Dict) -> Optional[Alert]:
        """Check for burnout risk alerts."""
        try:
            # Get burnout prediction
            burnout_pred = self.predictor.predict_burnout_risk()
            
            if 'error' in burnout_pred:
                return None
            
            risk_score = burnout_pred.get('burnout_risk_score', 0)
            risk_level = burnout_pred.get('risk_level', 'low')
            
            # Determine severity
            if risk_level == 'critical':
                severity = AlertSeverity.CRITICAL
            elif risk_level == 'high':
                severity = AlertSeverity.HIGH
            elif risk_level == 'medium':
                severity = AlertSeverity.MEDIUM
            else:
                return None  # Low risk doesn't trigger alert
            
            # Check if we're already in cooldown
            if self._is_in_cooldown(AlertType.BURNOUT_RISK):
                return None
            
            alert = Alert(
                id=f"burnout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.BURNOUT_RISK,
                severity=severity,
                title=f"Burnout Risk: {risk_level.upper()}",
                message=f"Current burnout risk score is {risk_score:.2f} ({risk_level} level)",
                timestamp=datetime.now(),
                metrics={'burnout_risk_score': risk_score},
                recommendations=burnout_pred.get('recommendations', [])
            )
            
            return alert
            
        except Exception as e:
            print(f"Burnout risk check failed: {e}")
            return None
    
    def _check_performance_decline(self, features: Dict) -> Optional[Alert]:
        """Check for performance decline alerts."""
        try:
            # Get recent performance trend
            if 'performance_composite_ma_6h' not in features:
                return None
            
            current_perf = features.get('performance_composite', 0)
            recent_avg = features.get('performance_composite_ma_6h', current_perf)
            
            # Calculate decline rate
            if recent_avg > 0:
                decline_rate = (current_perf - recent_avg) / recent_avg
            else:
                decline_rate = 0
            
            # Determine severity
            severity = None
            for sev in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
                threshold = self.thresholds.get_threshold('performance_decline_rate', sev)
                if threshold and decline_rate <= threshold:
                    severity = sev
                    break
            
            if not severity:
                return None
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.PERFORMANCE_DECLINE):
                return None
            
            alert = Alert(
                id=f"perf_decline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.PERFORMANCE_DECLINE,
                severity=severity,
                title=f"Performance Decline: {severity.value.upper()}",
                message=f"Performance declined by {abs(decline_rate)*100:.1f}% in recent hours",
                timestamp=datetime.now(),
                metrics={
                    'current_performance': current_perf,
                    'recent_average': recent_avg,
                    'decline_rate': decline_rate
                },
                recommendations=self._get_performance_decline_recommendations(decline_rate, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Performance decline check failed: {e}")
            return None
    
    def _check_fatigue(self, features: Dict) -> Optional[Alert]:
        """Check for fatigue indicators."""
        try:
            fatigue_indicators = []
            
            # Check typing speed decline
            if 'wpm' in features and 'wpm_ma_6h' in features:
                current_wpm = features['wpm']
                avg_wpm = features['wpm_ma_6h']
                
                if avg_wpm > 0:
                    wpm_decline = (current_wpm - avg_wpm) / avg_wpm
                    if wpm_decline < -0.15:  # 15% decline
                        fatigue_indicators.append(("typing_speed_decline", wpm_decline))
            
            # Check increased error rate
            if 'error_rate' in features:
                error_rate = features['error_rate']
                if error_rate > 0.15:  # High error rate
                    fatigue_indicators.append(("high_error_rate", error_rate))
            
            # Check high cognitive load
            if 'cognitive_load_score' in features:
                cog_load = features['cognitive_load_score']
                if cog_load > 0.8:  # High cognitive load
                    fatigue_indicators.append(("high_cognitive_load", cog_load))
            
            # Determine fatigue level
            if len(fatigue_indicators) >= 3:
                severity = AlertSeverity.CRITICAL
            elif len(fatigue_indicators) >= 2:
                severity = AlertSeverity.HIGH
            elif len(fatigue_indicators) >= 1:
                severity = AlertSeverity.MEDIUM
            else:
                return None
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.FATIGUE_DETECTED):
                return None
            
            metrics = dict(fatigue_indicators)
            
            alert = Alert(
                id=f"fatigue_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.FATIGUE_DETECTED,
                severity=severity,
                title=f"Fatigue Detected: {severity.value.upper()}",
                message=f"Multiple fatigue indicators detected: {len(fatigue_indicators)} factors",
                timestamp=datetime.now(),
                metrics=metrics,
                recommendations=self._get_fatigue_recommendations(fatigue_indicators, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Fatigue check failed: {e}")
            return None
    
    def _check_sleep_deprivation(self, features: Dict) -> Optional[Alert]:
        """Check for sleep deprivation alerts."""
        try:
            if 'sleep_debt' not in features:
                return None
            
            sleep_debt = features['sleep_debt']
            
            # Determine severity
            severity = None
            for sev in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
                threshold = self.thresholds.get_threshold('sleep_debt', sev)
                if threshold and sleep_debt >= threshold:
                    severity = sev
                    break
            
            if not severity:
                return None
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.SLEEP_DEPRIVATION):
                return None
            
            alert = Alert(
                id=f"sleep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.SLEEP_DEPRIVATION,
                severity=severity,
                title=f"Sleep Deprivation: {severity.value.upper()}",
                message=f"Sleep debt of {sleep_debt:.1f} hours detected",
                timestamp=datetime.now(),
                metrics={'sleep_debt': sleep_debt},
                recommendations=self._get_sleep_recommendations(sleep_debt, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Sleep deprivation check failed: {e}")
            return None
    
    def _check_error_rate(self, features: Dict) -> Optional[Alert]:
        """Check for error rate spikes."""
        try:
            if 'error_rate' not in features:
                return None
            
            error_rate = features['error_rate']
            
            # Determine severity
            severity = None
            for sev in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
                threshold = self.thresholds.get_threshold('error_rate', sev)
                if threshold and error_rate >= threshold:
                    severity = sev
                    break
            
            if not severity:
                return None
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.ERROR_RATE_SPIKE):
                return None
            
            alert = Alert(
                id=f"error_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.ERROR_RATE_SPIKE,
                severity=severity,
                title=f"Error Rate Spike: {severity.value.upper()}",
                message=f"Error rate of {error_rate*100:.1f}% detected",
                timestamp=datetime.now(),
                metrics={'error_rate': error_rate},
                recommendations=self._get_error_rate_recommendations(error_rate, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Error rate check failed: {e}")
            return None
    
    def _check_cognitive_overload(self, features: Dict) -> Optional[Alert]:
        """Check for cognitive overload alerts."""
        try:
            if 'cognitive_load_score' not in features:
                return None
            
            cog_load = features['cognitive_load_score']
            
            # Determine severity
            severity = None
            for sev in [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW]:
                threshold = self.thresholds.get_threshold('cognitive_load', sev)
                if threshold and cog_load >= threshold:
                    severity = sev
                    break
            
            if not severity:
                return None
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.COGNITIVE_OVERLOAD):
                return None
            
            alert = Alert(
                id=f"cognitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.COGNITIVE_OVERLOAD,
                severity=severity,
                title=f"Cognitive Overload: {severity.value.upper()}",
                message=f"Cognitive load score of {cog_load:.2f} detected",
                timestamp=datetime.now(),
                metrics={'cognitive_load_score': cog_load},
                recommendations=self._get_cognitive_overload_recommendations(cog_load, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Cognitive overload check failed: {e}")
            return None
    
    def _check_anomalies(self, features: Dict) -> Optional[Alert]:
        """Check for statistical anomalies."""
        try:
            anomalies = []
            
            # Check for anomalies in key metrics using baseline models
            key_metrics = ['wpm', 'error_rate', 'sentiment_score', 'cognitive_load_score']
            
            for metric in key_metrics:
                if metric in features:
                    value = features[metric]
                    
                    # Simple anomaly detection (could be enhanced with trained models)
                    if metric == 'wpm' and value < 10:  # Very slow typing
                        anomalies.append((metric, value, "extremely_low"))
                    elif metric == 'error_rate' and value > 0.4:  # Very high error rate
                        anomalies.append((metric, value, "extremely_high"))
                    elif metric == 'sentiment_score' and value < -0.8:  # Very negative sentiment
                        anomalies.append((metric, value, "extremely_negative"))
                    elif metric == 'cognitive_load_score' and value > 0.95:  # Near maximum cognitive load
                        anomalies.append((metric, value, "extremely_high"))
            
            if not anomalies:
                return None
            
            # Determine severity based on number and type of anomalies
            if len(anomalies) >= 2 or any("extremely" in anomaly[2] for anomaly in anomalies):
                severity = AlertSeverity.HIGH
            else:
                severity = AlertSeverity.MEDIUM
            
            # Check cooldown
            if self._is_in_cooldown(AlertType.ANOMALY_DETECTED):
                return None
            
            metrics = {metric: value for metric, value, _ in anomalies}
            
            alert = Alert(
                id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type=AlertType.ANOMALY_DETECTED,
                severity=severity,
                title=f"Anomaly Detected: {severity.value.upper()}",
                message=f"Statistical anomalies detected in {len(anomalies)} metrics",
                timestamp=datetime.now(),
                metrics=metrics,
                recommendations=self._get_anomaly_recommendations(anomalies, severity)
            )
            
            return alert
            
        except Exception as e:
            print(f"Anomaly check failed: {e}")
            return None
    
    def _is_in_cooldown(self, alert_type: AlertType) -> bool:
        """Check if alert type is in cooldown period."""
        if alert_type not in self.alert_cooldown:
            return False
        
        cooldown_period = self.alert_cooldown[alert_type]
        
        # Find last alert of this type
        for alert in reversed(self.active_alerts):
            if alert.type == alert_type:
                time_since_last = datetime.now() - alert.timestamp
                return time_since_last < cooldown_period
        
        return False
    
    def _filter_alerts_by_cooldown(self, alerts: List[Alert]) -> List[Alert]:
        """Filter alerts based on cooldown periods."""
        filtered = []
        
        for alert in alerts:
            if not self._is_in_cooldown(alert.type):
                filtered.append(alert)
        
        return filtered
    
    def _get_performance_decline_recommendations(self, decline_rate: float, severity: AlertSeverity) -> List[str]:
        """Get recommendations for performance decline."""
        recommendations = []
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("STOP: Take immediate break - performance is severely declining")
            recommendations.append("Consider ending work for the day")
            recommendations.append("Review workload and priorities")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("Take a 15-20 minute break immediately")
            recommendations.append("Reduce task complexity")
            recommendations.append("Check for environmental factors (lighting, noise, etc.)")
        elif severity == AlertSeverity.MEDIUM:
            recommendations.append("Take a short break and stretch")
            recommendations.append("Hydrate and have a healthy snack")
            recommendations.append("Review current task difficulty")
        else:  # LOW
            recommendations.append("Consider taking a brief break")
            recommendations.append("Check your posture and ergonomics")
        
        return recommendations
    
    def _get_fatigue_recommendations(self, indicators: List, severity: AlertSeverity) -> List[str]:
        """Get recommendations for fatigue."""
        recommendations = []
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("IMMEDIATE ACTION: Stop working and rest")
            recommendations.append("Take a 30+ minute break or nap if possible")
            recommendations.append("Do not attempt complex tasks")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("Take a 20 minute break now")
            recommendations.append("Switch to simpler, less demanding tasks")
            recommendations.append("Avoid important decisions until rested")
        else:
            recommendations.append("Take a 10-15 minute break")
            recommendations.append("Practice relaxation techniques")
            recommendations.append("Consider ending work early today")
        
        # Specific recommendations based on indicators
        for indicator_type, value in indicators:
            if indicator_type == "typing_speed_decline":
                recommendations.append("Your typing speed has decreased significantly")
            elif indicator_type == "high_error_rate":
                recommendations.append("Double-check important work - error rate is elevated")
            elif indicator_type == "high_cognitive_load":
                recommendations.append("Reduce multitasking and focus on one task")
        
        return recommendations
    
    def _get_sleep_recommendations(self, sleep_debt: float, severity: AlertSeverity) -> List[str]:
        """Get recommendations for sleep deprivation."""
        recommendations = []
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("CRITICAL: Prioritize sleep immediately")
            recommendations.append("Consider taking a sick day to recover")
            recommendations.append("Avoid caffeine and screens before bedtime")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("Significant sleep debt detected - prioritize rest")
            recommendations.append("End work early today and get extra sleep")
            recommendations.append("Maintain consistent sleep schedule")
        else:
            recommendations.append("Mild sleep debt - aim for earlier bedtime")
            recommendations.append("Improve sleep hygiene practices")
            recommendations.append("Consider short power nap if possible")
        
        return recommendations
    
    def _get_error_rate_recommendations(self, error_rate: float, severity: AlertSeverity) -> List[str]:
        """Get recommendations for high error rate."""
        recommendations = []
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("CRITICAL: Error rate is extremely high")
            recommendations.append("Stop and review work before continuing")
            recommendations.append("Take a break and return with fresh eyes")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("High error rate detected - slow down")
            recommendations.append("Double-check all important work")
            recommendations.append("Consider switching to less demanding tasks")
        else:
            recommendations.append("Error rate is elevated - be more careful")
            recommendations.append("Take short breaks between tasks")
            recommendations.append("Focus on accuracy over speed")
        
        return recommendations
    
    def _get_cognitive_overload_recommendations(self, cog_load: float, severity: AlertSeverity) -> List[str]:
        """Get recommendations for cognitive overload."""
        recommendations = []
        
        if severity == AlertSeverity.CRITICAL:
            recommendations.append("CRITICAL: Cognitive overload detected")
            recommendations.append("STOP and take immediate break")
            recommendations.append("Simplify your environment and tasks")
        elif severity == AlertSeverity.HIGH:
            recommendations.append("High cognitive load - reduce complexity")
            recommendations.append("Focus on one task at a time")
            recommendations.append("Take regular breaks to reset")
        else:
            recommendations.append("Cognitive load is elevated")
            recommendations.append("Minimize distractions")
            recommendations.append("Break complex tasks into smaller steps")
        
        return recommendations
    
    def _get_anomaly_recommendations(self, anomalies: List, severity: AlertSeverity) -> List[str]:
        """Get recommendations for anomalies."""
        recommendations = []
        
        recommendations.append("Unusual patterns detected in your metrics")
        recommendations.append("Review your current work conditions")
        
        if severity == AlertSeverity.HIGH:
            recommendations.append("Multiple anomalies detected - take a break")
            recommendations.append("Check for environmental or health factors")
        
        # Specific recommendations based on anomaly types
        for metric, value, anomaly_type in anomalies:
            if metric == 'wpm' and "low" in anomaly_type:
                recommendations.append("Typing speed is unusually low - possible fatigue")
            elif metric == 'error_rate' and "high" in anomaly_type:
                recommendations.append("Error rate is unusually high - slow down and focus")
            elif metric == 'sentiment_score' and "negative" in anomaly_type:
                recommendations.append("Sentiment is unusually negative - consider stress management")
            elif metric == 'cognitive_load_score' and "high" in anomaly_type:
                recommendations.append("Cognitive load is extremely high - reduce task complexity")
        
        return recommendations
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.active_alerts.remove(alert)
                return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = self.active_alerts.copy()
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alert activity."""
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        
        # Count by severity
        severity_counts = {sev.value: 0 for sev in AlertSeverity}
        for alert in self.active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = {atype.value: 0 for atype in AlertType}
        for alert in self.active_alerts:
            type_counts[alert.type.value] += 1
        
        # Recent activity (last 24 hours)
        recent_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > recent_time]
        
        return {
            'total_alerts_generated': total_alerts,
            'active_alerts': active_alerts,
            'alerts_by_severity': severity_counts,
            'alerts_by_type': type_counts,
            'recent_alerts_24h': len(recent_alerts),
            'last_alert_time': self.alert_history[-1].timestamp.isoformat() if self.alert_history else None
        }


class AlertManager:
    """Main alert management system."""
    
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.generator = AlertGenerator(db_path)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Notification settings
        self.notification_settings = {
            'enable_desktop_notifications': True,
            'enable_email_notifications': False,
            'email_address': '',
            'minimum_severity': AlertSeverity.MEDIUM
        }
    
    def start_monitoring(self, check_interval_minutes: int = 5):
        """Start continuous alert monitoring."""
        if self.monitoring_active:
            print("Alert monitoring is already active")
            return
        
        def monitoring_loop():
            self.monitoring_active = True
            print(f"Started alert monitoring with {check_interval_minutes} minute intervals")
            
            while self.monitoring_active:
                try:
                    # Check for alerts
                    new_alerts = self.generator.check_all_alerts()
                    
                    # Process new alerts
                    for alert in new_alerts:
                        self._process_alert(alert)
                    
                    # Clean up old resolved alerts
                    self._cleanup_old_alerts()
                    
                    time.sleep(check_interval_minutes * 60)
                    
                except Exception as e:
                    print(f"Alert monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        print("Alert monitoring stopped")
    
    def _process_alert(self, alert: Alert):
        """Process a new alert."""
        print(f"ALERT: {alert.title} - {alert.message}")
        
        # Check if alert meets minimum severity threshold
        min_severity = self.notification_settings['minimum_severity']
        severity_levels = {
            AlertSeverity.LOW: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.HIGH: 3,
            AlertSeverity.CRITICAL: 4
        }
        
        if severity_levels[alert.severity] >= severity_levels[min_severity]:
            # Send notifications
            if self.notification_settings['enable_desktop_notifications']:
                self._send_desktop_notification(alert)
            
            if self.notification_settings['enable_email_notifications'] and self.notification_settings['email_address']:
                self._send_email_notification(alert)
    
    def _send_desktop_notification(self, alert: Alert):
        """Send desktop notification (placeholder)."""
        try:
            # In a real implementation, this would use a desktop notification library
            print(f"[DESKTOP NOTIFICATION] {alert.title}: {alert.message}")
        except Exception as e:
            print(f"Failed to send desktop notification: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification (placeholder)."""
        try:
            # In a real implementation, this would configure and send an email
            print(f"[EMAIL NOTIFICATION] To: {self.notification_settings['email_address']}")
            print(f"Subject: {alert.title}")
            print(f"Message: {alert.message}")
        except Exception as e:
            print(f"Failed to send email notification: {e}")
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Remove old resolved alerts from history
        self.generator.alert_history = [
            alert for alert in self.generator.alert_history 
            if alert.timestamp > cutoff_time or not alert.resolved
        ]
    
    def get_dashboard_data(self) -> Dict:
        """Get alert data for dashboard display."""
        active_alerts = self.generator.get_active_alerts()
        alert_summary = self.generator.get_alert_summary()
        
        return {
            'active_alerts': [
                {
                    'id': alert.id,
                    'type': alert.type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'metrics': alert.metrics,
                    'recommendations': alert.recommendations,
                    'acknowledged': alert.acknowledged
                }
                for alert in active_alerts
            ],
            'summary': alert_summary,
            'monitoring_active': self.monitoring_active
        }


if __name__ == "__main__":
    # Test alert system
    print("Testing alert system...")
    
    manager = AlertManager()
    
    # Test alert generation
    print("\n=== Testing Alert Generation ===")
    alerts = manager.generator.check_all_alerts()
    print(f"Generated {len(alerts)} alerts")
    
    for alert in alerts:
        print(f"- {alert.title}: {alert.message}")
    
    # Get alert summary
    print("\n=== Alert Summary ===")
    summary = manager.generator.get_alert_summary()
    print(json.dumps(summary, indent=2))
    
    # Test dashboard data
    print("\n=== Dashboard Data ===")
    dashboard = manager.get_dashboard_data()
    print(json.dumps(dashboard, indent=2, default=str))
    
    print("\nAlert system test completed!")
