"""
Analytics Module
Contains trend analysis and alert generation functionality.
"""

from .trend_analysis import TrendAnalyzer
from .alerts import AlertManager, Alert, AlertType, AlertSeverity

__all__ = [
    'TrendAnalyzer',
    'AlertManager',
    'Alert',
    'AlertType',
    'AlertSeverity'
]
