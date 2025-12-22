"""
Trend Analysis Module
Analyzes trends in performance, burnout risk, and behavioral patterns over time.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from ml.feature_engineering import FeatureEngineer
from ml.inference import RealTimePredictor


class TrendAnalyzer:
    """Analyzes trends in performance and burnout metrics over time."""
    
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer(db_path)
        self.predictor = RealTimePredictor(db_path)
        
        # Create plots directory
        os.makedirs("plots", exist_ok=True)
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical feature data for trend analysis."""
        try:
            # Create feature matrix for the specified period
            feature_matrix, _ = self.feature_engineer.create_feature_matrix(days * 24)
            
            if feature_matrix.empty:
                print("No historical data available")
                return pd.DataFrame()
            
            # Add derived metrics
            feature_matrix = self._add_derived_metrics(feature_matrix)
            
            return feature_matrix
            
        except Exception as e:
            print(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics for trend analysis."""
        try:
            # Rolling averages for key metrics
            key_metrics = ['cognitive_load_score', 'wpm', 'error_rate', 'sentiment_score']
            
            for metric in key_metrics:
                if metric in df.columns:
                    # Different window sizes for different trend perspectives
                    df[f'{metric}_ma_6h'] = df[metric].rolling(window=6, min_periods=1).mean()
                    df[f'{metric}_ma_24h'] = df[metric].rolling(window=24, min_periods=1).mean()
                    df[f'{metric}_ma_7d'] = df[metric].rolling(window=168, min_periods=1).mean()  # 7 days
            
            # Trend indicators (rate of change)
            for metric in key_metrics:
                if metric in df.columns:
                    df[f'{metric}_trend_6h'] = df[metric].diff(periods=6)
                    df[f'{metric}_trend_24h'] = df[metric].diff(periods=24)
            
            # Performance composite score
            performance_metrics = ['wpm', 'focus_ratio', 'sentiment_score']
            available_perf_metrics = [m for m in performance_metrics if m in df.columns]
            
            if available_perf_metrics:
                # Normalize metrics to 0-1 scale
                for metric in available_perf_metrics:
                    if metric == 'wpm':
                        # Normalize WPM (assuming 0-100 WPM range)
                        df[f'{metric}_norm'] = np.clip(df[metric] / 100, 0, 1)
                    else:
                        # Other metrics are already normalized
                        df[f'{metric}_norm'] = np.clip(df[metric], 0, 1)
                
                norm_metrics = [f'{m}_norm' for m in available_perf_metrics]
                df['performance_composite'] = df[norm_metrics].mean(axis=1)
            
            # Burnout risk composite score
            burnout_metrics = ['cognitive_load_score', 'error_rate', 'sleep_debt']
            available_burnout_metrics = [m for m in burnout_metrics if m in df.columns]
            
            if available_burnout_metrics:
                norm_burnout_metrics = []
                for metric in available_burnout_metrics:
                    if metric == 'error_rate':
                        # Normalize error rate (assuming 0-0.5 range)
                        df[f'{metric}_norm'] = np.clip(df[metric] / 0.5, 0, 1)
                    else:
                        df[f'{metric}_norm'] = np.clip(df[metric], 0, 1)
                    norm_burnout_metrics.append(f'{metric}_norm')
                
                df['burnout_composite'] = df[norm_burnout_metrics].mean(axis=1)
            
            return df
            
        except Exception as e:
            print(f"Failed to add derived metrics: {e}")
            return df
    
    def analyze_performance_trends(self, days: int = 30) -> Dict:
        """Analyze performance trends over time."""
        print(f"Analyzing performance trends for last {days} days...")
        
        df = self.get_historical_data(days)
        
        if df.empty:
            return {"error": "No data available for trend analysis"}
        
        analysis = {
            'period_days': days,
            'data_points': len(df),
            'analysis_date': datetime.now().isoformat()
        }
        
        try:
            # Overall trend analysis
            if 'performance_composite' in df.columns:
                perf_data = df['performance_composite'].dropna()
                
                if len(perf_data) > 10:
                    # Linear trend
                    x = np.arange(len(perf_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, perf_data)
                    
                    analysis['overall_performance_trend'] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'trend_direction': 'improving' if slope > 0 else 'declining',
                        'trend_strength': 'strong' if abs(slope) > 0.01 else 'moderate' if abs(slope) > 0.005 else 'weak'
                    }
                    
                    # Recent trend (last 7 days)
                    recent_data = perf_data.tail(168)  # Last 7 days
                    if len(recent_data) > 10:
                        x_recent = np.arange(len(recent_data))
                        slope_recent, _, r_value_recent, p_value_recent, _ = stats.linregress(x_recent, recent_data)
                        
                        analysis['recent_performance_trend'] = {
                            'slope': slope_recent,
                            'r_squared': r_value_recent ** 2,
                            'trend_direction': 'improving' if slope_recent > 0 else 'declining',
                            'trend_strength': 'strong' if abs(slope_recent) > 0.01 else 'moderate' if abs(slope_recent) > 0.005 else 'weak'
                        }
            
            # Component-wise trends
            key_metrics = ['wpm', 'focus_ratio', 'sentiment_score', 'cognitive_load_score', 'error_rate']
            analysis['component_trends'] = {}
            
            for metric in key_metrics:
                if metric in df.columns:
                    metric_data = df[metric].dropna()
                    
                    if len(metric_data) > 10:
                        x = np.arange(len(metric_data))
                        slope, _, r_value, p_value, _ = stats.linregress(x, metric_data)
                        
                        analysis['component_trends'][metric] = {
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'trend_direction': 'improving' if slope > 0 else 'declining',
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
            
            # Pattern detection
            analysis['patterns'] = self._detect_patterns(df)
            
            # Seasonality analysis
            analysis['seasonality'] = self._analyze_seasonality(df)
            
            return analysis
            
        except Exception as e:
            analysis['error'] = str(e)
            return analysis
    
    def analyze_burnout_trends(self, days: int = 30) -> Dict:
        """Analyze burnout risk trends over time."""
        print(f"Analyzing burnout trends for last {days} days...")
        
        df = self.get_historical_data(days)
        
        if df.empty:
            return {"error": "No data available for trend analysis"}
        
        analysis = {
            'period_days': days,
            'data_points': len(df),
            'analysis_date': datetime.now().isoformat()
        }
        
        try:
            # Burnout composite trend
            if 'burnout_composite' in df.columns:
                burnout_data = df['burnout_composite'].dropna()
                
                if len(burnout_data) > 10:
                    x = np.arange(len(burnout_data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, burnout_data)
                    
                    analysis['overall_burnout_trend'] = {
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'p_value': p_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'trend_strength': 'strong' if abs(slope) > 0.01 else 'moderate' if abs(slope) > 0.005 else 'weak'
                    }
                    
                    # Current risk level
                    current_risk = burnout_data.iloc[-1]
                    if current_risk < 0.3:
                        risk_level = "low"
                    elif current_risk < 0.6:
                        risk_level = "medium"
                    elif current_risk < 0.8:
                        risk_level = "high"
                    else:
                        risk_level = "critical"
                    
                    analysis['current_risk_level'] = risk_level
                    analysis['current_risk_score'] = float(current_risk)
            
            # Individual burnout factors
            burnout_factors = ['cognitive_load_score', 'error_rate', 'sleep_debt', 'app_fragmentation']
            analysis['factor_trends'] = {}
            
            for factor in burnout_factors:
                if factor in df.columns:
                    factor_data = df[factor].dropna()
                    
                    if len(factor_data) > 10:
                        x = np.arange(len(factor_data))
                        slope, _, r_value, p_value, _ = stats.linregress(x, factor_data)
                        
                        analysis['factor_trends'][factor] = {
                            'slope': slope,
                            'r_squared': r_value ** 2,
                            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                            'significance': 'significant' if p_value < 0.05 else 'not_significant',
                            'current_value': float(factor_data.iloc[-1]) if len(factor_data) > 0 else None
                        }
            
            # Risk escalation detection
            analysis['risk_escalation'] = self._detect_risk_escalation(df)
            
            # Recovery periods
            analysis['recovery_periods'] = self._detect_recovery_periods(df)
            
            return analysis
            
        except Exception as e:
            analysis['error'] = str(e)
            return analysis
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect recurring patterns in the data."""
        patterns = {}
        
        try:
            if 'performance_composite' in df.columns:
                perf_data = df['performance_composite'].dropna()
                
                if len(perf_data) > 24:  # Need at least 1 day of data
                    # Daily patterns (hourly)
                    hourly_avg = perf_data.groupby(perf_data.index.hour).mean()
                    
                    # Find peak and low performance hours
                    peak_hour = hourly_avg.idxmax()
                    low_hour = hourly_avg.idxmin()
                    
                    patterns['daily'] = {
                        'peak_performance_hour': int(peak_hour),
                        'lowest_performance_hour': int(low_hour),
                        'peak_value': float(hourly_avg.max()),
                        'low_value': float(hourly_avg.min()),
                        'daily_variation': float(hourly_avg.max() - hourly_avg.min())
                    }
                    
                    # Weekly patterns
                    if len(perf_data) > 168:  # Need at least 1 week
                        weekly_avg = perf_data.groupby(perf_data.index.dayofweek).mean()
                        peak_day = weekly_avg.idxmax()
                        low_day = weekly_avg.idxmin()
                        
                        patterns['weekly'] = {
                            'peak_performance_day': int(peak_day),  # 0=Monday, 6=Sunday
                            'lowest_performance_day': int(low_day),
                            'peak_day_value': float(weekly_avg.max()),
                            'low_day_value': float(weekly_avg.min())
                        }
            
            # Detect performance cycles
            if 'performance_composite' in df.columns:
                perf_data = df['performance_composite'].dropna()
                
                # Find peaks (high performance periods)
                peaks, _ = find_peaks(perf_data, distance=24)  # Peaks at least 24 hours apart
                valleys, _ = find_peaks(-perf_data, distance=24)  # Valleys (low performance)
                
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks)
                    patterns['performance_cycles'] = {
                        'peak_count': len(peaks),
                        'valley_count': len(valleys),
                        'avg_peak_interval_hours': float(np.mean(peak_intervals)) if len(peak_intervals) > 0 else None,
                        'cycle_regularity': 'regular' if np.std(peak_intervals) < 12 else 'irregular' if len(peak_intervals) > 0 else 'insufficient_data'
                    }
        
        except Exception as e:
            patterns['error'] = str(e)
        
        return patterns
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal patterns in the data."""
        seasonality = {}
        
        try:
            if 'performance_composite' in df.columns:
                perf_data = df['performance_composite'].dropna()
                
                if len(perf_data) > 168:  # Need at least 1 week
                    # Hour of day seasonality
                    hourly_performance = perf_data.groupby(perf_data.index.hour).mean()
                    seasonality['hourly'] = {
                        'values': hourly_performance.tolist(),
                        'hours': list(hourly_performance.index),
                        'peak_hour': int(hourly_performance.idxmax()),
                        'lowest_hour': int(hourly_performance.idxmin())
                    }
                    
                    # Day of week seasonality
                    if len(perf_data) > 168 * 2:  # Need at least 2 weeks
                        weekly_performance = perf_data.groupby(perf_data.index.dayofweek).mean()
                        seasonality['weekly'] = {
                            'values': weekly_performance.tolist(),
                            'days': list(weekly_performance.index),  # 0=Monday, 6=Sunday
                            'peak_day': int(weekly_performance.idxmax()),
                            'lowest_day': int(weekly_performance.idxmin())
                        }
            
            # Sleep patterns seasonality
            if 'sleep_quality_score' in df.columns:
                sleep_data = df['sleep_quality_score'].dropna()
                
                if len(sleep_data) > 7:  # Need at least 1 week
                    hourly_sleep = sleep_data.groupby(sleep_data.index.hour).mean()
                    seasonality['sleep_hourly'] = {
                        'values': hourly_sleep.tolist(),
                        'hours': list(hourly_sleep.index),
                        'best_sleep_hour': int(hourly_sleep.idxmax()),
                        'worst_sleep_hour': int(hourly_sleep.idxmin())
                    }
        
        except Exception as e:
            seasonality['error'] = str(e)
        
        return seasonality
    
    def _detect_risk_escalation(self, df: pd.DataFrame) -> Dict:
        """Detect periods of rapid burnout risk escalation."""
        escalation = {}
        
        try:
            if 'burnout_composite' in df.columns:
                burnout_data = df['burnout_composite'].dropna()
                
                if len(burnout_data) > 24:
                    # Calculate rate of change (6-hour windows)
                    rate_of_change = burnout_data.diff(periods=6)
                    
                    # Find periods of rapid increase
                    rapid_increase_threshold = rate_of_change.quantile(0.9)  # Top 10% of increases
                    escalation_periods = rate_of_change[rate_of_change > rapid_increase_threshold]
                    
                    if len(escalation_periods) > 0:
                        escalation['recent_escalations'] = {
                            'count': len(escalation_periods),
                            'avg_increase_rate': float(escalation_periods.mean()),
                            'max_increase_rate': float(escalation_periods.max()),
                            'last_escalation': escalation_periods.index[-1].isoformat() if len(escalation_periods) > 0 else None
                        }
                    
                    # Check if currently in escalation
                    if len(rate_of_change) > 0:
                        current_rate = rate_of_change.iloc[-1]
                        escalation['currently_escalating'] = current_rate > rapid_increase_threshold * 0.5
                        escalation['current_rate'] = float(current_rate)
        
        except Exception as e:
            escalation['error'] = str(e)
        
        return escalation
    
    def _detect_recovery_periods(self, df: pd.DataFrame) -> Dict:
        """Detect periods of recovery (decreasing burnout risk)."""
        recovery = {}
        
        try:
            if 'burnout_composite' in df.columns:
                burnout_data = df['burnout_composite'].dropna()
                
                if len(burnout_data) > 24:
                    # Calculate rate of decrease
                    rate_of_change = burnout_data.diff(periods=6)
                    
                    # Find periods of recovery (negative change)
                    recovery_periods = rate_of_change[rate_of_change < -0.01]  # Significant decrease
                    
                    if len(recovery_periods) > 0:
                        recovery['recovery_periods'] = {
                            'count': len(recovery_periods),
                            'avg_recovery_rate': float(recovery_periods.mean()),
                            'max_recovery_rate': float(recovery_periods.min()),  # Most negative
                            'last_recovery': recovery_periods.index[-1].isoformat() if len(recovery_periods) > 0 else None
                        }
                    
                    # Recovery efficiency (how much burnout decreases per hour of rest)
                    if 'sleep_quality_score' in df.columns:
                        # Correlate sleep quality with burnout reduction
                        sleep_data = df['sleep_quality_score'].dropna()
                        
                        # Align datasets
                        common_index = burnout_data.index.intersection(sleep_data.index)
                        if len(common_index) > 10:
                            burnout_aligned = burnout_data.loc[common_index]
                            sleep_aligned = sleep_data.loc[common_index]
                            
                            # Calculate correlation
                            correlation = np.corrcoef(burnout_aligned, sleep_aligned)[0, 1]
                            recovery['sleep_burnout_correlation'] = float(correlation)
                            recovery['recovery_efficiency'] = 'high' if correlation < -0.3 else 'moderate' if correlation < -0.1 else 'low'
        
        except Exception as e:
            recovery['error'] = str(e)
        
        return recovery
    
    def generate_trend_report(self, days: int = 30) -> Dict:
        """Generate comprehensive trend analysis report."""
        print(f"Generating {day}-day trend report...")
        
        # Get performance and burnout analyses
        performance_analysis = self.analyze_performance_trends(days)
        burnout_analysis = self.analyze_burnout_trends(days)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'period_days': days,
                'data_source': 'performance_data.db'
            },
            'performance_trends': performance_analysis,
            'burnout_trends': burnout_analysis
        }
        
        # Add insights and recommendations
        report['insights'] = self._generate_insights(performance_analysis, burnout_analysis)
        report['recommendations'] = self._generate_recommendations(performance_analysis, burnout_analysis)
        
        # Save report
        report_file = f"trend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Trend report saved to {report_file}")
        
        return report
    
    def _generate_insights(self, perf_analysis: Dict, burnout_analysis: Dict) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        try:
            # Performance insights
            if 'overall_performance_trend' in perf_analysis:
                trend = perf_analysis['overall_performance_trend']
                if trend['trend_direction'] == 'declining' and trend['trend_strength'] in ['strong', 'moderate']:
                    insights.append("Performance is showing a significant declining trend")
                elif trend['trend_direction'] == 'improving' and trend['trend_strength'] in ['strong', 'moderate']:
                    insights.append("Performance is showing a significant improving trend")
            
            # Burnout insights
            if 'current_risk_level' in burnout_analysis:
                risk_level = burnout_analysis['current_risk_level']
                if risk_level in ['high', 'critical']:
                    insights.append(f"Current burnout risk is {risk_level} - immediate attention needed")
                elif risk_level == 'medium':
                    insights.append("Burnout risk is elevated - preventive measures recommended")
            
            # Pattern insights
            if 'patterns' in perf_analysis and 'daily' in perf_analysis['patterns']:
                daily = perf_analysis['patterns']['daily']
                if daily['daily_variation'] > 0.3:
                    insights.append("Large daily performance variation detected - consider energy management")
            
            # Escalation insights
            if 'risk_escalation' in burnout_analysis and burnout_analysis['risk_escalation'].get('currently_escalating'):
                insights.append("Burnout risk is currently escalating - take immediate action")
            
            # Recovery insights
            if 'recovery_periods' in burnout_analysis and burnout_analysis['recovery_periods'].get('recovery_efficiency') == 'low':
                insights.append("Recovery efficiency is low - focus on improving sleep quality")
        
        except Exception as e:
            insights.append(f"Insight generation error: {e}")
        
        return insights
    
    def _generate_recommendations(self, perf_analysis: Dict, burnout_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on trends."""
        recommendations = []
        
        try:
            # Performance recommendations
            if 'overall_performance_trend' in perf_analysis:
                trend = perf_analysis['overall_performance_trend']
                if trend['trend_direction'] == 'declining':
                    recommendations.append("Schedule regular breaks to prevent performance decline")
                    recommendations.append("Review and optimize your work environment")
            
            # Burnout recommendations
            if 'current_risk_level' in burnout_analysis:
                risk_level = burnout_analysis['current_risk_level']
                if risk_level == 'critical':
                    recommendations.append("IMMEDIATE ACTION: Take a break and consider ending work for the day")
                    recommendations.append("Contact support systems - don't face this alone")
                elif risk_level == 'high':
                    recommendations.append("Reduce workload and prioritize essential tasks only")
                    recommendations.append("Schedule extended break periods")
                elif risk_level == 'medium':
                    recommendations.append("Increase break frequency and duration")
                    recommendations.append("Focus on stress management techniques")
            
            # Pattern-based recommendations
            if 'patterns' in perf_analysis and 'daily' in perf_analysis['patterns']:
                daily = perf_analysis['patterns']['daily']
                peak_hour = daily['peak_performance_hour']
                recommendations.append(f"Schedule important tasks around hour {peak_hour} when performance peaks")
            
            # Factor-specific recommendations
            if 'factor_trends' in burnout_analysis:
                for factor, trend in burnout_analysis['factor_trends'].items():
                    if trend['trend_direction'] == 'increasing' and trend['significance'] == 'significant':
                        if factor == 'cognitive_load_score':
                            recommendations.append("Reduce multitasking and cognitive overhead")
                        elif factor == 'error_rate':
                            recommendations.append("Slow down and focus on accuracy over speed")
                        elif factor == 'sleep_debt':
                            recommendations.append("Prioritize sleep - aim for 7-9 hours per night")
                        elif factor == 'app_fragmentation':
                            recommendations.append("Minimize app switching and distractions")
        
        except Exception as e:
            recommendations.append(f"Recommendation generation error: {e}")
        
        return recommendations
    
    def create_trend_visualizations(self, days: int = 30) -> Dict[str, str]:
        """Create visualization plots for trends."""
        print(f"Creating trend visualizations for last {days} days...")
        
        df = self.get_historical_data(days)
        
        if df.empty:
            return {"error": "No data available for visualization"}
        
        plot_files = {}
        
        try:
            # Performance trend plot
            if 'performance_composite' in df.columns:
                plt.figure(figsize=(12, 6))
                
                # Plot performance with moving averages
                plt.plot(df.index, df['performance_composite'], alpha=0.5, label='Hourly Performance')
                
                if 'performance_composite_ma_24h' in df.columns:
                    plt.plot(df.index, df['performance_composite_ma_24h'], linewidth=2, label='24h Average')
                
                if 'performance_composite_ma_7d' in df.columns:
                    plt.plot(df.index, df['performance_composite_ma_7d'], linewidth=2, label='7d Average')
                
                plt.xlabel('Date')
                plt.ylabel('Performance Score')
                plt.title(f'Performance Trend - Last {days} Days')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = "plots/performance_trend.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['performance_trend'] = filename
            
            # Burnout risk trend plot
            if 'burnout_composite' in df.columns:
                plt.figure(figsize=(12, 6))
                
                plt.plot(df.index, df['burnout_composite'], color='red', alpha=0.7, label='Burnout Risk')
                
                # Add risk level zones
                plt.axhspan(0, 0.3, alpha=0.1, color='green', label='Low Risk')
                plt.axhspan(0.3, 0.6, alpha=0.1, color='yellow', label='Medium Risk')
                plt.axhspan(0.6, 0.8, alpha=0.1, color='orange', label='High Risk')
                plt.axhspan(0.8, 1.0, alpha=0.1, color='red', label='Critical Risk')
                
                plt.xlabel('Date')
                plt.ylabel('Burnout Risk Score')
                plt.title(f'Burnout Risk Trend - Last {days} Days')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = "plots/burnout_trend.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['burnout_trend'] = filename
            
            # Component trends plot
            key_metrics = ['wpm', 'focus_ratio', 'sentiment_score', 'cognitive_load_score', 'error_rate']
            available_metrics = [m for m in key_metrics if m in df.columns]
            
            if available_metrics:
                fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 2*len(available_metrics)))
                if len(available_metrics) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(available_metrics):
                    axes[i].plot(df.index, df[metric], alpha=0.7)
                    axes[i].set_ylabel(metric.replace('_', ' ').title())
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_title(f'{metric.replace("_", " ").title()} Trend')
                
                plt.xlabel('Date')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = "plots/component_trends.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['component_trends'] = filename
            
            # Daily patterns heatmap
            if 'performance_composite' in df.columns and len(df) > 168:
                # Create heatmap of performance by hour and day of week
                df['hour'] = df.index.hour
                df['day_of_week'] = df.index.dayofweek
                
                pivot_table = df.groupby(['day_of_week', 'hour'])['performance_composite'].mean().unstack()
                
                plt.figure(figsize=(12, 6))
                sns.heatmap(pivot_table, cmap='RdYlGn', annot=False, cbar_kws={'label': 'Performance Score'})
                plt.xlabel('Hour of Day')
                plt.ylabel('Day of Week (0=Mon, 6=Sun)')
                plt.title('Performance Patterns by Hour and Day of Week')
                plt.tight_layout()
                
                filename = "plots/performance_heatmap.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['performance_heatmap'] = filename
        
        except Exception as e:
            plot_files['error'] = str(e)
        
        return plot_files


if __name__ == "__main__":
    # Test trend analysis
    print("Testing trend analysis...")
    
    analyzer = TrendAnalyzer()
    
    # Generate trend report
    report = analyzer.generate_trend_report(days=7)  # Use 7 days for testing
    print("\nTrend Analysis Report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Create visualizations
    plots = analyzer.create_trend_visualizations(days=7)
    print(f"\nVisualization plots created: {list(plots.keys())}")
    
    print("\nTrend analysis test completed!")
