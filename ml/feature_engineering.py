"""
Feature Engineering Module
Transforms raw data into meaningful features for ML models.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureEngineer:
    def __init__(self, db_path: str = "data/performance_data.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def extract_system_features(self, hours: int = 24) -> pd.DataFrame:
        """Extract features from system metrics data."""
        conn = sqlite3.connect(self.db_path)
        
        # Get system metrics
        system_df = pd.read_sql_query('''
            SELECT timestamp, cpu_percent, memory_percent, active_processes,
                   battery_percent, power_plugged
            FROM system_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn)
        
        conn.close()
        
        if system_df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
        
        # Time-based features
        system_df['hour_of_day'] = system_df['timestamp'].dt.hour
        system_df['day_of_week'] = system_df['timestamp'].dt.dayofweek
        system_df['is_weekend'] = (system_df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Rolling window features
        windows = [5, 10, 20]  # Number of samples
        
        for window in windows:
            system_df[f'cpu_rolling_mean_{window}'] = system_df['cpu_percent'].rolling(window).mean()
            system_df[f'cpu_rolling_std_{window}'] = system_df['cpu_percent'].rolling(window).std()
            system_df[f'memory_rolling_mean_{window}'] = system_df['memory_percent'].rolling(window).mean()
            system_df[f'memory_rolling_std_{window}'] = system_df['memory_percent'].rolling(window).std()
        
        # Rate of change features
        system_df['cpu_rate_of_change'] = system_df['cpu_percent'].diff()
        system_df['memory_rate_of_change'] = system_df['memory_percent'].diff()
        
        # Cognitive load indicators
        system_df['cognitive_load_score'] = (
            system_df['cpu_percent'] * 0.4 + 
            system_df['memory_percent'] * 0.3 + 
            (system_df['active_processes'] / 100) * 0.3
        )
        
        return system_df
    
    def extract_app_usage_features(self, hours: int = 24) -> pd.DataFrame:
        """Extract features from app usage data."""
        conn = sqlite3.connect(self.db_path)
        
        # Get app usage data
        app_df = pd.read_sql_query('''
            SELECT timestamp, app_name, duration_seconds, is_switch, session_duration
            FROM app_usage 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn)
        
        conn.close()
        
        if app_df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        app_df['timestamp'] = pd.to_datetime(app_df['timestamp'])
        
        # Calculate focus and fragmentation metrics
        features = []
        
        # Group by hour windows
        app_df['hour'] = app_df['timestamp'].dt.floor('H')
        
        for hour, group in app_df.groupby('hour'):
            hour_features = {
                'timestamp': hour,
                'total_app_switches': group['is_switch'].sum(),
                'unique_apps': group['app_name'].nunique(),
                'avg_session_duration': group['session_duration'].mean(),
                'total_active_time': group['duration_seconds'].sum(),
                'app_fragmentation': group['app_name'].nunique() / (group['duration_seconds'].sum() / 3600) if group['duration_seconds'].sum() > 0 else 0,
                'switch_rate': group['is_switch'].sum() / (group['duration_seconds'].sum() / 3600) if group['duration_seconds'].sum() > 0 else 0
            }
            
            # Calculate app usage entropy
            app_durations = group.groupby('app_name')['duration_seconds'].sum()
            if len(app_durations) > 0:
                total_duration = app_durations.sum()
                probabilities = app_durations / total_duration
                entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
                hour_features['app_usage_entropy'] = entropy
            else:
                hour_features['app_usage_entropy'] = 0
            
            # Focus time (time in top 3 apps)
            if len(app_durations) > 0:
                top_apps_time = app_durations.nlargest(3).sum()
                hour_features['focus_ratio'] = top_apps_time / total_duration if total_duration > 0 else 0
            else:
                hour_features['focus_ratio'] = 0
            
            features.append(hour_features)
        
        return pd.DataFrame(features)
    
    def extract_typing_features(self, hours: int = 24) -> pd.DataFrame:
        """Extract features from typing metrics data."""
        conn = sqlite3.connect(self.db_path)
        
        # Get typing metrics
        typing_df = pd.read_sql_query('''
            SELECT timestamp, wpm, error_rate, avg_key_interval, 
                   backspace_count, typing_variance, session_duration
            FROM typing_metrics 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn)
        
        conn.close()
        
        if typing_df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        typing_df['timestamp'] = pd.to_datetime(typing_df['timestamp'])
        
        # Time-based features
        typing_df['hour_of_day'] = typing_df['timestamp'].dt.hour
        
        # Rolling window features for typing patterns
        windows = [3, 5, 10]
        
        for window in windows:
            typing_df[f'wpm_rolling_mean_{window}'] = typing_df['wpm'].rolling(window).mean()
            typing_df[f'wpm_rolling_std_{window}'] = typing_df['wpm'].rolling(window).std()
            typing_df[f'error_rate_rolling_mean_{window}'] = typing_df['error_rate'].rolling(window).mean()
            typing_df[f'typing_variance_rolling_mean_{window}'] = typing_df['typing_variance'].rolling(window).mean()
        
        # Fatigue indicators
        typing_df['typing_fatigue_score'] = (
            typing_df['error_rate'] * 0.4 + 
            typing_df['typing_variance'] * 0.3 + 
            (typing_df['backspace_count'] / typing_df['session_duration']) * 0.3
        ).fillna(0)
        
        # Performance degradation
        typing_df['wpm_trend'] = typing_df['wpm'].diff()
        typing_df['error_trend'] = typing_df['error_rate'].diff()
        
        return typing_df
    
    def extract_sleep_features(self, hours: int = 168) -> pd.DataFrame:  # 7 days
        """Extract features from sleep proxy data."""
        conn = sqlite3.connect(self.db_path)
        
        # Get sleep periods
        sleep_df = pd.read_sql_query('''
            SELECT sleep_start, sleep_end, duration_hours, sleep_quality_score
            FROM sleep_periods 
            WHERE sleep_start > datetime('now', '-{} hours')
            ORDER BY sleep_start
        '''.format(hours), conn)
        
        conn.close()
        
        if sleep_df.empty:
            return pd.DataFrame()
        
        # Convert timestamps
        sleep_df['sleep_start'] = pd.to_datetime(sleep_df['sleep_start'])
        if 'sleep_end' in sleep_df.columns and sleep_df['sleep_end'].notna().any():
            sleep_df['sleep_end'] = pd.to_datetime(sleep_df['sleep_end'])
        
        # Sleep regularity features
        sleep_df['sleep_start_hour'] = sleep_df['sleep_start'].dt.hour + sleep_df['sleep_start'].dt.minute / 60
        sleep_df['day_of_week'] = sleep_df['sleep_start'].dt.dayofweek
        
        # Calculate sleep debt
        optimal_sleep = 8.0
        sleep_df['sleep_debt'] = np.maximum(0, optimal_sleep - sleep_df['duration_hours'])
        
        # Sleep consistency
        if len(sleep_df) > 1:
            sleep_df['bedtime_consistency'] = sleep_df['sleep_start_hour'].rolling(window=3).std().fillna(0)
            sleep_df['duration_consistency'] = sleep_df['duration_hours'].rolling(window=3).std().fillna(0)
        else:
            sleep_df['bedtime_consistency'] = 0
            sleep_df['duration_consistency'] = 0
        
        return sleep_df
    
    def extract_journal_features(self, hours: int = 168) -> pd.DataFrame:  # 7 days
        """Extract features from journal entries."""
        conn = sqlite3.connect(self.db_path)
        
        # Get journal entries
        journal_df = pd.read_sql_query('''
            SELECT timestamp, mood_rating, stress_level, energy_level,
                   productivity_rating, sentiment_score, word_count
            FROM journal_entries 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp
        '''.format(hours), conn)
        
        conn.close()
        
        if journal_df.empty:
            return pd.DataFrame()
        
        # Convert timestamp
        journal_df['timestamp'] = pd.to_datetime(journal_df['timestamp'])
        
        # Time-based features
        journal_df['hour_of_day'] = journal_df['timestamp'].dt.hour
        journal_df['day_of_week'] = journal_df['timestamp'].dt.dayofweek
        
        # Rolling averages for mood trends
        windows = [3, 7]
        
        for window in windows:
            journal_df[f'mood_rolling_mean_{window}'] = journal_df['mood_rating'].rolling(window).mean()
            journal_df[f'stress_rolling_mean_{window}'] = journal_df['stress_level'].rolling(window).mean()
            journal_df[f'energy_rolling_mean_{window}'] = journal_df['energy_level'].rolling(window).mean()
            journal_df[f'sentiment_rolling_mean_{window}'] = journal_df['sentiment_score'].rolling(window).mean()
        
        # Mental health indicators
        journal_df['mental_health_score'] = (
            (journal_df['mood_rating'] / 10) * 0.3 +  # Normalize to 0-1
            ((10 - journal_df['stress_level']) / 10) * 0.3 +  # Invert stress
            (journal_df['energy_level'] / 10) * 0.2 +
            ((journal_df['sentiment_score'] + 1) / 2) * 0.2  # Normalize sentiment to 0-1
        ).fillna(0)
        
        # Trend features
        journal_df['mood_trend'] = journal_df['mood_rating'].diff()
        journal_df['stress_trend'] = journal_df['stress_level'].diff()
        journal_df['energy_trend'] = journal_df['energy_level'].diff()
        
        return journal_df
    
    def create_feature_matrix(self, hours: int = 24) -> Tuple[pd.DataFrame, List[str]]:
        """Create comprehensive feature matrix from all data sources."""
        # Extract features from all sources
        system_features = self.extract_system_features(hours)
        app_features = self.extract_app_usage_features(hours)
        typing_features = self.extract_typing_features(hours)
        sleep_features = self.extract_sleep_features(min(hours * 7, 168))  # Get more sleep data
        journal_features = self.extract_journal_features(min(hours * 7, 168))  # Get more journal data
        
        # Merge features on timestamp
        feature_matrix = pd.DataFrame()
        
        if not system_features.empty:
            feature_matrix = system_features
        
        if not app_features.empty:
            if feature_matrix.empty:
                feature_matrix = app_features
            else:
                # Merge on nearest timestamp
                feature_matrix = pd.merge_asof(
                    feature_matrix.sort_values('timestamp'),
                    app_features.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta('1 hour')
                )
        
        if not typing_features.empty:
            if feature_matrix.empty:
                feature_matrix = typing_features
            else:
                feature_matrix = pd.merge_asof(
                    feature_matrix.sort_values('timestamp'),
                    typing_features.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    tolerance=pd.Timedelta('1 hour')
                )
        
        # Add sleep and journal features (less frequent, use last available)
        if not sleep_features.empty:
            if feature_matrix.empty:
                feature_matrix = sleep_features
            else:
                feature_matrix = pd.merge_asof(
                    feature_matrix.sort_values('timestamp'),
                    sleep_features.sort_values('timestamp'),
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta('24 hours')
                )
        
        if not journal_features.empty:
            if feature_matrix.empty:
                feature_matrix = journal_features
            else:
                feature_matrix = pd.merge_asof(
                    feature_matrix.sort_values('timestamp'),
                    journal_features.sort_values('timestamp'),
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta('24 hours')
                )
        
        # Clean and prepare features
        if feature_matrix.empty:
            return pd.DataFrame(), []
        
        # Remove timestamp column (keep for reference but not for modeling)
        timestamps = feature_matrix['timestamp'].copy()
        feature_matrix = feature_matrix.drop(columns=['timestamp'])
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
        feature_matrix = feature_matrix.fillna(0)  # Fill remaining NaNs with 0
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        # Add timestamp back
        feature_matrix['timestamp'] = timestamps
        
        return feature_matrix, self.feature_names
    
    def create_target_variables(self, hours: int = 24) -> pd.DataFrame:
        """Create target variables for supervised learning."""
        # For self-supervised learning, create proxy labels
        feature_matrix, _ = self.create_feature_matrix(hours * 2)  # Get more data for targets
        
        if feature_matrix.empty:
            return pd.DataFrame()
        
        targets = pd.DataFrame()
        targets['timestamp'] = feature_matrix['timestamp']
        
        # Target 1: Productivity drop (decrease in typing speed + increase in errors)
        if 'wpm' in feature_matrix.columns:
            targets['productivity_drop'] = (
                (feature_matrix['wpm'].diff() < -5) |  # WPM drop
                (feature_matrix['error_rate'].diff() > 0.05)  # Error rate increase
            ).astype(int)
        
        # Target 2: Cognitive overload (high CPU + high app switching)
        if 'cognitive_load_score' in feature_matrix.columns and 'total_app_switches' in feature_matrix.columns:
            targets['cognitive_overload'] = (
                (feature_matrix['cognitive_load_score'] > 0.7) &
                (feature_matrix['total_app_switches'] > feature_matrix['total_app_switches'].quantile(0.75))
            ).astype(int)
        
        # Target 3: Fatigue indicator (typing variance + sentiment decline)
        if 'typing_fatigue_score' in feature_matrix.columns and 'sentiment_score' in feature_matrix.columns:
            targets['fatigue_event'] = (
                (feature_matrix['typing_fatigue_score'] > 0.6) |
                (feature_matrix['sentiment_score'].diff() < -0.2)
            ).astype(int)
        
        # Target 4: Burnout risk (sustained high stress + low mood)
        if 'stress_rolling_mean_7' in feature_matrix.columns and 'mood_rolling_mean_7' in feature_matrix.columns:
            targets['burnout_risk'] = (
                (feature_matrix['stress_rolling_mean_7'] > 7) &
                (feature_matrix['mood_rolling_mean_7'] < 4)
            ).astype(int)
        
        return targets
    
    def prepare_training_data(self, hours: int = 168) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with features and targets."""
        # Get features and targets
        feature_matrix, feature_names = self.create_feature_matrix(hours)
        targets = self.create_target_variables(hours)
        
        if feature_matrix.empty or targets.empty:
            return np.array([]), np.array([]), []
        
        # Align features and targets
        merged_data = pd.merge(
            feature_matrix,
            targets,
            on='timestamp',
            how='inner'
        )
        
        if merged_data.empty:
            return np.array([]), np.array([]), []
        
        # Separate features and targets
        feature_columns = [col for col in feature_names if col in merged_data.columns]
        target_columns = [col for col in targets.columns if col != 'timestamp' and col in merged_data.columns]
        
        X = merged_data[feature_columns].values
        y = merged_data[target_columns].values
        
        # Standardize features
        if len(X) > 0:
            X = self.scaler.fit_transform(X)
        
        return X, y, feature_columns
    
    def extract_realtime_features(self) -> Dict:
        """Extract real-time features for current prediction."""
        # Get recent data (last hour)
        feature_matrix, feature_names = self.create_feature_matrix(1)
        
        if feature_matrix.empty:
            return {}
        
        # Get the most recent feature row
        latest_features = feature_matrix.iloc[-1]
        
        # Convert to dictionary
        features_dict = latest_features.to_dict()
        
        # Standardize using fitted scaler
        if hasattr(self.scaler, 'mean_'):
            feature_values = []
            for name in feature_names:
                if name in features_dict and not pd.isna(features_dict[name]):
                    feature_values.append(features_dict[name])
                else:
                    feature_values.append(0)
            
            if len(feature_values) == len(self.scaler.mean_):
                standardized_values = self.scaler.transform([feature_values])[0]
                for i, name in enumerate(feature_names):
                    features_dict[f'{name}_standardized'] = standardized_values[i]
        
        return features_dict
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using statistical methods."""
        if len(X) == 0 or len(y) == 0:
            return {}
        
        importance_scores = {}
        
        # Use correlation with targets
        for i, feature_name in enumerate(self.feature_names):
            if i < X.shape[1]:
                correlations = []
                for target_col in range(y.shape[1]):
                    correlation = np.corrcoef(X[:, i], y[:, target_col])[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
                
                if correlations:
                    importance_scores[feature_name] = np.mean(correlations)
                else:
                    importance_scores[feature_name] = 0.0
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance


if __name__ == "__main__":
    # Test feature engineering
    engineer = FeatureEngineer()
    
    print("Creating feature matrix...")
    features, feature_names = engineer.create_feature_matrix(24)
    
    if not features.empty:
        print(f"Feature matrix shape: {features.shape}")
        print(f"Feature names: {feature_names[:10]}...")  # Show first 10
        
        print("\nExtracting real-time features...")
        realtime_features = engineer.extract_realtime_features()
        print(f"Real-time features: {len(realtime_features)} features extracted")
        
        print("\nPreparing training data...")
        X, y, names = engineer.prepare_training_data(168)
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        if len(X) > 0:
            print("\nCalculating feature importance...")
            importance = engineer.get_feature_importance(X, y)
            print("Top 5 important features:")
            for i, (feature, score) in enumerate(list(importance.items())[:5]):
                print(f"{i+1}. {feature}: {score:.3f}")
    else:
        print("No data available for feature engineering")
