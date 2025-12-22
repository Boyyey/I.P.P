"""
Inference Module
Handles real-time predictions and model inference for burnout and performance prediction.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import pickle
from collections import deque
import threading
import time

# Import our modules
from feature_engineering import FeatureEngineer
from models import ModelManager, BaselineModels
from training import ModelTrainer


class RealTimePredictor:
    """Real-time prediction system for continuous monitoring."""
    
    def __init__(self, db_path: str = "data/performance_data.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer(db_path)
        self.model_manager = ModelManager(model_dir)
        
        # Prediction cache
        self.prediction_cache = deque(maxlen=100)  # Keep last 100 predictions
        self.last_prediction_time = None
        self.prediction_interval = 300  # 5 minutes between predictions
        
        # Load trained models
        self.load_models()
        
        # Prediction history
        self.prediction_history = []
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            # Load model manager models
            load_result = self.model_manager.load_models()
            
            # Load baseline models
            baseline_file = os.path.join(self.model_dir, "baseline_models.pkl")
            if os.path.exists(baseline_file):
                with open(baseline_file, 'rb') as f:
                    self.baseline_models = pickle.load(f)
            else:
                self.baseline_models = BaselineModels()
            
            print(f"Models loaded: {load_result}")
            return True
            
        except Exception as e:
            print(f"Failed to load models: {e}")
            self.baseline_models = BaselineModels()
            return False
    
    def extract_current_features(self) -> Dict:
        """Extract current features for real-time prediction."""
        try:
            features = self.feature_engineer.extract_realtime_features()
            
            if not features:
                print("No features available for prediction")
                return {}
            
            # Convert to numpy array for ML models
            feature_names = self.feature_engineer.feature_names
            feature_values = []
            
            for name in feature_names:
                if name in features and not pd.isna(features[name]):
                    feature_values.append(features[name])
                else:
                    feature_values.append(0)  # Default value for missing features
            
            features_array = np.array(feature_values).reshape(1, -1)
            
            return {
                'features_dict': features,
                'features_array': features_array,
                'feature_names': feature_names,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return {}
    
    def predict_performance_trend(self, hours_ahead: int = 24) -> Dict:
        """Predict performance trend for the next N hours."""
        current_features = self.extract_current_features()
        
        if not current_features:
            return {"error": "No features available for prediction"}
        
        try:
            features_array = current_features['features_array']
            
            # Make prediction using available models
            prediction_result = self.model_manager.predict_performance_metrics(
                features_array, hours_ahead
            )
            
            if 'error' in prediction_result:
                # Fallback to baseline models
                prediction_result = self._predict_with_baseline(current_features['features_dict'])
            
            # Add context
            prediction_result.update({
                'prediction_type': 'performance_trend',
                'hours_ahead': hours_ahead,
                'timestamp': current_features['timestamp'],
                'features_used': len(current_features['feature_names'])
            })
            
            # Cache prediction
            self.prediction_cache.append(prediction_result)
            self.last_prediction_time = datetime.now()
            
            return prediction_result
            
        except Exception as e:
            return {"error": f"Performance prediction failed: {str(e)}"}
    
    def predict_burnout_risk(self) -> Dict:
        """Predict current burnout risk."""
        current_features = self.extract_current_features()
        
        if not current_features:
            return {"error": "No features available for prediction"}
        
        try:
            features_array = current_features['features_array']
            
            # Make burnout prediction
            burnout_result = self.model_manager.predict_burnout_risk(features_array)
            
            if 'error' in burnout_result:
                # Fallback to baseline anomaly detection
                burnout_result = self._predict_burnout_with_baseline(current_features['features_dict'])
            
            # Add recommendations based on risk level
            burnout_result['recommendations'] = self._generate_burnout_recommendations(
                burnout_result.get('risk_level', 'unknown'),
                current_features['features_dict']
            )
            
            # Add context
            burnout_result.update({
                'prediction_type': 'burnout_risk',
                'timestamp': current_features['timestamp'],
                'features_used': len(current_features['feature_names'])
            })
            
            return burnout_result
            
        except Exception as e:
            return {"error": f"Burnout prediction failed: {str(e)}"}
    
    def _predict_with_baseline(self, features_dict: Dict) -> Dict:
        """Fallback prediction using baseline models."""
        try:
            predictions = {}
            
            # Use moving average predictions for key metrics
            key_metrics = ['cognitive_load_score', 'wpm', 'sentiment_score']
            
            for metric in key_metrics:
                model_key = f'ma_{metric}'
                if hasattr(self.baseline_models, 'models') and model_key in self.baseline_models.models:
                    ma_prediction = self.baseline_models.predict_with_moving_average(model_key)
                    predictions[metric] = ma_prediction
            
            # Create simple trend prediction
            if predictions:
                avg_prediction = np.mean(list(predictions.values()))
                return {
                    'predictions': [avg_prediction],
                    'method': 'baseline_moving_average',
                    'confidence': 'low',
                    'baseline_metrics': predictions
                }
            else:
                return {
                    'predictions': [0.5],  # Neutral prediction
                    'method': 'default',
                    'confidence': 'very_low'
                }
                
        except Exception as e:
            return {"error": f"Baseline prediction failed: {str(e)}"}
    
    def _predict_burnout_with_baseline(self, features_dict: Dict) -> Dict:
        """Fallback burnout prediction using baseline anomaly detection."""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # Check for anomalies in key metrics
            key_metrics = ['cognitive_load_score', 'wpm', 'sentiment_score']
            
            for metric in key_metrics:
                if metric in features_dict:
                    value = features_dict[metric]
                    model_key = f'zscore_{metric}'
                    
                    if hasattr(self.baseline_models, 'models') and model_key in self.baseline_models.models:
                        anomaly_result = self.baseline_models.predict_anomaly_score(model_key, value)
                        
                        if anomaly_result['is_anomaly']:
                            risk_score += 0.2
                            risk_factors.append(f"Anomaly detected in {metric}")
            
            # Simple heuristic based on feature values
            if 'cognitive_load_score' in features_dict:
                if features_dict['cognitive_load_score'] > 0.8:
                    risk_score += 0.3
                    risk_factors.append("High cognitive load")
            
            if 'error_rate' in features_dict and features_dict['error_rate'] > 0.2:
                risk_score += 0.2
                risk_factors.append("High error rate")
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.6:
                risk_level = "medium"
            elif risk_score < 0.8:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            return {
                'burnout_risk_score': min(risk_score, 1.0),
                'risk_level': risk_level,
                'method': 'baseline_anomaly_detection',
                'confidence': 'low',
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            return {"error": f"Baseline burnout prediction failed: {str(e)}"}
    
    def _generate_burnout_recommendations(self, risk_level: str, features: Dict) -> List[str]:
        """Generate recommendations based on burnout risk level and current features."""
        recommendations = []
        
        if risk_level == "low":
            recommendations.append("Continue maintaining healthy work habits")
            recommendations.append("Keep monitoring your performance patterns")
        
        elif risk_level == "medium":
            recommendations.append("Consider taking short breaks every hour")
            recommendations.append("Ensure you're getting adequate sleep")
            if features.get('cognitive_load_score', 0) > 0.6:
                recommendations.append("Reduce multitasking and focus on one task at a time")
        
        elif risk_level == "high":
            recommendations.append("Take a 15-20 minute break immediately")
            recommendations.append("Postpone non-urgent tasks if possible")
            recommendations.append("Practice stress reduction techniques")
            if features.get('error_rate', 0) > 0.15:
                recommendations.append("Your error rate is elevated - double-check important work")
        
        elif risk_level == "critical":
            recommendations.append("STOP - Take a break immediately")
            recommendations.append("Consider ending work for the day")
            recommendations.append("Seek support from colleagues or supervisor")
            recommendations.append("Focus on basic self-care: hydration, nutrition, rest")
        
        # Add specific recommendations based on features
        if features.get('avg_key_interval', 0) > 0.5:  # Slow typing
            recommendations.append("Your typing speed has decreased - you may be fatigued")
        
        if features.get('app_fragmentation', 0) > 10:  # High app switching
            recommendations.append("High app switching detected - try to minimize distractions")
        
        return recommendations
    
    def get_prediction_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent predictions."""
        if not self.prediction_cache:
            return {"message": "No predictions available yet"}
        
        # Convert cache to list for analysis
        recent_predictions = list(self.prediction_cache)
        
        # Analyze performance predictions
        performance_preds = [p for p in recent_predictions if p.get('prediction_type') == 'performance_trend']
        burnout_preds = [p for p in recent_predictions if p.get('prediction_type') == 'burnout_risk']
        
        summary = {
            'summary_period_hours': hours,
            'total_predictions': len(recent_predictions),
            'performance_predictions': len(performance_preds),
            'burnout_predictions': len(burnout_preds),
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None
        }
        
        # Performance trend analysis
        if performance_preds:
            perf_values = []
            for pred in performance_preds:
                if 'predictions' in pred and pred['predictions']:
                    perf_values.extend(pred['predictions'])
            
            if perf_values:
                summary['performance_trend'] = {
                    'avg_predicted_performance': np.mean(perf_values),
                    'performance_variance': np.var(perf_values),
                    'trend_direction': 'improving' if len(perf_values) > 1 and perf_values[-1] > perf_values[0] else 'declining'
                }
        
        # Burnout risk analysis
        if burnout_preds:
            risk_scores = [p.get('burnout_risk_score', 0) for p in burnout_preds]
            risk_levels = [p.get('risk_level', 'unknown') for p in burnout_preds]
            
            if risk_scores:
                summary['burnout_risk_analysis'] = {
                    'avg_risk_score': np.mean(risk_scores),
                    'max_risk_score': np.max(risk_scores),
                    'current_risk_level': burnout_preds[-1].get('risk_level', 'unknown') if burnout_preds else 'unknown',
                    'risk_trend': 'increasing' if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else 'stable'
                }
        
        return summary
    
    def start_continuous_monitoring(self, interval_minutes: int = 5):
        """Start continuous monitoring in background thread."""
        def monitoring_loop():
            while True:
                try:
                    # Make predictions
                    perf_pred = self.predict_performance_trend()
                    burnout_pred = self.predict_burnout_risk()
                    
                    # Store in history
                    self.prediction_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'performance_prediction': perf_pred,
                        'burnout_prediction': burnout_pred
                    })
                    
                    # Keep history manageable
                    if len(self.prediction_history) > 1000:
                        self.prediction_history = self.prediction_history[-500:]
                    
                    # Check for alerts
                    self._check_for_alerts(burnout_pred)
                    
                    time.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        print(f"Started continuous monitoring with {interval_minutes} minute intervals")
    
    def _check_for_alerts(self, burnout_prediction: Dict):
        """Check if alerts should be triggered based on predictions."""
        risk_level = burnout_prediction.get('risk_level', '')
        risk_score = burnout_prediction.get('burnout_risk_score', 0)
        
        if risk_level in ['high', 'critical']:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'burnout_risk',
                'severity': risk_level,
                'risk_score': risk_score,
                'message': f"High burnout risk detected: {risk_level} (score: {risk_score:.2f})",
                'recommendations': burnout_prediction.get('recommendations', [])
            }
            
            # In a real system, this would trigger notifications
            print(f"ALERT: {alert['message']}")
            return alert
        
        return None


class PredictionAPI:
    """Simple API-like interface for predictions."""
    
    def __init__(self, db_path: str = "data/performance_data.db", model_dir: str = "models"):
        self.predictor = RealTimePredictor(db_path, model_dir)
    
    def get_current_performance_prediction(self, hours_ahead: int = 24) -> Dict:
        """Get current performance prediction."""
        return self.predictor.predict_performance_trend(hours_ahead)
    
    def get_current_burnout_risk(self) -> Dict:
        """Get current burnout risk assessment."""
        return self.predictor.predict_burnout_risk()
    
    def get_prediction_dashboard(self) -> Dict:
        """Get comprehensive prediction dashboard."""
        performance = self.get_current_performance_prediction()
        burnout = self.get_current_burnout_risk()
        summary = self.predictor.get_prediction_summary()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_prediction': performance,
            'burnout_risk': burnout,
            'summary': summary,
            'status': 'active' if not performance.get('error') and not burnout.get('error') else 'error'
        }
    
    def get_feature_analysis(self) -> Dict:
        """Get analysis of current features."""
        features = self.predictor.extract_current_features()
        
        if not features:
            return {"error": "No features available"}
        
        features_dict = features['features_dict']
        
        # Analyze key features
        analysis = {
            'timestamp': features['timestamp'],
            'total_features': len(features_dict),
            'key_metrics': {}
        }
        
        key_metrics = ['cognitive_load_score', 'wpm', 'error_rate', 'sentiment_score', 
                      'focus_ratio', 'sleep_debt', 'mental_health_score']
        
        for metric in key_metrics:
            if metric in features_dict:
                value = features_dict[metric]
                analysis['key_metrics'][metric] = {
                    'value': value,
                    'status': self._get_metric_status(metric, value),
                    'description': self._get_metric_description(metric, value)
                }
        
        return analysis
    
    def _get_metric_status(self, metric: str, value: float) -> str:
        """Get status assessment for a metric."""
        if metric == 'cognitive_load_score':
            if value < 0.3:
                return 'low'
            elif value < 0.7:
                return 'normal'
            else:
                return 'high'
        
        elif metric == 'wpm':
            if value < 20:
                return 'slow'
            elif value < 60:
                return 'normal'
            else:
                return 'fast'
        
        elif metric == 'error_rate':
            if value < 0.05:
                return 'excellent'
            elif value < 0.15:
                return 'normal'
            else:
                return 'high'
        
        elif metric == 'sentiment_score':
            if value < -0.2:
                return 'negative'
            elif value < 0.2:
                return 'neutral'
            else:
                return 'positive'
        
        elif metric == 'focus_ratio':
            if value < 0.5:
                return 'distracted'
            elif value < 0.8:
                return 'normal'
            else:
                return 'focused'
        
        elif metric == 'sleep_debt':
            if value < 1:
                return 'well_rested'
            elif value < 3:
                return 'mild_debt'
            else:
                return 'significant_debt'
        
        elif metric == 'mental_health_score':
            if value < 0.4:
                return 'poor'
            elif value < 0.7:
                return 'moderate'
            else:
                return 'good'
        
        else:
            return 'unknown'
    
    def _get_metric_description(self, metric: str, value: float) -> str:
        """Get human-readable description for metric value."""
        status = self._get_metric_status(metric, value)
        
        descriptions = {
            'cognitive_load_score': {
                'low': 'Low cognitive demand - relaxed state',
                'normal': 'Moderate cognitive activity',
                'high': 'High cognitive demand - potential overload'
            },
            'wpm': {
                'slow': 'Reduced typing speed - possible fatigue',
                'normal': 'Normal typing speed',
                'fast': 'High typing speed - good focus'
            },
            'error_rate': {
                'excellent': 'Very low error rate - high accuracy',
                'normal': 'Normal error rate',
                'high': 'Elevated error rate - potential fatigue'
            },
            'sentiment_score': {
                'negative': 'Negative sentiment detected',
                'neutral': 'Neutral sentiment',
                'positive': 'Positive sentiment detected'
            },
            'focus_ratio': {
                'distracted': 'Low focus - many interruptions',
                'normal': 'Moderate focus levels',
                'focused': 'High focus - concentrated work'
            },
            'sleep_debt': {
                'well_rested': 'Well rested - adequate sleep',
                'mild_debt': 'Mild sleep debt - consider more rest',
                'significant_debt': 'Significant sleep debt - prioritize sleep'
            },
            'mental_health_score': {
                'poor': 'Low mental health score - high stress',
                'moderate': 'Moderate mental health score',
                'good': 'Good mental health score - low stress'
            }
        }
        
        return descriptions.get(metric, {}).get(status, f'{metric}: {value:.2f}')


if __name__ == "__main__":
    # Test the prediction system
    print("Testing prediction system...")
    
    api = PredictionAPI()
    
    # Test current predictions
    print("\n=== Current Predictions ===")
    dashboard = api.get_prediction_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Test feature analysis
    print("\n=== Feature Analysis ===")
    feature_analysis = api.get_feature_analysis()
    print(json.dumps(feature_analysis, indent=2, default=str))
    
    # Test prediction summary
    print("\n=== Prediction Summary ===")
    summary = api.predictor.get_prediction_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    print("\nPrediction system test completed!")
