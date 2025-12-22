"""
ML Models Module
Contains various machine learning models for predicting burnout and performance metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
import os

# ML models
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Deep learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaselineModels:
    """Baseline statistical models for initial predictions."""
    
    def __init__(self):
        self.models = {}
        self.is_trained = {}
    
    def train_moving_average_model(self, data: pd.DataFrame, target_column: str, window: int = 5) -> Dict:
        """Train a simple moving average model."""
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        # Calculate moving average
        data[f'{target_column}_ma_{window}'] = data[target_column].rolling(window=window).mean()
        
        # Simple prediction: next value = current moving average
        predictions = data[f'{target_column}_ma_{window}'].shift(1)
        
        # Calculate performance
        actual = data[target_column].dropna()
        predicted = predictions.dropna()
        
        # Align actual and predicted
        common_index = actual.index.intersection(predicted.index)
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        mae = np.mean(np.abs(actual_aligned - predicted_aligned))
        
        model_info = {
            'model_type': 'moving_average',
            'window': window,
            'mse': mse,
            'mae': mae,
            'last_ma_value': data[f'{target_column}_ma_{window}'].iloc[-1]
        }
        
        self.models[f'ma_{target_column}'] = model_info
        self.is_trained[f'ma_{target_column}'] = True
        
        return model_info
    
    def train_zscore_anomaly_model(self, data: pd.DataFrame, target_column: str, threshold: float = 2.0) -> Dict:
        """Train a Z-score anomaly detection model."""
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        values = data[target_column].dropna()
        
        if len(values) < 10:
            raise ValueError("Not enough data for Z-score model")
        
        mean_val = values.mean()
        std_val = values.std()
        
        # Detect anomalies
        z_scores = np.abs((values - mean_val) / std_val)
        anomalies = z_scores > threshold
        
        model_info = {
            'model_type': 'zscore_anomaly',
            'mean': mean_val,
            'std': std_val,
            'threshold': threshold,
            'anomaly_rate': anomalies.mean(),
            'last_z_score': z_scores.iloc[-1] if len(z_scores) > 0 else 0
        }
        
        self.models[f'zscore_{target_column}'] = model_info
        self.is_trained[f'zscore_{target_column}'] = True
        
        return model_info
    
    def predict_with_moving_average(self, model_key: str, steps_ahead: int = 1) -> float:
        """Make prediction using moving average model."""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained")
        
        model_info = self.models[model_key]
        # For moving average, prediction is the last calculated MA
        return model_info['last_ma_value']
    
    def predict_anomaly_score(self, model_key: str, value: float) -> Dict:
        """Calculate anomaly score for a value."""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not trained")
        
        model_info = self.models[model_key]
        
        z_score = abs((value - model_info['mean']) / model_info['std'])
        is_anomaly = z_score > model_info['threshold']
        
        return {
            'z_score': z_score,
            'is_anomaly': is_anomaly,
            'anomaly_level': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'low'
        }


class TimeSeriesModels:
    """Advanced time-series forecasting models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = {}
    
    def train_random_forest_regressor(self, X: np.ndarray, y: np.ndarray, 
                                    model_name: str = "rf_regressor") -> Dict:
        """Train Random Forest regressor for time series prediction."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data")
        
        # Use last column as target for regression
        if y.shape[1] > 1:
            y_target = y[:, -1]  # Use last target column
        else:
            y_target = y.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_target))
        X_clean = X[valid_mask]
        y_clean = y_target[valid_mask]
        
        if len(X_clean) < 10:
            raise ValueError("Not enough clean data for training")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_clean, y_clean, cv=tscv, scoring='neg_mean_squared_error')
        
        # Train on full dataset
        model.fit(X_clean, y_clean)
        
        model_info = {
            'model_type': 'random_forest_regressor',
            'cv_mse_scores': -cv_scores,
            'mean_cv_mse': (-cv_scores).mean(),
            'feature_importance': model.feature_importances_.tolist(),
            'model': model
        }
        
        self.models[model_name] = model_info
        self.is_trained[model_name] = True
        
        return model_info
    
    def train_random_forest_classifier(self, X: np.ndarray, y: np.ndarray,
                                     model_name: str = "rf_classifier") -> Dict:
        """Train Random Forest classifier for event prediction."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data")
        
        # Use first column as target for classification
        if y.shape[1] > 0:
            y_target = y[:, 0]  # Use first target column
        else:
            y_target = y.flatten()
        
        # Remove NaN values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_target))
        X_clean = X[valid_mask]
        y_clean = y_target[valid_mask]
        
        if len(X_clean) < 10:
            raise ValueError("Not enough clean data for training")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_clean, y_clean, cv=tscv, scoring='accuracy')
        
        # Train on full dataset
        model.fit(X_clean, y_clean)
        
        model_info = {
            'model_type': 'random_forest_classifier',
            'cv_accuracy_scores': cv_scores,
            'mean_cv_accuracy': cv_scores.mean(),
            'feature_importance': model.feature_importances_.tolist(),
            'class_distribution': dict(zip(*np.unique(y_clean, return_counts=True))),
            'model': model
        }
        
        self.models[model_name] = model_info
        self.is_trained[model_name] = True
        
        return model_info
    
    def predict_regression(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make regression predictions."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]['model']
        return model.predict(X)
    
    def predict_classification(self, model_name: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make classification predictions with probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]['model']
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities


if TORCH_AVAILABLE:
    class LSTMPredictor(nn.Module):
        """LSTM model for time series prediction."""
        
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
            super(LSTMPredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            
            # LSTM forward pass
            out, _ = self.lstm(x, (h0, c0))
            out = self.dropout(out[:, -1, :])  # Take last time step
            out = self.fc(out)
            return out
    
    class TimeSeriesDataset(Dataset):
        """PyTorch dataset for time series data."""
        
        def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.sequence_length = sequence_length
        
        def __len__(self):
            return len(self.X) - self.sequence_length
        
        def __getitem__(self, idx):
            return (
                self.X[idx:idx+self.sequence_length],
                self.y[idx+self.sequence_length]
            )


class DeepLearningModels:
    """Deep learning models for advanced predictions."""
    
    def __init__(self):
        self.models = {}
        self.is_trained = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 10, epochs: int = 50,
                        model_name: str = "lstm_predictor") -> Dict:
        """Train LSTM model for time series prediction."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for deep learning models")
        
        if len(X) < sequence_length + 1:
            raise ValueError("Not enough data for LSTM training")
        
        # Prepare data
        dataset = TimeSeriesDataset(X, y.flatten(), sequence_length)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = LSTMPredictor(input_size=X.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        model_info = {
            'model_type': 'lstm',
            'sequence_length': sequence_length,
            'epochs': epochs,
            'final_loss': losses[-1],
            'training_losses': losses,
            'model': model,
            'device': str(self.device)
        }
        
        self.models[model_name] = model_info
        self.is_trained[model_name] = True
        
        return model_info
    
    def predict_lstm(self, model_name: str, X: np.ndarray, steps_ahead: int = 1) -> np.ndarray:
        """Make predictions using trained LSTM model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model_info = self.models[model_name]
        model = model_info['model']
        sequence_length = model_info['sequence_length']
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            # Use last sequence as input
            input_sequence = torch.FloatTensor(X[-sequence_length:]).unsqueeze(0).to(self.device)
            
            for _ in range(steps_ahead):
                output = model(input_sequence)
                predictions.append(output.cpu().numpy()[0, 0])
                
                # Update input sequence for next prediction
                input_sequence = torch.cat([
                    input_sequence[:, 1:, :],
                    output.unsqueeze(1)
                ], dim=1)
        
        return np.array(predictions)


class EnsembleModel:
    """Ensemble model combining multiple predictors."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.is_trained = False
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
    
    def predict_regression(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble regression prediction."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif isinstance(model, dict) and 'model' in model:
                pred = model['model'].predict(X)
            else:
                continue
            
            weight = self.weights[name] / total_weight
            predictions.append(pred * weight)
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            raise ValueError("No valid predictions from ensemble models")
    
    def predict_classification(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble classification prediction."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        probabilities = []
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict(X)
                prob = model.predict_proba(X)
            elif isinstance(model, dict) and 'model' in model:
                pred = model['model'].predict(X)
                prob = model['model'].predict_proba(X)
            else:
                continue
            
            weight = self.weights[name] / total_weight
            predictions.append(pred * weight)
            probabilities.append(prob * weight)
        
        if predictions and probabilities:
            # Weighted averaging
            final_predictions = np.round(np.sum(predictions, axis=0)).astype(int)
            final_probabilities = np.sum(probabilities, axis=0)
            return final_predictions, final_probabilities
        else:
            raise ValueError("No valid predictions from ensemble models")


class ModelManager:
    """Main model management class."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.baseline_models = BaselineModels()
        self.ts_models = TimeSeriesModels()
        self.dl_models = DeepLearningModels() if TORCH_AVAILABLE else None
        self.ensemble = EnsembleModel()
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train all available models."""
        results = {}
        
        if len(X) == 0 or len(y) == 0:
            return {"error": "No training data available"}
        
        try:
            # Train baseline models
            print("Training baseline models...")
            # Note: Baseline models need DataFrame, not numpy arrays
            # This would be called from the training module
            
            # Train time series models
            print("Training Random Forest models...")
            rf_reg_result = self.ts_models.train_random_forest_regressor(X, y, "rf_performance")
            rf_cls_result = self.ts_models.train_random_forest_classifier(X, y, "rf_burnout")
            
            results['rf_regressor'] = rf_reg_result
            results['rf_classifier'] = rf_cls_result
            
            # Train deep learning models if available
            if self.dl_models and len(X) > 20:
                print("Training LSTM model...")
                try:
                    lstm_result = self.dl_models.train_lstm_model(X, y, epochs=30)
                    results['lstm'] = lstm_result
                except Exception as e:
                    print(f"LSTM training failed: {e}")
            
            # Create ensemble
            print("Creating ensemble...")
            if 'rf_regressor' in results:
                self.ensemble.add_model('rf', results['rf_regressor']['model'], weight=0.6)
            if 'lstm' in results:
                self.ensemble.add_model('lstm', results['lstm']['model'], weight=0.4)
            
            self.ensemble.is_trained = True
            results['ensemble_created'] = True
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def predict_performance_metrics(self, features: np.ndarray, hours_ahead: int = 24) -> Dict:
        """Predict performance metrics for future hours."""
        if not self.ensemble.is_trained:
            return {"error": "No trained models available"}
        
        try:
            # Make predictions
            if hasattr(self.ensemble, 'predict_regression'):
                predictions = self.ensemble.predict_regression(features)
            else:
                # Fallback to best available model
                if 'rf_performance' in self.ts_models.models:
                    predictions = self.ts_models.predict_regression('rf_performance', features)
                else:
                    return {"error": "No regression models available"}
            
            # Format predictions
            if isinstance(predictions, np.ndarray):
                if predictions.ndim == 0:
                    predictions = [float(predictions)]
                else:
                    predictions = predictions.tolist()
            
            return {
                'predictions': predictions,
                'hours_ahead': hours_ahead,
                'confidence': 'medium',  # Would calculate actual confidence
                'model_type': 'ensemble'
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_burnout_risk(self, features: np.ndarray) -> Dict:
        """Predict burnout risk."""
        if not self.ensemble.is_trained:
            return {"error": "No trained models available"}
        
        try:
            # Make classification predictions
            if hasattr(self.ensemble, 'predict_classification'):
                predictions, probabilities = self.ensemble.predict_classification(features)
            else:
                # Fallback to available model
                if 'rf_burnout' in self.ts_models.models:
                    predictions, probabilities = self.ts_models.predict_classification('rf_burnout', features)
                else:
                    return {"error": "No classification models available"}
            
            # Calculate risk score
            if len(probabilities) > 1:
                risk_score = probabilities[1]  # Probability of positive class (burnout)
            else:
                risk_score = float(predictions[0]) if len(predictions) > 0 else 0.0
            
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
                'burnout_risk_score': risk_score,
                'risk_level': risk_level,
                'prediction': int(predictions[0]) if len(predictions) > 0 else 0,
                'confidence': max(probabilities) if len(probabilities) > 0 else 0.0,
                'model_type': 'ensemble'
            }
            
        except Exception as e:
            return {"error": f"Burnout prediction failed: {str(e)}"}
    
    def save_models(self) -> Dict:
        """Save all trained models to disk."""
        saved_models = {}
        
        try:
            # Save time series models
            for name, model_info in self.ts_models.models.items():
                if 'model' in model_info:
                    filename = os.path.join(self.model_dir, f"{name}.pkl")
                    with open(filename, 'wb') as f:
                        pickle.dump(model_info['model'], f)
                    saved_models[name] = filename
            
            # Save deep learning models
            if self.dl_models:
                for name, model_info in self.dl_models.models.items():
                    if 'model' in model_info:
                        filename = os.path.join(self.model_dir, f"{name}.pth")
                        torch.save(model_info['model'].state_dict(), filename)
                        saved_models[name] = filename
            
            return {"saved_models": saved_models, "status": "success"}
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def load_models(self) -> Dict:
        """Load saved models from disk."""
        loaded_models = {}
        
        try:
            # Load time series models
            for filename in os.listdir(self.model_dir):
                if filename.endswith('.pkl'):
                    name = filename.replace('.pkl', '')
                    filepath = os.path.join(self.model_dir, filename)
                    
                    with open(filepath, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Reconstruct model info
                    model_info = {
                        'model_type': 'random_forest',
                        'model': model,
                        'loaded_from': filepath
                    }
                    
                    self.ts_models.models[name] = model_info
                    self.ts_models.is_trained[name] = True
                    loaded_models[name] = filepath
            
            return {"loaded_models": loaded_models, "status": "success"}
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    # Test model training with dummy data
    print("Testing ML models...")
    
    # Create dummy data
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100, 3)
    
    manager = ModelManager()
    
    print("Training models...")
    results = manager.train_all_models(X_dummy, y_dummy)
    print("Training results:", json.dumps(results, indent=2, default=str))
    
    # Test predictions
    if 'rf_regressor' in results:
        print("\nTesting predictions...")
        X_test = np.random.randn(5, 10)
        
        perf_pred = manager.predict_performance_metrics(X_test)
        print("Performance prediction:", perf_pred)
        
        burnout_pred = manager.predict_burnout_risk(X_test)
        print("Burnout prediction:", burnout_pred)
        
        # Save models
        save_result = manager.save_models()
        print("Save result:", save_result)
