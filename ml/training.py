"""
Training Module
Handles model training, validation, and optimization.
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from feature_engineering import FeatureEngineer
from models import ModelManager, BaselineModels


class ModelTrainer:
    """Main training coordinator for all models."""
    
    def __init__(self, db_path: str = "data/performance_data.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = model_dir
        self.feature_engineer = FeatureEngineer(db_path)
        self.model_manager = ModelManager(model_dir)
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Training history
        self.training_history = []
    
    def prepare_training_dataset(self, days: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare complete training dataset."""
        print(f"Preparing training dataset from last {days} days...")
        
        # Get features and targets
        X, y, feature_names = self.feature_engineer.prepare_training_data(days * 24)
        
        if len(X) == 0:
            print("No training data available")
            return np.array([]), np.array([]), []
        
        print(f"Dataset prepared: X shape={X.shape}, y shape={y.shape}")
        print(f"Features: {len(feature_names)}")
        
        return X, y, feature_names
    
    def train_baseline_models(self, days: int = 7) -> Dict:
        """Train baseline statistical models."""
        print("Training baseline models...")
        
        results = {}
        
        # Get recent data for baseline models
        feature_matrix, _ = self.feature_engineer.create_feature_matrix(days * 24)
        
        if feature_matrix.empty:
            return {"error": "No data available for baseline training"}
        
        baseline = BaselineModels()
        
        # Train moving average models for key metrics
        key_metrics = ['cognitive_load_score', 'wpm', 'sentiment_score']
        
        for metric in key_metrics:
            if metric in feature_matrix.columns:
                try:
                    ma_result = baseline.train_moving_average_model(feature_matrix, metric, window=5)
                    results[f'ma_{metric}'] = ma_result
                    print(f"Trained MA model for {metric}: MSE={ma_result['mse']:.4f}")
                except Exception as e:
                    print(f"Failed to train MA model for {metric}: {e}")
                
                try:
                    zscore_result = baseline.train_zscore_anomaly_model(feature_matrix, metric, threshold=2.0)
                    results[f'zscore_{metric}'] = zscore_result
                    print(f"Trained Z-score model for {metric}: anomaly_rate={zscore_result['anomaly_rate']:.3f}")
                except Exception as e:
                    print(f"Failed to train Z-score model for {metric}: {e}")
        
        # Save baseline models
        baseline_file = os.path.join(self.model_dir, "baseline_models.pkl")
        with open(baseline_file, 'wb') as f:
            pickle.dump(baseline, f)
        
        results['baseline_saved'] = baseline_file
        
        return results
    
    def train_advanced_models(self, days: int = 30) -> Dict:
        """Train advanced ML models."""
        print("Training advanced ML models...")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_dataset(days)
        
        if len(X) == 0:
            return {"error": "No training data available"}
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Train models
        training_results = self.model_manager.train_all_models(X_train, y_train)
        
        # Validate models
        validation_results = self._validate_models(X_val, y_val)
        
        # Combine results
        results = {
            'training': training_results,
            'validation': validation_results,
            'feature_names': feature_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'training_date': datetime.now().isoformat()
        }
        
        # Save training results
        self._save_training_results(results)
        
        return results
    
    def _validate_models(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Validate trained models on validation set."""
        print("Validating models...")
        
        validation_results = {}
        
        try:
            # Test regression models
            if 'rf_performance' in self.model_manager.ts_models.models:
                rf_pred = self.model_manager.ts_models.predict_regression('rf_performance', X_val)
                
                # Calculate metrics for each target column
                if len(rf_pred) > 0 and y_val.shape[1] > 0:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    mse = mean_squared_error(y_val[:, -1], rf_pred)
                    mae = mean_absolute_error(y_val[:, -1], rf_pred)
                    r2 = r2_score(y_val[:, -1], rf_pred)
                    
                    validation_results['rf_regression'] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2,
                        'predictions_mean': np.mean(rf_pred),
                        'actuals_mean': np.mean(y_val[:, -1])
                    }
            
            # Test classification models
            if 'rf_burnout' in self.model_manager.ts_models.models:
                rf_pred, rf_prob = self.model_manager.ts_models.predict_classification('rf_burnout', X_val)
                
                if len(rf_pred) > 0 and y_val.shape[1] > 0:
                    accuracy = np.mean(rf_pred == y_val[:, 0].astype(int))
                    
                    validation_results['rf_classification'] = {
                        'accuracy': accuracy,
                        'predictions_distribution': dict(zip(*np.unique(rf_pred, return_counts=True))),
                        'actuals_distribution': dict(zip(*np.unique(y_val[:, 0].astype(int), return_counts=True))),
                        'avg_confidence': np.mean(np.max(rf_prob, axis=1))
                    }
            
            # Test ensemble if available
            if self.model_manager.ensemble.is_trained:
                try:
                    ensemble_pred = self.model_manager.predict_performance_metrics(X_val)
                    if 'predictions' in ensemble_pred:
                        ensemble_reg_pred = np.array(ensemble_pred['predictions'])
                        if len(ensemble_reg_pred) > 0 and y_val.shape[1] > 0:
                            mse = mean_squared_error(y_val[:, -1], ensemble_reg_pred)
                            validation_results['ensemble_regression'] = {'mse': mse}
                    
                    burnout_pred = self.model_manager.predict_burnout_risk(X_val)
                    if 'burnout_risk_score' in burnout_pred:
                        validation_results['ensemble_burnout'] = {
                            'avg_risk_score': burnout_pred['burnout_risk_score']
                        }
                except Exception as e:
                    print(f"Ensemble validation failed: {e}")
        
        except Exception as e:
            validation_results['error'] = str(e)
        
        return validation_results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Perform hyperparameter tuning for best models."""
        print("Performing hyperparameter tuning...")
        
        tuning_results = {}
        
        if len(X) < 50:
            return {"error": "Not enough data for hyperparameter tuning"}
        
        # Focus on Random Forest as it's generally most effective
        try:
            # Regression tuning
            param_grid_reg = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use last target column for regression
            y_reg = y[:, -1] if y.shape[1] > 0 else y.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_reg))
            X_clean = X[valid_mask]
            y_clean = y_reg[valid_mask]
            
            if len(X_clean) > 30:
                grid_search_reg = GridSearchCV(
                    rf_reg, param_grid_reg, cv=3, 
                    scoring='neg_mean_squared_error', n_jobs=-1
                )
                grid_search_reg.fit(X_clean, y_clean)
                
                tuning_results['regression'] = {
                    'best_params': grid_search_reg.best_params_,
                    'best_score': -grid_search_reg.best_score_,
                    'best_estimator': grid_search_reg.best_estimator_
                }
            
            # Classification tuning
            if y.shape[1] > 0:
                param_grid_cls = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', None]
                }
                
                rf_cls = RandomForestClassifier(random_state=42, n_jobs=-1)
                
                # Use first target column for classification
                y_cls = y[:, 0].astype(int)
                
                # Remove NaN values
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_cls))
                X_clean = X[valid_mask]
                y_clean = y_cls[valid_mask]
                
                if len(X_clean) > 30 and len(np.unique(y_clean)) > 1:
                    grid_search_cls = GridSearchCV(
                        rf_cls, param_grid_cls, cv=3,
                        scoring='accuracy', n_jobs=-1
                    )
                    grid_search_cls.fit(X_clean, y_clean)
                    
                    tuning_results['classification'] = {
                        'best_params': grid_search_cls.best_params_,
                        'best_score': grid_search_cls.best_score_,
                        'best_estimator': grid_search_cls.best_estimator_
                    }
        
        except Exception as e:
            tuning_results['error'] = str(e)
        
        return tuning_results
    
    def train_with_cross_validation(self, days: int = 30) -> Dict:
        """Train models with comprehensive cross-validation."""
        print("Training with cross-validation...")
        
        # Prepare data
        X, y, feature_names = self.prepare_training_dataset(days)
        
        if len(X) == 0:
            return {"error": "No training data available"}
        
        cv_results = {}
        
        # Time series cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = {
            'regression_mse': [],
            'classification_accuracy': [],
            'fold_sizes': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"Cross-validation fold {fold + 1}/5")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train models on this fold
            fold_models = ModelManager(f"{self.model_dir}/fold_{fold}")
            fold_training = fold_models.train_all_models(X_train_fold, y_train_fold)
            
            # Validate on this fold
            fold_validation = self._validate_models(X_val_fold, y_val_fold)
            
            # Collect scores
            if 'rf_regression' in fold_validation:
                cv_scores['regression_mse'].append(fold_validation['rf_regression']['mse'])
            
            if 'rf_classification' in fold_validation:
                cv_scores['classification_accuracy'].append(fold_validation['rf_classification']['accuracy'])
            
            cv_scores['fold_sizes'].append(len(train_idx))
            
            cv_results[f'fold_{fold}'] = {
                'training': fold_training,
                'validation': fold_validation
            }
        
        # Calculate average scores
        avg_scores = {}
        for metric, values in cv_scores.items():
            if values:
                avg_scores[f'avg_{metric}'] = np.mean(values)
                avg_scores[f'std_{metric}'] = np.std(values)
        
        cv_results['cross_validation_summary'] = avg_scores
        cv_results['total_samples'] = len(X)
        
        return cv_results
    
    def create_training_plots(self, results: Dict) -> Dict:
        """Create visualization plots for training results."""
        print("Creating training plots...")
        
        plot_files = {}
        
        try:
            # Feature importance plot
            if 'rf_regression' in results.get('validation', {}):
                importances = self.model_manager.ts_models.models['rf_performance'].get('feature_importance', [])
                if importances:
                    feature_names = results.get('feature_names', [])
                    
                    plt.figure(figsize=(10, 6))
                    top_indices = np.argsort(importances)[-15:]  # Top 15 features
                    top_importances = [importances[i] for i in top_indices]
                    top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_indices]
                    
                    plt.barh(range(len(top_importances)), top_importances)
                    plt.yticks(range(len(top_features)), top_features)
                    plt.xlabel('Feature Importance')
                    plt.title('Top 15 Feature Importance')
                    plt.tight_layout()
                    
                    plot_file = "plots/feature_importance.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files['feature_importance'] = plot_file
            
            # Training/validation performance plot
            if 'cross_validation_summary' in results:
                summary = results['cross_validation_summary']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Regression MSE
                mse_values = summary.get('avg_regression_mse', 0)
                mse_std = summary.get('std_regression_mse', 0)
                ax1.bar(['Regression MSE'], [mse_values], yerr=[mse_std], capsize=5)
                ax1.set_ylabel('Mean Squared Error')
                ax1.set_title('Regression Performance')
                
                # Classification Accuracy
                acc_values = summary.get('avg_classification_accuracy', 0)
                acc_std = summary.get('std_classification_accuracy', 0)
                ax2.bar(['Classification Accuracy'], [acc_values], yerr=[acc_std], capsize=5)
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Classification Performance')
                ax2.set_ylim(0, 1)
                
                plt.tight_layout()
                plot_file = "plots/model_performance.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['model_performance'] = plot_file
            
            # Learning curves (if available)
            if 'training' in results:
                training_results = results['training']
                
                # Plot training losses for LSTM if available
                if 'lstm' in training_results and 'training_losses' in training_results['lstm']:
                    losses = training_results['lstm']['training_losses']
                    
                    plt.figure(figsize=(8, 5))
                    plt.plot(losses)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('LSTM Training Loss')
                    plt.grid(True)
                    
                    plot_file = "plots/lstm_training_loss.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files['lstm_training_loss'] = plot_file
        
        except Exception as e:
            print(f"Plot creation failed: {e}")
            plot_files['error'] = str(e)
        
        return plot_files
    
    def _save_training_results(self, results: Dict):
        """Save training results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.model_dir, f"training_results_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Training results saved to {results_file}")
        
        # Also save as latest
        latest_file = os.path.join(self.model_dir, "latest_training_results.json")
        with open(latest_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
    
    def run_full_training_pipeline(self, days: int = 30) -> Dict:
        """Run complete training pipeline."""
        print("Starting full training pipeline...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'days_of_data': days
        }
        
        try:
            # Step 1: Train baseline models
            print("\n=== Step 1: Training Baseline Models === acquisitio")
            baseline_results = self.train_baseline_models(days=min(days, 7))
            pipeline_results['baseline_training'] = baseline_results
            
            # Step 2: Train advanced models
            print("\n=== Step 2: Training Advanced Models ===")
            advanced_results = self.train_advanced_models(days)
            pipeline_results['advanced_training'] = advanced_results
            
            # Step 3: Hyperparameter tuning (if enough data)
            if len(advanced_results.get('feature_names', [])) > 10:
                print("\n=== Step 3: Hyperparameter Tuning ===")
                X, y, _ = self.prepare_training_dataset(days)
                if len(X) > 50:
                    tuning_results = self.hyperparameter_tuning(X, y)
                    pipeline_results['hyperparameter_tuning'] = tuning_results
            
            # Step 4: Cross-validation
            print("\n=== Step 4: Cross-Validation ===")
            cv_results = self.train_with_cross_validation(days)
            pipeline_results['cross_validation'] = cv_results
            
            # Step 5: Create plots
            print("\n=== Step 5: Creating Visualizations ===")
            plot_results = self.create_training_plots(advanced_results)
            pipeline_results['plots'] = plot_results
            
            # Step 6: Save final models
            print("\n=== Step 6: Saving Models ===")
            save_results = self.model_manager.save_models()
            pipeline_results['model_saving'] = save_results
            
            pipeline_results['status'] = 'completed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            
            # Calculate training duration
            start = datetime.fromisoformat(pipeline_results['start_time'])
            end = datetime.fromisoformat(pipeline_results['end_time'])
            pipeline_results['duration_minutes'] = (end - start).total_seconds() / 60
            
            print(f"\nTraining pipeline completed in {pipeline_results['duration_minutes']:.1f} minutes")
            
        except Exception as e:
            pipeline_results['status'] = 'failed'
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
        
        # Save pipeline results
        self._save_training_results(pipeline_results)
        
        return pipeline_results
    
    def evaluate_model_performance(self, test_days: int = 7) -> Dict:
        """Evaluate model performance on recent test data."""
        print(f"Evaluating model performance on last {test_days} days...")
        
        # Prepare test data
        X_test, y_test, feature_names = self.prepare_training_dataset(test_days)
        
        if len(X_test) == 0:
            return {"error": "No test data available"}
        
        evaluation_results = {
            'test_samples': len(X_test),
            'test_period_days': test_days,
            'evaluation_date': datetime.now().isoformat()
        }
        
        try:
            # Test regression performance
            perf_pred = self.model_manager.predict_performance_metrics(X_test)
            if 'predictions' in perf_pred:
                predictions = np.array(perf_pred['predictions'])
                if len(predictions) > 0 and y_test.shape[1] > 0:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    actuals = y_test[:, -1]
                    mse = mean_squared_error(actuals, predictions)
                    mae = mean_absolute_error(actuals, predictions)
                    r2 = r2_score(actuals, predictions)
                    
                    evaluation_results['regression_performance'] = {
                        'mse': mse,
                        'mae': mae,
                        'r2_score': r2,
                        'predictions_mean': float(np.mean(predictions)),
                        'actuals_mean': float(np.mean(actuals)),
                        'predictions_std': float(np.std(predictions)),
                        'actuals_std': float(np.std(actuals))
                    }
            
            # Test classification performance
            burnout_pred = self.model_manager.predict_burnout_risk(X_test)
            if 'burnout_risk_score' in burnout_pred:
                risk_scores = [burnout_pred['burnout_risk_score']] if isinstance(burnout_pred['burnout_risk_score'], (int, float)) else burnout_pred['burnout_risk_score']
                
                if y_test.shape[1] > 0:
                    actual_burnout = y_test[:, 0].astype(int)
                    
                    # Convert risk scores to binary predictions (threshold = 0.5)
                    binary_predictions = [1 if score > 0.5 else 0 for score in risk_scores]
                    
                    accuracy = np.mean(binary_predictions == actual_burnout)
                    
                    evaluation_results['classification_performance'] = {
                        'accuracy': accuracy,
                        'avg_risk_score': float(np.mean(risk_scores)),
                        'risk_score_std': float(np.std(risk_scores)),
                        'actual_burnout_rate': float(np.mean(actual_burnout)),
                        'predicted_burnout_rate': float(np.mean(binary_predictions))
                    }
            
            evaluation_results['status'] = 'completed'
            
        except Exception as e:
            evaluation_results['status'] = 'failed'
            evaluation_results['error'] = str(e)
        
        return evaluation_results


if __name__ == "__main__":
    # Test training pipeline
    trainer = ModelTrainer()
    
    print("Running training pipeline test...")
    results = trainer.run_full_training_pipeline(days=7)  # Use 7 days for testing
    
    print("\nTraining pipeline results:")
    print(json.dumps(results, indent=2, default=str))
    
    # Test evaluation
    if results['status'] == 'completed':
        print("\nEvaluating model performance...")
        eval_results = trainer.evaluate_model_performance(test_days=3)
        print("Evaluation results:", json.dumps(eval_results, indent=2, default=str))
