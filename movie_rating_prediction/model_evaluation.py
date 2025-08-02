"""
Model Evaluation Module for Movie Rating Prediction

This module handles model training, evaluation, and performance metrics
for movie rating prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A class for training and evaluating movie rating prediction models.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, features_df: pd.DataFrame, target: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            features_df: DataFrame with features
            target: Target variable (ratings)
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
        """
        print("=== Data Preparation ===")
        print(f"Total samples: {len(features_df)}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_df, target, test_size=test_size, random_state=random_state
        )
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Feature dimensions: {self.X_train.shape[1]}")
    
    def initialize_models(self) -> None:
        """Initialize different regression models for comparison."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        print(f"Initialized {len(self.models)} models for evaluation")
    
    def train_models(self) -> None:
        """Train all initialized models."""
        if self.X_train is None:
            raise ValueError("Data not prepared. Please run prepare_data() first.")
        
        print("=== Model Training ===")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models and return performance metrics.
        
        Returns:
            Dictionary with model performance metrics
        """
        if self.X_test is None:
            raise ValueError("Data not prepared. Please run prepare_data() first.")
        
        print("=== Model Evaluation ===")
        
        self.results = {}
        
        for name, model in self.models.items():
            try:
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(self.y_train, y_pred_train)
                test_metrics = self._calculate_metrics(self.y_test, y_pred_test)
                
                self.results[name] = {
                    'train_mse': train_metrics['mse'],
                    'train_mae': train_metrics['mae'],
                    'train_r2': train_metrics['r2'],
                    'test_mse': test_metrics['mse'],
                    'test_mae': test_metrics['mae'],
                    'test_r2': test_metrics['r2'],
                    'overfitting': train_metrics['r2'] - test_metrics['r2']
                }
                
                print(f"✓ {name} evaluated")
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {str(e)}")
                self.results[name] = None
        
        return self.results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    def cross_validate_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Please run prepare_data() first.")
        
        print(f"=== Cross-Validation ({cv_folds} folds) ===")
        
        cv_results = {}
        
        for name, model in self.models.items():
            try:
                # Perform cross-validation
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv_folds, scoring='r2')
                
                cv_results[name] = {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'scores': scores.tolist()
                }
                
                print(f"✓ {name}: R² = {scores.mean():.4f} (±{scores.std():.4f})")
                
            except Exception as e:
                print(f"✗ Error in CV for {name}: {str(e)}")
                cv_results[name] = None
        
        return cv_results
    
    def select_best_model(self, metric: str = 'test_r2') -> str:
        """
        Select the best model based on specified metric.
        
        Args:
            metric: Metric to use for selection
            
        Returns:
            Name of the best model
        """
        if not self.results:
            raise ValueError("Models not evaluated. Please run evaluate_models() first.")
        
        valid_results = {name: results for name, results in self.results.items() 
                        if results is not None}
        
        if not valid_results:
            raise ValueError("No valid model results available.")
        
        # For R², higher is better; for MSE/MAE, lower is better
        if metric in ['test_r2', 'train_r2']:
            best_name = max(valid_results.keys(), key=lambda x: valid_results[x][metric])
        else:
            best_name = min(valid_results.keys(), key=lambda x: valid_results[x][metric])
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"Best model selected: {best_name} (metric: {metric})")
        return best_name
    
    def hyperparameter_tuning(self, model_name: str = 'Random Forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model to tune
            
        Returns:
            Dictionary with tuning results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Please run prepare_data() first.")
        
        print(f"=== Hyperparameter Tuning for {model_name} ===")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Ridge Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return {}
        
        # Get base model
        if model_name == 'Random Forest':
            base_model = RandomForestRegressor(random_state=42)
        elif model_name == 'Ridge Regression':
            base_model = Ridge()
        elif model_name == 'Lasso Regression':
            base_model = Lasso()
        else:
            print(f"Model {model_name} not supported for tuning")
            return {}
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, 
            param_grids[model_name], 
            cv=3, 
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best parameters
        self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'model': grid_search.best_estimator_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """
        Create visualization comparing model performances.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.results:
            raise ValueError("Models not evaluated. Please run evaluate_models() first.")
        
        # Prepare data for plotting
        valid_results = {name: results for name, results in self.results.items() 
                        if results is not None}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        models = list(valid_results.keys())
        train_r2 = [valid_results[model]['train_r2'] for model in models]
        test_r2 = [valid_results[model]['test_r2'] for model in models]
        test_mae = [valid_results[model]['test_mae'] for model in models]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² comparison
        x_pos = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x_pos - width/2, train_r2, width, label='Train R²', alpha=0.8)
        axes[0].bar(x_pos + width/2, test_r2, width, label='Test R²', alpha=0.8)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('Model Performance: R² Score')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE comparison
        axes[1].bar(models, test_mae, color='coral', alpha=0.8)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Model Performance: Test MAE')
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        # Overfitting analysis
        overfitting = [valid_results[model]['overfitting'] for model in models]
        colors = ['red' if x > 0.1 else 'green' for x in overfitting]
        axes[2].bar(models, overfitting, color=colors, alpha=0.8)
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('Overfitting (Train R² - Test R²)')
        axes[2].set_title('Overfitting Analysis')
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def analyze_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Analyze feature importance using the best model (if it supports feature importance).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Please run select_best_model() first.")
        
        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"Model {self.best_model_name} does not support feature importance analysis")
            return pd.DataFrame()
        
        feature_names = self.X_train.columns
        importances = self.best_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"=== Top {top_n} Feature Importances ({self.best_model_name}) ===")
        print(importance_df.head(top_n))
        
        return importance_df.head(top_n)
    
    def generate_performance_report(self) -> str:
        """
        Generate a comprehensive performance report.
        
        Returns:
            String with performance report
        """
        if not self.results:
            return "No evaluation results available."
        
        report = "=== Model Performance Report ===\n\n"
        
        valid_results = {name: results for name, results in self.results.items() 
                        if results is not None}
        
        for model_name, metrics in valid_results.items():
            report += f"{model_name}:\n"
            report += f"  Test R²: {metrics['test_r2']:.4f}\n"
            report += f"  Test MAE: {metrics['test_mae']:.4f}\n"
            report += f"  Test RMSE: {np.sqrt(metrics['test_mse']):.4f}\n"
            report += f"  Overfitting: {metrics['overfitting']:.4f}\n\n"
        
        if self.best_model_name:
            report += f"Best Model: {self.best_model_name}\n"
            best_metrics = valid_results[self.best_model_name]
            report += f"Best Test R²: {best_metrics['test_r2']:.4f}\n"
        
        return report


def main():
    """
    Demonstration of the ModelEvaluator functionality.
    """
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    print("=== Model Evaluation Demo ===")
    
    # Load and prepare data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(sample_data=True)
    cleaned_data = preprocessor.clean_data()
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    features_df, target = feature_engineer.engineer_features(cleaned_data)
    scaled_features = feature_engineer.scale_features(features_df)
    
    # Initialize model evaluator
    evaluator = ModelEvaluator()
    
    # Prepare data
    evaluator.prepare_data(scaled_features, target)
    
    # Initialize and train models
    evaluator.initialize_models()
    evaluator.train_models()
    
    # Evaluate models
    results = evaluator.evaluate_models()
    
    # Cross-validation
    cv_results = evaluator.cross_validate_models()
    
    # Select best model
    best_model = evaluator.select_best_model()
    
    # Generate report
    report = evaluator.generate_performance_report()
    print(report)
    
    # Plot comparison
    evaluator.plot_model_comparison()
    
    # Feature importance
    importance_df = evaluator.analyze_feature_importance()


if __name__ == "__main__":
    main()