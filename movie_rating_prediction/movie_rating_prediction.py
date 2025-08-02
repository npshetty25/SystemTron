"""
Movie Rating Prediction System

This is the main script for the Movie Rating Prediction system that analyzes
historical movie data and predicts movie ratings based on features like genre,
director, and actors.

The system includes:
1. Data preprocessing and cleaning
2. Feature engineering and selection
3. Model training and evaluation
4. Prediction and analysis capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_evaluation import ModelEvaluator


class MovieRatingPredictor:
    """
    Main class for the Movie Rating Prediction system.
    """
    
    def __init__(self):
        """Initialize the Movie Rating Prediction system."""
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_evaluator = ModelEvaluator()
        
        self.raw_data = None
        self.cleaned_data = None
        self.features = None
        self.target = None
        self.is_trained = False
        
        print("Movie Rating Prediction System initialized")
    
    def load_and_preprocess_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and preprocess the movie data.
        
        Args:
            data_path: Path to the movie dataset (optional)
            
        Returns:
            Cleaned DataFrame
        """
        print("=== Step 1: Data Loading and Preprocessing ===")
        
        # Load data
        self.raw_data = self.preprocessor.load_data(data_path, sample_data=True)
        
        # Explore data
        self.preprocessor.explore_data()
        
        # Clean data
        self.cleaned_data = self.preprocessor.clean_data()
        
        print(f"Data preprocessing completed. Final shape: {self.cleaned_data.shape}")
        return self.cleaned_data
    
    def engineer_and_prepare_features(self) -> tuple:
        """
        Engineer features from the cleaned data.
        
        Returns:
            Tuple of (features_df, target_series)
        """
        print("\n=== Step 2: Feature Engineering ===")
        
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please run load_and_preprocess_data() first.")
        
        # Engineer features
        self.features, self.target = self.feature_engineer.engineer_features(self.cleaned_data)
        
        # Scale features
        self.features = self.feature_engineer.scale_features(self.features)
        
        # Get feature statistics
        stats = self.feature_engineer.get_feature_stats(self.features)
        print(f"Feature engineering completed. Total features: {stats['total_features']}")
        
        return self.features, self.target
    
    def train_and_evaluate_models(self) -> Dict[str, Any]:
        """
        Train and evaluate multiple models.
        
        Returns:
            Dictionary with evaluation results
        """
        print("\n=== Step 3: Model Training and Evaluation ===")
        
        if self.features is None or self.target is None:
            raise ValueError("Features not prepared. Please run engineer_and_prepare_features() first.")
        
        # Prepare data for training
        self.model_evaluator.prepare_data(self.features, self.target)
        
        # Initialize models
        self.model_evaluator.initialize_models()
        
        # Train models
        self.model_evaluator.train_models()
        
        # Evaluate models
        results = self.model_evaluator.evaluate_models()
        
        # Cross-validation
        cv_results = self.model_evaluator.cross_validate_models()
        
        # Select best model
        best_model_name = self.model_evaluator.select_best_model()
        
        self.is_trained = True
        
        print(f"Model training completed. Best model: {best_model_name}")
        
        return {
            'evaluation_results': results,
            'cv_results': cv_results,
            'best_model': best_model_name
        }
    
    def hyperparameter_optimization(self, model_name: str = 'Random Forest') -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for a specific model.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n=== Step 4: Hyperparameter Optimization for {model_name} ===")
        
        if not self.is_trained:
            raise ValueError("Models not trained. Please run train_and_evaluate_models() first.")
        
        # Perform hyperparameter tuning
        tuning_results = self.model_evaluator.hyperparameter_tuning(model_name)
        
        # Re-evaluate with tuned model
        tuned_model_name = f'{model_name}_tuned'
        if tuned_model_name in self.model_evaluator.models:
            # Quick evaluation of tuned model
            tuned_model = self.model_evaluator.models[tuned_model_name]
            y_pred = tuned_model.predict(self.model_evaluator.X_test)
            
            from sklearn.metrics import r2_score, mean_absolute_error
            tuned_r2 = r2_score(self.model_evaluator.y_test, y_pred)
            tuned_mae = mean_absolute_error(self.model_evaluator.y_test, y_pred)
            
            print(f"Tuned model performance - R²: {tuned_r2:.4f}, MAE: {tuned_mae:.4f}")
            
            tuning_results.update({
                'tuned_r2': tuned_r2,
                'tuned_mae': tuned_mae
            })
        
        return tuning_results
    
    def predict_movie_rating(self, movie_data: Dict[str, Any]) -> float:
        """
        Predict rating for a single movie.
        
        Args:
            movie_data: Dictionary with movie information
            
        Returns:
            Predicted rating
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Please run train_and_evaluate_models() first.")
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([movie_data])
        
        # Fill missing values with defaults if needed
        required_cols = ['genre', 'director', 'actor', 'year', 'duration', 'budget']
        for col in required_cols:
            if col not in input_df.columns:
                if col in ['year', 'duration', 'budget']:
                    input_df[col] = self.cleaned_data[col].median()
                else:
                    input_df[col] = self.cleaned_data[col].mode()[0]
        
        # Engineer features for the input
        try:
            features, _ = self.feature_engineer.engineer_features(input_df)
            scaled_features = self.feature_engineer.scale_features(features, fit=False)
            
            # Make prediction using best model
            prediction = self.model_evaluator.best_model.predict(scaled_features)[0]
            
            return round(prediction, 1)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return 5.0  # Default rating
    
    def batch_predict(self, movies_data: List[Dict[str, Any]]) -> List[float]:
        """
        Predict ratings for multiple movies.
        
        Args:
            movies_data: List of dictionaries with movie information
            
        Returns:
            List of predicted ratings
        """
        predictions = []
        for movie_data in movies_data:
            try:
                prediction = self.predict_movie_rating(movie_data)
                predictions.append(prediction)
            except Exception as e:
                print(f"Error predicting for movie {movie_data.get('title', 'Unknown')}: {str(e)}")
                predictions.append(5.0)  # Default rating
        
        return predictions
    
    def analyze_data_insights(self) -> None:
        """
        Generate and display data insights and visualizations.
        """
        print("\n=== Step 5: Data Analysis and Insights ===")
        
        if self.cleaned_data is None:
            raise ValueError("No data available for analysis.")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution
        axes[0, 0].hist(self.cleaned_data['rating'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Movie Ratings')
        axes[0, 0].set_xlabel('Rating')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Ratings by genre
        genre_ratings = self.cleaned_data.groupby('genre')['rating'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(genre_ratings)), genre_ratings.values)
        axes[0, 1].set_title('Average Rating by Genre')
        axes[0, 1].set_xlabel('Genre')
        axes[0, 1].set_ylabel('Average Rating')
        axes[0, 1].set_xticks(range(len(genre_ratings)))
        axes[0, 1].set_xticklabels(genre_ratings.index, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Ratings by year
        yearly_ratings = self.cleaned_data.groupby('year')['rating'].mean()
        axes[0, 2].plot(yearly_ratings.index, yearly_ratings.values, marker='o')
        axes[0, 2].set_title('Average Rating by Year')
        axes[0, 2].set_xlabel('Year')
        axes[0, 2].set_ylabel('Average Rating')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Budget vs Rating
        axes[1, 0].scatter(self.cleaned_data['budget'], self.cleaned_data['rating'], alpha=0.6)
        axes[1, 0].set_title('Budget vs Rating')
        axes[1, 0].set_xlabel('Budget (millions)')
        axes[1, 0].set_ylabel('Rating')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Duration vs Rating
        axes[1, 1].scatter(self.cleaned_data['duration'], self.cleaned_data['rating'], alpha=0.6)
        axes[1, 1].set_title('Duration vs Rating')
        axes[1, 1].set_xlabel('Duration (minutes)')
        axes[1, 1].set_ylabel('Rating')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Top directors by average rating
        director_ratings = self.cleaned_data.groupby('director')['rating'].agg(['mean', 'count'])
        director_ratings = director_ratings[director_ratings['count'] >= 3]  # At least 3 movies
        top_directors = director_ratings.sort_values('mean', ascending=False).head(10)
        
        axes[1, 2].barh(range(len(top_directors)), top_directors['mean'])
        axes[1, 2].set_title('Top Directors by Average Rating (min 3 movies)')
        axes[1, 2].set_xlabel('Average Rating')
        axes[1, 2].set_ylabel('Director')
        axes[1, 2].set_yticks(range(len(top_directors)))
        axes[1, 2].set_yticklabels(top_directors.index, fontsize=8)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print insights
        print("=== Key Insights ===")
        print(f"Average rating: {self.cleaned_data['rating'].mean():.2f}")
        print(f"Rating standard deviation: {self.cleaned_data['rating'].std():.2f}")
        print(f"Most common genre: {self.cleaned_data['genre'].mode()[0]}")
        print(f"Most prolific director: {self.cleaned_data['director'].value_counts().index[0]}")
        print(f"Average movie duration: {self.cleaned_data['duration'].mean():.0f} minutes")
        print(f"Average budget: ${self.cleaned_data['budget'].mean():.1f} million")
    
    def generate_comprehensive_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            String with the complete report
        """
        report = "# Movie Rating Prediction System - Comprehensive Report\n\n"
        
        # Data overview
        if self.cleaned_data is not None:
            report += "## Data Overview\n"
            report += f"- Total movies: {len(self.cleaned_data)}\n"
            report += f"- Features: {list(self.cleaned_data.columns)}\n"
            report += f"- Average rating: {self.cleaned_data['rating'].mean():.2f}\n"
            report += f"- Rating range: {self.cleaned_data['rating'].min():.1f} - {self.cleaned_data['rating'].max():.1f}\n\n"
        
        # Feature engineering
        if self.features is not None:
            stats = self.feature_engineer.get_feature_stats(self.features)
            report += "## Feature Engineering\n"
            report += f"- Total engineered features: {stats['total_features']}\n"
            report += f"- Numerical features: {stats['numerical_features']}\n"
            report += f"- Binary features: {stats['binary_features']}\n\n"
        
        # Model performance
        if self.is_trained:
            model_report = self.model_evaluator.generate_performance_report()
            report += "## Model Performance\n"
            report += model_report + "\n"
        
        # Feature importance
        if self.is_trained and hasattr(self.model_evaluator.best_model, 'feature_importances_'):
            importance_df = self.model_evaluator.analyze_feature_importance(top_n=10)
            if not importance_df.empty:
                report += "## Top Feature Importances\n"
                for _, row in importance_df.iterrows():
                    report += f"- {row['feature']}: {row['importance']:.4f}\n"
                report += "\n"
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def run_complete_pipeline(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete movie rating prediction pipeline.
        
        Args:
            data_path: Path to the movie dataset (optional)
            
        Returns:
            Dictionary with all results
        """
        print("🎬 Starting Movie Rating Prediction Pipeline\n")
        
        # Step 1: Load and preprocess data
        cleaned_data = self.load_and_preprocess_data(data_path)
        
        # Step 2: Engineer features
        features, target = self.engineer_and_prepare_features()
        
        # Step 3: Train and evaluate models
        model_results = self.train_and_evaluate_models()
        
        # Step 4: Optimize best model
        best_model_name = model_results['best_model']
        if best_model_name in ['Random Forest', 'Ridge Regression']:
            optimization_results = self.hyperparameter_optimization(best_model_name)
        else:
            optimization_results = {}
        
        # Step 5: Generate insights
        self.analyze_data_insights()
        
        # Step 6: Model comparison visualization
        self.model_evaluator.plot_model_comparison()
        
        # Step 7: Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        print("\n🎉 Pipeline completed successfully!")
        
        return {
            'data_shape': cleaned_data.shape,
            'feature_count': len(features.columns),
            'model_results': model_results,
            'optimization_results': optimization_results,
            'report': report
        }


def demo_predictions():
    """
    Demonstrate the prediction capabilities with sample movies.
    """
    print("\n=== Prediction Demo ===")
    
    # Sample movies for prediction
    sample_movies = [
        {
            'title': 'Epic Action Movie',
            'genre': 'Action',
            'director': 'Christopher Nolan',
            'actor': 'Leonardo DiCaprio',
            'year': 2023,
            'duration': 150,
            'budget': 200
        },
        {
            'title': 'Romantic Comedy',
            'genre': 'Romance',
            'director': 'Nancy Meyers',
            'actor': 'Jennifer Lawrence',
            'year': 2022,
            'duration': 105,
            'budget': 30
        },
        {
            'title': 'Indie Drama',
            'genre': 'Drama',
            'director': 'Martin Scorsese',
            'actor': 'Robert De Niro',
            'year': 2021,
            'duration': 135,
            'budget': 15
        }
    ]
    
    return sample_movies


def main():
    """
    Main function to run the Movie Rating Prediction system.
    """
    # Initialize the system
    predictor = MovieRatingPredictor()
    
    # Run the complete pipeline
    results = predictor.run_complete_pipeline()
    
    # Demo predictions
    sample_movies = demo_predictions()
    predictions = predictor.batch_predict(sample_movies)
    
    print("\n=== Sample Predictions ===")
    for movie, prediction in zip(sample_movies, predictions):
        print(f"{movie['title']}: {prediction}/10")
    
    # Print final summary
    print(f"\n=== Final Summary ===")
    print(f"Dataset size: {results['data_shape']}")
    print(f"Features engineered: {results['feature_count']}")
    print(f"Best model: {results['model_results']['best_model']}")
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()