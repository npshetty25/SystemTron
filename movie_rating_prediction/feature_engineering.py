"""
Feature Engineering Module for Movie Rating Prediction

This module handles feature extraction and engineering for movie rating prediction.
It includes functions for encoding categorical variables, creating new features,
and scaling/normalizing data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    A class for engineering features from movie data for rating prediction.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        
    def engineer_features(self, data: pd.DataFrame, target_col: str = 'rating') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineer features from the input data.
        
        Args:
            data: Input DataFrame with movie data
            target_col: Name of the target column (rating)
            
        Returns:
            Tuple of (features_df, target_series)
        """
        print("=== Feature Engineering ===")
        print(f"Input data shape: {data.shape}")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Separate features and target
        if target_col in df.columns:
            target = df[target_col].copy()
            df = df.drop(columns=[target_col])
        else:
            target = None
            print(f"Warning: Target column '{target_col}' not found in data")
        
        # Initialize feature dataframe
        features_df = pd.DataFrame()
        
        # 1. Numerical features (direct copy)
        numerical_cols = ['year', 'duration', 'budget']
        for col in numerical_cols:
            if col in df.columns:
                features_df[col] = df[col]
                print(f"Added numerical feature: {col}")
        
        # 2. Create derived numerical features
        if 'year' in df.columns:
            current_year = 2024
            features_df['movie_age'] = current_year - df['year']
            print("Added derived feature: movie_age")
        
        if 'budget' in df.columns:
            # Create budget categories
            features_df['budget_log'] = np.log1p(df['budget'])  # Log transform for skewed budget data
            features_df['is_high_budget'] = (df['budget'] > df['budget'].median()).astype(int)
            print("Added derived features: budget_log, is_high_budget")
        
        if 'duration' in df.columns:
            # Create duration categories
            features_df['is_long_movie'] = (df['duration'] > 120).astype(int)
            features_df['duration_category'] = pd.cut(df['duration'], 
                                                     bins=[0, 90, 120, 150, float('inf')], 
                                                     labels=['short', 'medium', 'long', 'very_long'])
            print("Added derived features: is_long_movie, duration_category")
        
        # 3. Categorical feature encoding
        categorical_cols = ['genre', 'director', 'actor']
        
        for col in categorical_cols:
            if col in df.columns:
                # Label encoding for high-cardinality categorical features
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Transform using existing encoder
                    features_df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
                
                # One-hot encoding for top categories (to avoid too many features)
                top_categories = df[col].value_counts().head(10).index.tolist()
                for category in top_categories:
                    features_df[f'{col}_{category}'] = (df[col] == category).astype(int)
                
                print(f"Added encoded features for {col}: label encoding + top-10 one-hot")
        
        # 4. Feature interactions
        if 'director_encoded' in features_df.columns and 'genre_encoded' in features_df.columns:
            features_df['director_genre_interaction'] = features_df['director_encoded'] * features_df['genre_encoded']
            print("Added feature interaction: director_genre_interaction")
        
        if 'budget_log' in features_df.columns and 'movie_age' in features_df.columns:
            features_df['budget_age_ratio'] = features_df['budget_log'] / (features_df['movie_age'] + 1)
            print("Added feature interaction: budget_age_ratio")
        
        # 5. Handle duration_category (convert to numerical)
        if 'duration_category' in features_df.columns:
            duration_mapping = {'short': 1, 'medium': 2, 'long': 3, 'very_long': 4}
            features_df['duration_category_num'] = features_df['duration_category'].map(duration_mapping)
            features_df = features_df.drop('duration_category', axis=1)
        
        # 6. Statistical features by group
        if 'director_encoded' in features_df.columns and target is not None:
            # Average rating by director (using target leakage prevention)
            director_stats = df.groupby('director').agg({
                'year': ['mean', 'count'],
                'duration': 'mean',
                'budget': 'mean'
            }).round(2)
            
            director_stats.columns = ['_'.join(col).strip() for col in director_stats.columns]
            director_stats = director_stats.reset_index()
            
            # Merge director statistics
            temp_df = df[['director']].copy()
            temp_df = temp_df.merge(director_stats, on='director', how='left')
            
            for col in director_stats.columns[1:]:  # Skip 'director' column
                features_df[f'director_{col}'] = temp_df[col]
            
            print("Added director-based statistical features")
        
        # Store feature names
        self.feature_names = list(features_df.columns)
        
        print(f"Final feature set shape: {features_df.shape}")
        print(f"Total features created: {len(self.feature_names)}")
        
        return features_df, target
    
    def scale_features(self, features_df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            features_df: DataFrame with features
            fit: Whether to fit the scaler (True for training data, False for test data)
            
        Returns:
            DataFrame with scaled features
        """
        print("=== Feature Scaling ===")
        
        # Identify numerical columns (exclude binary/categorical encoded features)
        numerical_cols = []
        for col in features_df.columns:
            if features_df[col].dtype in ['int64', 'float64']:
                # Check if it's not a binary feature
                unique_vals = features_df[col].nunique()
                if unique_vals > 2:  # Not binary
                    numerical_cols.append(col)
        
        print(f"Scaling {len(numerical_cols)} numerical features: {numerical_cols}")
        
        scaled_df = features_df.copy()
        
        if numerical_cols:
            if fit:
                scaled_df[numerical_cols] = self.scaler.fit_transform(features_df[numerical_cols])
                self.is_fitted = True
                print("Fitted and transformed features")
            else:
                if not self.is_fitted:
                    raise ValueError("Scaler not fitted. Please fit on training data first.")
                scaled_df[numerical_cols] = self.scaler.transform(features_df[numerical_cols])
                print("Transformed features using existing scaler")
        
        return scaled_df
    
    def create_feature_importance_names(self) -> List[str]:
        """
        Get list of feature names for interpretation.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def get_feature_stats(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the engineered features.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'total_features': len(features_df.columns),
            'numerical_features': len(features_df.select_dtypes(include=[np.number]).columns),
            'binary_features': sum(1 for col in features_df.columns 
                                 if features_df[col].nunique() == 2),
            'feature_names': list(features_df.columns),
            'missing_values': features_df.isnull().sum().sum(),
            'data_shape': features_df.shape
        }
        
        return stats
    
    def select_top_features(self, features_df: pd.DataFrame, target: pd.Series, 
                           method: str = 'correlation', top_k: int = 20) -> pd.DataFrame:
        """
        Select top k features based on specified method.
        
        Args:
            features_df: DataFrame with features
            target: Target variable
            method: Feature selection method ('correlation', 'variance')
            top_k: Number of top features to select
            
        Returns:
            DataFrame with selected features
        """
        print(f"=== Feature Selection ({method}) ===")
        
        if method == 'correlation':
            # Calculate correlation with target
            correlations = features_df.corrwith(target).abs().sort_values(ascending=False)
            top_features = correlations.head(top_k).index.tolist()
            print(f"Selected top {len(top_features)} features by correlation")
            
        elif method == 'variance':
            # Select features with highest variance
            variances = features_df.var().sort_values(ascending=False)
            top_features = variances.head(top_k).index.tolist()
            print(f"Selected top {len(top_features)} features by variance")
            
        else:
            raise ValueError("Method must be 'correlation' or 'variance'")
        
        selected_df = features_df[top_features]
        print(f"Feature selection completed. Shape: {selected_df.shape}")
        
        return selected_df


def main():
    """
    Demonstration of the FeatureEngineer functionality.
    """
    # Create sample data for demonstration
    from data_preprocessing import DataPreprocessor
    
    print("=== Feature Engineering Demo ===")
    
    # Load and clean data
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(sample_data=True)
    cleaned_data = preprocessor.clean_data()
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    features_df, target = feature_engineer.engineer_features(cleaned_data)
    
    # Scale features
    scaled_features = feature_engineer.scale_features(features_df)
    
    # Get feature statistics
    stats = feature_engineer.get_feature_stats(scaled_features)
    print(f"\n=== Feature Statistics ===")
    for key, value in stats.items():
        if key != 'feature_names':
            print(f"{key}: {value}")
    
    # Feature selection
    if target is not None:
        selected_features = feature_engineer.select_top_features(
            scaled_features, target, method='correlation', top_k=15
        )
        print(f"\nSelected features shape: {selected_features.shape}")
    
    print(f"\n=== Sample of engineered features ===")
    print(scaled_features.head())


if __name__ == "__main__":
    main()