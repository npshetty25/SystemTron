"""
Data Preprocessing Module for Movie Rating Prediction

This module handles data loading, cleaning, and preprocessing for movie rating prediction.
It includes functions for handling missing values, data validation, and basic data transformations.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional, List


class DataPreprocessor:
    """
    A class for preprocessing movie data for rating prediction.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.data = None
        self.cleaned_data = None
        
    def load_data(self, data_path: Optional[str] = None, sample_data: bool = True) -> pd.DataFrame:
        """
        Load movie data from file or create sample data for demonstration.
        
        Args:
            data_path: Path to the dataset file (CSV format expected)
            sample_data: If True and no data_path provided, creates sample data
            
        Returns:
            DataFrame containing the loaded data
        """
        if data_path:
            try:
                self.data = pd.read_csv(data_path)
                print(f"Data loaded successfully from {data_path}")
                print(f"Shape: {self.data.shape}")
            except FileNotFoundError:
                print(f"File {data_path} not found. Creating sample data instead.")
                self.data = self._create_sample_data()
        else:
            if sample_data:
                self.data = self._create_sample_data()
                print("Sample data created for demonstration")
            else:
                raise ValueError("Either provide data_path or set sample_data=True")
                
        return self.data
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample movie data for demonstration purposes.
        
        Returns:
            DataFrame with sample movie data
        """
        np.random.seed(42)
        
        # Sample data with various genres, directors, and actors
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Thriller', 'Sci-Fi', 'Fantasy']
        directors = ['Christopher Nolan', 'Steven Spielberg', 'Martin Scorsese', 'Quentin Tarantino', 
                    'David Fincher', 'Ridley Scott', 'James Cameron', 'Tim Burton', 'Coen Brothers', 'Rian Johnson']
        actors = ['Leonardo DiCaprio', 'Tom Hanks', 'Robert Downey Jr.', 'Scarlett Johansson',
                 'Morgan Freeman', 'Brad Pitt', 'Johnny Depp', 'Will Smith', 'Jennifer Lawrence', 'Christian Bale']
        
        n_samples = 1000
        
        data = {
            'title': [f'Movie_{i}' for i in range(n_samples)],
            'genre': np.random.choice(genres, n_samples),
            'director': np.random.choice(directors, n_samples),
            'actor': np.random.choice(actors, n_samples),
            'year': np.random.randint(1990, 2024, n_samples),
            'duration': np.random.randint(90, 180, n_samples),
            'budget': np.random.exponential(50, n_samples),  # In millions
            'rating': None  # Will be generated based on features
        }
        
        df = pd.DataFrame(data)
        
        # Generate ratings based on features (for realistic simulation)
        # Higher ratings for certain directors, genres, etc.
        rating_base = 5.0
        
        # Director influence
        director_bonus = {
            'Christopher Nolan': 1.5, 'Steven Spielberg': 1.3, 'Martin Scorsese': 1.4,
            'Quentin Tarantino': 1.2, 'David Fincher': 1.1
        }
        
        # Genre influence  
        genre_bonus = {
            'Drama': 0.8, 'Thriller': 0.6, 'Sci-Fi': 0.4, 'Action': 0.2
        }
        
        ratings = []
        for _, row in df.iterrows():
            rating = rating_base
            rating += director_bonus.get(row['director'], 0)
            rating += genre_bonus.get(row['genre'], 0)
            rating += (row['duration'] - 120) * 0.01  # Slight duration influence
            rating += (row['budget'] - 50) * 0.005    # Slight budget influence
            rating += np.random.normal(0, 0.5)        # Random noise
            rating = max(1.0, min(10.0, rating))      # Clamp between 1-10
            ratings.append(round(rating, 1))
        
        df['rating'] = ratings
        
        # Introduce some missing values for realistic preprocessing
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices[:len(missing_indices)//3], 'budget'] = np.nan
        df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'duration'] = np.nan
        df.loc[missing_indices[2*len(missing_indices)//3:], 'year'] = np.nan
        
        return df
    
    def explore_data(self) -> None:
        """
        Display basic information about the dataset.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
            
        print("=== Data Exploration ===")
        print(f"Shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
        
        if 'rating' in self.data.columns:
            print(f"\nRating distribution:")
            print(self.data['rating'].value_counts().sort_index())
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and data quality issues.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        self.cleaned_data = self.data.copy()
        
        print("=== Data Cleaning ===")
        print(f"Original shape: {self.cleaned_data.shape}")
        
        # Handle missing values
        initial_missing = self.cleaned_data.isnull().sum().sum()
        print(f"Initial missing values: {initial_missing}")
        
        # Fill missing numerical values with median
        numerical_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.cleaned_data[col].isnull().any():
                median_val = self.cleaned_data[col].median()
                self.cleaned_data[col].fillna(median_val, inplace=True)
                print(f"Filled missing {col} with median: {median_val}")
        
        # Fill missing categorical values with mode
        categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.cleaned_data[col].isnull().any():
                mode_val = self.cleaned_data[col].mode()[0] if not self.cleaned_data[col].mode().empty else 'Unknown'
                self.cleaned_data[col].fillna(mode_val, inplace=True)
                print(f"Filled missing {col} with mode: {mode_val}")
        
        # Remove duplicates
        initial_rows = len(self.cleaned_data)
        self.cleaned_data.drop_duplicates(inplace=True)
        final_rows = len(self.cleaned_data)
        print(f"Removed {initial_rows - final_rows} duplicate rows")
        
        # Data validation
        if 'rating' in self.cleaned_data.columns:
            # Ensure ratings are within valid range
            invalid_ratings = (self.cleaned_data['rating'] < 1) | (self.cleaned_data['rating'] > 10)
            if invalid_ratings.any():
                print(f"Found {invalid_ratings.sum()} invalid ratings, clamping to 1-10 range")
                self.cleaned_data.loc[self.cleaned_data['rating'] < 1, 'rating'] = 1
                self.cleaned_data.loc[self.cleaned_data['rating'] > 10, 'rating'] = 10
        
        if 'year' in self.cleaned_data.columns:
            # Ensure years are reasonable
            current_year = 2024
            invalid_years = (self.cleaned_data['year'] < 1900) | (self.cleaned_data['year'] > current_year)
            if invalid_years.any():
                print(f"Found {invalid_years.sum()} invalid years")
                median_year = self.cleaned_data['year'].median()
                self.cleaned_data.loc[invalid_years, 'year'] = median_year
        
        final_missing = self.cleaned_data.isnull().sum().sum()
        print(f"Final missing values: {final_missing}")
        print(f"Final shape: {self.cleaned_data.shape}")
        
        return self.cleaned_data
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned data. If data hasn't been cleaned yet, clean it first.
        
        Returns:
            Cleaned DataFrame
        """
        if self.cleaned_data is None:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            return self.clean_data()
        return self.cleaned_data
    
    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save the cleaned data to a CSV file.
        
        Args:
            output_path: Path where to save the cleaned data
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
            
        self.cleaned_data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")


def main():
    """
    Demonstration of the DataPreprocessor functionality.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load sample data
    data = preprocessor.load_data(sample_data=True)
    
    # Explore the data
    preprocessor.explore_data()
    
    # Clean the data
    cleaned_data = preprocessor.clean_data()
    
    print("\n=== Sample of cleaned data ===")
    print(cleaned_data.head())


if __name__ == "__main__":
    main()