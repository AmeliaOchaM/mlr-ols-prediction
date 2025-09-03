"""
Polynomial Features Generator
Generates polynomial features (X1², X1*X2, etc.) for independent variables
Part of the preprocessing pipeline between splitting and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement
import os

class PolynomialFeatureGenerator:
    def __init__(self, degree=2, interaction_only=False, include_bias=False):
        """
        Initialize polynomial feature generator
        
        Parameters:
        degree (int): Maximum degree of polynomial features (default: 2)
        interaction_only (bool): If True, only interaction features (X1*X2) without powers (X1²)
        include_bias (bool): If True, include bias column (constant 1s)
        """
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly_transformer = None
        self.feature_names = None
        self.target_column = 'RR'  # Target variable: rainfall
        
        # Expected independent variables (6 features)
        self.expected_features = ['TAVG', 'FF_AVG', 'RH_AVG', 'SS', 'DDD_X', 'DDD_CAR']
        
        # Mapping untuk DDD_CAR (arah angin) ke nilai numerik
        self.wind_direction_mapping = {
            'N': 0,      'NE': 45,    'E': 90,     'SE': 135,
            'S': 180,    'SW': 225,   'W': 270,    'NW': 315,
            'C': 0,      'CALM': 0,   'VAR': 180,
            'NNE': 22.5, 'ENE': 67.5, 'ESE': 112.5, 'SSE': 157.5,
            'SSW': 202.5, 'WSW': 247.5, 'WNW': 292.5, 'NNW': 337.5
        }
        
    def load_data(self, file_path):
        """Load CSV data"""
        try:
            print(f"Loading data from: {file_path}")
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Preprocess data to handle categorical columns like DDD_CAR"""
        print("Preprocessing data...")
        data_processed = data.copy()
        
        # Convert DDD_CAR to numeric if it's categorical
        if 'DDD_CAR' in data_processed.columns:
            if data_processed['DDD_CAR'].dtype == 'object':
                print("Converting DDD_CAR wind directions to numeric...")
                unique_dirs = data_processed['DDD_CAR'].unique()
                print(f"Found directions: {unique_dirs}")
                
                data_processed['DDD_CAR'] = data_processed['DDD_CAR'].map(self.wind_direction_mapping)
                
                # Check for unmapped values
                unmapped = data_processed['DDD_CAR'].isnull().sum()
                if unmapped > 0:
                    print(f"Warning: {unmapped} wind directions could not be mapped")
                    # Fill with median
                    median_dir = data_processed['DDD_CAR'].median()
                    data_processed['DDD_CAR'].fillna(median_dir, inplace=True)
        
        # Handle other missing values
        for col in data_processed.columns:
            if data_processed[col].dtype in ['int64', 'float64']:
                missing = data_processed[col].isnull().sum()
                if missing > 0:
                    print(f"Filling {missing} missing values in {col}")
                    data_processed[col].fillna(data_processed[col].median(), inplace=True)
        
        print("Data preprocessing completed!")
        return data_processed
    
    def separate_features_target(self, data):
        """Separate features (X) and target (y)"""
        print(f"Dataset columns: {list(data.columns)}")
        
        if self.target_column in data.columns:
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]
            
            # Verify we have the expected 6 independent variables
            actual_features = list(X.columns)
            print(f"Expected features: {self.expected_features}")
            print(f"Actual features: {actual_features}")
            
            if len(actual_features) == 6:
                print("✅ Correct number of independent variables (6)")
            else:
                print(f"⚠️  Expected 6 features, found {len(actual_features)}")
            
            print(f"Features (X): {X.shape}")
            print(f"Target (y): {y.shape}")
            return X, y
        else:
            print(f"Warning: Target column '{self.target_column}' not found!")
            print("Treating all columns as features")
            return data, None
    
    def generate_polynomial_features(self, X_train, X_test=None):
        """
        Generate polynomial features for training and optionally test data
        
        Parameters:
        X_train (DataFrame): Training features
        X_test (DataFrame): Test features (optional)
        
        Returns:
        X_train_poly, X_test_poly (or None if X_test not provided)
        """
        print(f"\n{'='*60}")
        print("GENERATING POLYNOMIAL FEATURES")
        print(f"{'='*60}")
        print(f"Degree: {self.degree}")
        print(f"Interaction only: {self.interaction_only}")
        print(f"Include bias: {self.include_bias}")
        
        # Initialize polynomial transformer
        self.poly_transformer = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        
        # Fit and transform training data
        print(f"\nOriginal features: {X_train.shape[1]}")
        X_train_poly_array = self.poly_transformer.fit_transform(X_train)
        
        # Get feature names
        self.feature_names = self.poly_transformer.get_feature_names_out(X_train.columns)
        
        # Convert to DataFrame
        X_train_poly = pd.DataFrame(
            X_train_poly_array, 
            columns=self.feature_names,
            index=X_train.index
        )
        
        print(f"Polynomial features: {X_train_poly.shape[1]}")
        print(f"New features added: {X_train_poly.shape[1] - X_train.shape[1]}")
        
        # Show some example features
        print(f"\nExample polynomial features:")
        for i, name in enumerate(self.feature_names[:10]):  # Show first 10
            print(f"  {name}")
        if len(self.feature_names) > 10:
            print(f"  ... and {len(self.feature_names) - 10} more")
        
        # Transform test data if provided
        X_test_poly = None
        if X_test is not None:
            print(f"\nTransforming test data...")
            X_test_poly_array = self.poly_transformer.transform(X_test)
            X_test_poly = pd.DataFrame(
                X_test_poly_array,
                columns=self.feature_names,
                index=X_test.index
            )
            print(f"Test polynomial features: {X_test_poly.shape}")
        
        return X_train_poly, X_test_poly
    
    def save_polynomial_data(self, X_poly, y, output_path):
        """Save polynomial features with target variable"""
        try:
            # Combine features and target
            if y is not None:
                data_poly = pd.concat([X_poly, y], axis=1)
            else:
                data_poly = X_poly
            
            print(f"Saving polynomial data to: {output_path}")
            data_poly.to_csv(output_path, index=False)
            print(f"✅ Saved successfully! Shape: {data_poly.shape}")
            
            return output_path
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return None
    
    def show_feature_statistics(self, X_original, X_poly):
        """Show statistics about generated features"""
        print(f"\n{'='*60}")
        print("FEATURE STATISTICS")
        print(f"{'='*60}")
        
        print(f"Original features: {X_original.shape[1]}")
        print(f"Polynomial features: {X_poly.shape[1]}")
        print(f"Features added: {X_poly.shape[1] - X_original.shape[1]}")
        
        # Categorize features
        original_features = list(X_original.columns)
        squared_features = [name for name in self.feature_names if '^2' in name]
        interaction_features = [name for name in self.feature_names if '*' in name and '^' not in name]
        higher_order = [name for name in self.feature_names 
                       if ('^3' in name or '^4' in name or name.count('*') > 1)]
        
        print(f"\nFeature breakdown:")
        print(f"  Original features: {len(original_features)}")
        print(f"  Squared features (X²): {len(squared_features)}")
        print(f"  Interaction features (X*Y): {len(interaction_features)}")
        if higher_order:
            print(f"  Higher order features: {len(higher_order)}")
        
        # Show examples of each type
        if squared_features:
            print(f"\nSquared features examples: {squared_features[:3]}")
        if interaction_features:
            print(f"Interaction features examples: {interaction_features[:3]}")
        
        # Calculate expected polynomial features for 6 variables
        expected_poly_count = 6  # Original features
        if not self.interaction_only:
            expected_poly_count += 6  # Squared terms: TAVG², FF_AVG², etc.
        expected_poly_count += 15  # Interaction terms: TAVG*FF_AVG, TAVG*RH_AVG, etc. (6 choose 2)
        
        print(f"\nExpected features for 6 variables (degree {self.degree}):")
        print(f"  - Original: 6")
        if not self.interaction_only:
            print(f"  - Squared: 6")
        print(f"  - Interactions: 15")
        print(f"  - Total expected: {expected_poly_count}")
        print(f"  - Actual generated: {X_poly.shape[1]}")
        
        if X_poly.shape[1] == expected_poly_count:
            print("✅ Feature count matches expectation!")
        else:
            print("⚠️  Feature count differs from expectation")

def process_single_file(input_file, output_file, degree=2, interaction_only=False):
    """Process a single CSV file to generate polynomial features"""
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING: {input_file}")
    print(f"{'='*80}")
    
    # Initialize generator
    poly_gen = PolynomialFeatureGenerator(
        degree=degree, 
        interaction_only=interaction_only
    )
    
    # Load data
    data = poly_gen.load_data(input_file)
    if data is None:
        return False
    
    # Separate features and target
    X, y = poly_gen.separate_features_target(data)
    
    # Generate polynomial features
    X_poly, _ = poly_gen.generate_polynomial_features(X)
    
    # Show statistics
    poly_gen.show_feature_statistics(X, X_poly)
    
    # Save results
    result_file = poly_gen.save_polynomial_data(X_poly, y, output_file)
    
    if result_file:
        print(f"\n✅ SUCCESS!")
        print(f"Input: {input_file}")
        print(f"Output: {result_file}")
        return True
    else:
        print(f"\n❌ FAILED!")
        return False

def process_train_test_files(train_file, test_file, output_dir, degree=2, interaction_only=False):
    """Process both train and test files with same polynomial transformation"""
    
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING TRAIN-TEST PAIR")
    print(f"{'='*80}")
    
    # Check if files exist
    if not os.path.exists(train_file):
        print(f"Error: Train file '{train_file}' not found!")
        return False, False
        
    if not os.path.exists(test_file):
        print(f"Error: Test file '{test_file}' not found!")
        return False, False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    poly_gen = PolynomialFeatureGenerator(
        degree=degree, 
        interaction_only=interaction_only
    )
    
    # Load train data
    print("📊 Loading training data...")
    train_data = poly_gen.load_data(train_file)
    if train_data is None:
        return False, False
    
    # Preprocess train data
    train_data = poly_gen.preprocess_data(train_data)
    
    # Load test data
    print("📊 Loading test data...")
    test_data = poly_gen.load_data(test_file)
    if test_data is None:
        return False, False
    
    # Preprocess test data
    test_data = poly_gen.preprocess_data(test_data)
    
    # Separate features and targets
    X_train, y_train = poly_gen.separate_features_target(train_data)
    X_test, y_test = poly_gen.separate_features_target(test_data)
    
    # Generate polynomial features for both train and test
    print("🔄 Generating polynomial features...")
    X_train_poly, X_test_poly = poly_gen.generate_polynomial_features(X_train, X_test)
    
    # Show statistics
    poly_gen.show_feature_statistics(X_train, X_train_poly)
    
    # Generate output filenames
    train_output = os.path.join(output_dir, "train_polynomial.csv")
    test_output = os.path.join(output_dir, "test_polynomial.csv")
    
    # Save results
    print("💾 Saving polynomial train data...")
    train_success = poly_gen.save_polynomial_data(X_train_poly, y_train, train_output)
    
    print("💾 Saving polynomial test data...")
    test_success = poly_gen.save_polynomial_data(X_test_poly, y_test, test_output)
    
    if train_success and test_success:
        print(f"\n✅ BOTH FILES PROCESSED SUCCESSFULLY!")
        print(f"Train: {train_file} → {train_output}")
        print(f"Test: {test_file} → {test_output}")
        print(f"Features: {X_train.shape[1]} → {X_train_poly.shape[1]}")
        return True, True
    else:
        print(f"\n❌ PROCESSING FAILED!")
        return bool(train_success), bool(test_success)

def main():
    """Main function"""
    print("Polynomial Features Generator")
    print("=" * 40)
    
    # Configuration
    DEGREE = 2  # Polynomial degree (2 = X², X*Y)
    INTERACTION_ONLY = False  # False = include X², True = only X*Y
    
    # File paths
    train_file = "./pre-prosesing/input_data/train.csv"
    test_file = "./pre-prosesing/input_data/test.csv"
    output_dir = "./pre-prosesing/polynomial_data"
    
    print(f"Configuration:")
    print(f"  Polynomial degree: {DEGREE}")
    print(f"  Interaction only: {INTERACTION_ONLY}")
    print(f"  Output directory: {output_dir}")
    
    # Process train and test files together
    train_success, test_success = process_train_test_files(
        train_file, 
        test_file, 
        output_dir, 
        degree=DEGREE, 
        interaction_only=INTERACTION_ONLY
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Train file: {'✅ Success' if train_success else '❌ Failed'}")
    print(f"Test file: {'✅ Success' if test_success else '❌ Failed'}")
    
    if train_success and test_success:
        print("\n🎉 All polynomial features generated successfully!")
        print("📁 Next step: Run scaling.py on the polynomial data")
    else:
        print("\n⚠️  Some files failed to process. Please check the errors above.")

if __name__ == "__main__":
    main()