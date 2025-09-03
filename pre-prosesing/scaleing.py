"""
CSV Data Scaling Script
Reads data_cleaned.csv and scales all columns including RR (with log transform)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import os

class CSVDataScaler:
    def __init__(self, file_path):
        """
        Initialize with CSV file path
        """
        self.file_path = file_path
        self.raw_data = None
        self.cleaned_data = None
        self.scaled_data = None
        self.scaler = None
        
        # No more ignored columns - RR will be transformed then scaled
        self.ignore_columns = []  
        
        # Mapping untuk DDD_CAR (arah angin) ke nilai numerik
        self.wind_direction_mapping = {
            'N': 0,      'NE': 45,    'E': 90,     'SE': 135,
            'S': 180,    'SW': 225,   'W': 270,    'NW': 315,
            'C': 0,      'CALM': 0,   'VAR': 180,
            'NNE': 22.5, 'ENE': 67.5, 'ESE': 112.5, 'SSE': 157.5,
            'SSW': 202.5, 'WSW': 247.5, 'WNW': 292.5, 'NNW': 337.5
        }
        
    def load_data(self):
        """Load CSV data"""
        try:
            print(f"Loading CSV file: {self.file_path}")
            self.raw_data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully! Shape: {self.raw_data.shape}")
            print(f"Columns: {list(self.raw_data.columns)}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Data exploration"""
        if self.raw_data is None:
            print("Data belum di-load!")
            return
        
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"Shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        
        # Analyze each column
        for col in self.raw_data.columns:
            print(f"\n{col}:")
            print(f"  Type: {self.raw_data[col].dtype}")
            print(f"  Missing: {self.raw_data[col].isnull().sum()}")
            
            if self.raw_data[col].dtype in ['int64', 'float64']:
                print(f"  Range: {self.raw_data[col].min()} to {self.raw_data[col].max()}")
                print(f"  Mean: {self.raw_data[col].mean():.2f}")
            else:
                unique_vals = self.raw_data[col].unique()
                print(f"  Unique values: {unique_vals}")
        
        print(f"\nColumns to ignore: {self.ignore_columns}")
    
    def clean_data(self):
        """Clean and prepare data"""
        if self.raw_data is None:
            print("Data belum di-load!")
            return
        
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        self.cleaned_data = self.raw_data.copy()
        
        # Convert DDD_CAR to numeric if it's categorical
        if 'DDD_CAR' in self.cleaned_data.columns:
            if self.cleaned_data['DDD_CAR'].dtype == 'object':
                print("Converting DDD_CAR wind directions to numeric...")
                unique_dirs = self.cleaned_data['DDD_CAR'].unique()
                print(f"Found directions: {unique_dirs}")
                
                self.cleaned_data['DDD_CAR'] = self.cleaned_data['DDD_CAR'].map(self.wind_direction_mapping)
                
                # Check for unmapped values
                unmapped = self.cleaned_data['DDD_CAR'].isnull().sum()
                if unmapped > 0:
                    print(f"Warning: {unmapped} wind directions could not be mapped")
                    # Fill with median
                    median_dir = self.cleaned_data['DDD_CAR'].median()
                    self.cleaned_data['DDD_CAR'].fillna(median_dir, inplace=True)
        
        # Handle missing values for all numeric columns
        for col in self.cleaned_data.columns:
            if col not in self.ignore_columns and self.cleaned_data[col].dtype in ['int64', 'float64']:
                missing = self.cleaned_data[col].isnull().sum()
                if missing > 0:
                    print(f"Filling {missing} missing values in {col}")
                    self.cleaned_data[col].fillna(self.cleaned_data[col].median(), inplace=True)
        
        print("Data cleaning completed!")
        return self.cleaned_data
    
    def transform_rr(self):
        """Transform RR using log1p"""
        if 'RR' in self.cleaned_data.columns:
            print("\n" + "="*50)
            print("RR TRANSFORMATION")
            print("="*50)
            
            original_rr = self.cleaned_data['RR'].copy()
            print(f"Original RR stats:")
            print(f"  Min: {original_rr.min():.4f}")
            print(f"  Max: {original_rr.max():.4f}")
            print(f"  Mean: {original_rr.mean():.4f}")
            print(f"  Zero values: {(original_rr == 0).sum()}")
            
            # Apply log1p transformation
            self.cleaned_data['RR'] = np.log1p(original_rr)
            
            print(f"Transformed RR (log1p) stats:")
            print(f"  Min: {self.cleaned_data['RR'].min():.4f}")
            print(f"  Max: {self.cleaned_data['RR'].max():.4f}")
            print(f"  Mean: {self.cleaned_data['RR'].mean():.4f}")
            print("RR transformation completed!")
        else:
            print("RR column not found - skipping transformation")
    
    def scale_data(self, scaling_method='standard'):
        """Scale data including transformed RR"""
        if self.cleaned_data is None:
            print("Data belum di-clean!")
            return
        
        # Transform RR before scaling
        self.transform_rr()
        
        print("\n" + "="*50)
        print(f"DATA SCALING - {scaling_method.upper()}")
        print("="*50)
        
        # Identify columns to scale
        cols_to_scale = []
        cols_to_keep = []
        
        for col in self.cleaned_data.columns:
            if col in self.ignore_columns:
                cols_to_keep.append(col)
                print(f"Keeping {col} unchanged (ignored column)")
            elif self.cleaned_data[col].dtype in ['int64', 'float64']:
                cols_to_scale.append(col)
            else:
                cols_to_keep.append(col)
                print(f"Keeping {col} unchanged (non-numeric)")
        
        print(f"Columns to scale: {cols_to_scale}")
        
        # Choose scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'minmax', or 'robust'")
        
        # Scale data
        if cols_to_scale:
            scaled_values = self.scaler.fit_transform(self.cleaned_data[cols_to_scale])
            scaled_df = pd.DataFrame(scaled_values, columns=cols_to_scale, index=self.cleaned_data.index)
            
            # Combine scaled and unscaled columns
            self.scaled_data = pd.DataFrame()
            
            # Maintain original column order
            for col in self.cleaned_data.columns:
                if col in cols_to_scale:
                    self.scaled_data[col] = scaled_df[col]
                else:
                    self.scaled_data[col] = self.cleaned_data[col]
        else:
            print("No numeric columns found for scaling!")
            self.scaled_data = self.cleaned_data.copy()
        
        print("\nScaling completed!")
        
        # Show scaling statistics
        if cols_to_scale:
            print("\nScaling Statistics:")
            for col in cols_to_scale:
                orig_mean = self.cleaned_data[col].mean()
                orig_std = self.cleaned_data[col].std()
                scaled_mean = self.scaled_data[col].mean()
                scaled_std = self.scaled_data[col].std()
                
                if col == 'RR':
                    print(f"{col} (log transformed): {orig_mean:.4f}±{orig_std:.4f} → {scaled_mean:.4f}±{scaled_std:.4f}")
                else:
                    print(f"{col}: {orig_mean:.2f}±{orig_std:.2f} → {scaled_mean:.4f}±{scaled_std:.4f}")
        
        return self.scaled_data
    
    def save_results(self, output_filename=None):
        """Save scaled data to CSV"""
        if self.scaled_data is None:
            print("No scaled data to save!")
            return
        
        # Generate output filename if not provided
        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_filename = f"{base_name}_scaled.csv"
            print(f"Generated output file: {output_filename}")
        
        try:
            print(f"\nSaving scaled data to '{output_filename}'...")
            self.scaled_data.to_csv(output_filename, index=False)
            print(f"✅ Saved successfully!")
            
            # Show preview
            print("\nPreview of scaled data:")
            print(self.scaled_data.head())
            
            return output_filename
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return None

def main():
    """Main function"""
    print("CSV Data Scaler with RR Transformation")
    print("="*40)
    
    # Daftar path file input (masukkan path di sini)
    file_paths = [
        "./pre-prosesing/input_data/test.csv",  # File pertama
        "./pre-prosesing/input_data/train.csv"     # File kedua
    ]
    
    # Proses setiap file satu per satu
    for i, file_path in enumerate(file_paths, 1):
        print("\n" + "="*60)
        print(f"🚀 PROCESSING FILE KE-{i}: {file_path}")
        print("="*60)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found! Skipping to next file.")
            continue  # Lanjut ke file berikutnya jika error
        
        # Initialize scaler untuk file ini
        scaler = CSVDataScaler(file_path)
        
        # Process data
        print("\n📊 Step 1: Loading data...")
        if scaler.load_data() is None:
            continue  # Skip jika load gagal
        
        print("\n🔍 Step 2: Exploring data...")
        scaler.explore_data()
        
        print("\n🧹 Step 3: Cleaning data...")
        scaler.clean_data()
        
        print("\n📊 Step 4: Scaling data (with RR transformation)...")
        scaler.scale_data(scaling_method='standard')
        
        print("\n💾 Step 5: Saving results...")
        # Tentukan folder output
        output_dir = "./pre-prosesing/output_data"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = os.path.join(output_dir, f"{base_name}_scaled.csv")
        output_file = scaler.save_results(output_filename) 

        if output_file:
            print("\n" + "="*60)
            print(f"✅ PROCESSING FILE KE-{i} COMPLETED!")
            print("="*60)
            print(f"Input: {file_path}")
            print(f"Output: {output_file}")
            print(f"Shape: {scaler.scaled_data.shape}")
            print(f"RR transformed with log1p and scaled")
    
        print("\n" + "="*60)
        print(f"⭐ FINISHED PROCESSING FILE KE-{i}")
        print("="*60)
    
    # Jika semua file selesai
    print("\n" + "="*60)
    print("✅ SCALING MULTIPLE FILES COMPLETED!")
    print("="*60)
    print(f"📈 Processed {len(file_paths)} files")
    print("Proses seluruhnya selesai!")

if __name__ == "__main__":
    main()