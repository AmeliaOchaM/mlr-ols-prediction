import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedCSVCleaner:
    def __init__(self, input_file, output_file=None):
        self.input_file = input_file
        self.output_file = output_file or f"{os.path.splitext(input_file)[0]}_advanced_cleaned.csv"
        self.df = None
        self.original_shape = None
        self.missing_patterns = {}
        
    def load_data(self):
        """Load CSV and identify data types"""
        print(f"📂 Loading data from: {self.input_file}")
        # self.df = pd.read_csv(self.input_file)
        self.df = pd.read_csv(
            self.input_file,
            na_values=['-', '--', '---', 'nan', 'NaN', 'null', 'NULL',
                    '8888', '9999', '-999', '-99', '99.9', '999.9']
        )
        self.original_shape = self.df.shape
        print(f"✅ Original data shape: {self.original_shape}")
        return self
    
    def identify_missing_patterns(self):
        """Identify different patterns of missing values"""
        print("\n🔍 IDENTIFYING MISSING VALUE PATTERNS")
        print("-" * 45)
        
        # Define missing value patterns for rainfall data
        missing_patterns = {
            'numeric_missing': ['-', '--', '---', 'nan', 'NaN', 'null', 'NULL'],
            'rainfall_missing': ['8888', '9999', '-999', '-99', '99.9', '999.9'],
            'wind_missing': ['C', 'CALM', 'VAR', '0'],  # For wind direction
            'extreme_outliers': []  # Will be detected statistically
        }
        
        for col in self.df.columns:
            col_missing = {}
            total_missing = 0
            
            print(f"\n📊 Column: {col}")
            
            for pattern_name, patterns in missing_patterns.items():
                if pattern_name == 'extreme_outliers':
                    continue
                    
                pattern_count = 0
                for pattern in patterns:
                    if col == 'DDD_CAR' and pattern_name == 'wind_missing':
                        # Special handling for wind direction
                        continue
                    else:
                        count = (self.df[col].astype(str) == str(pattern)).sum()
                        pattern_count += count
                
                if pattern_count > 0:
                    col_missing[pattern_name] = pattern_count
                    total_missing += pattern_count
                    print(f"  {pattern_name}: {pattern_count} values")
            
            # Check for standard NaN
            nan_count = self.df[col].isna().sum()
            if nan_count > 0:
                col_missing['standard_nan'] = nan_count
                total_missing += nan_count
                print(f"  standard_nan: {nan_count} values")
            
            if total_missing > 0:
                print(f"  📈 Total missing: {total_missing}/{len(self.df)} ({total_missing/len(self.df)*100:.1f}%)")
                self.missing_patterns[col] = col_missing
            else:
                print(f"  ✅ No missing values found")
    
    def clean_missing_values(self):
        """Advanced missing value cleaning with different strategies per column"""
        print("\n🧹 CLEANING MISSING VALUES")
        print("-" * 30)
        
        # Convert all missing patterns to NaN first
        missing_indicators = ['-', '--', '---', 'nan', 'NaN', 'null', 'NULL', 
                            '8888', '9999', '-999', '-99', '99.9', '999.9']
        
        # Clean each column based on its type and characteristics
        for col in self.df.columns:
            print(f"\n🔧 Processing {col}...")
            original_missing = self.df[col].isna().sum()
            
            # Replace missing indicators with NaN
            for indicator in missing_indicators:
                self.df[col] = self.df[col].replace(indicator, np.nan)
            
            # Convert to numeric if possible (except DDD_CAR)
            if col != 'DDD_CAR':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            new_missing = self.df[col].isna().sum()
            converted_missing = new_missing - original_missing
            
            if converted_missing > 0:
                print(f"  📝 Converted {converted_missing} indicators to NaN")
            
            # Apply different imputation strategies
            if col == 'RR':  # Rainfall - most important
                self.df[col] = self._impute_rainfall(self.df[col])
            elif col in ['TAVG', 'FF_AVG', 'RH_AVG', 'SS']:  # Continuous meteorological
                self.df[col] = self._impute_meteorological(self.df[col], col)
            elif col == 'DDD_CAR':  # Categorical wind direction
                self.df[col] = self._impute_wind_direction(self.df[col])
            elif col == 'DDD_X':  # Wind direction in degrees
                self.df[col] = self._impute_wind_degrees(self.df[col])
            
            final_missing = self.df[col].isna().sum()
            print(f"  ✅ Final missing: {final_missing}")
    
    def _impute_rainfall(self, series):
        """Smart rainfall imputation using multiple methods"""
        missing_count = series.isna().sum()
        if missing_count == 0:
            return series
        
        print(f"    🌧️  Rainfall imputation: {missing_count} missing values")
        
        # Strategy 1: Use weather context (high humidity + low sun = likely rain)
        if 'RH_AVG' in self.df.columns and 'SS' in self.df.columns:
            # High humidity (>85%) and low sunshine (<3) suggests rain conditions
            rain_conditions = (self.df['RH_AVG'] > 85) & (self.df['SS'] < 3)
            dry_conditions = (self.df['RH_AVG'] < 70) & (self.df['SS'] > 6)
            
            # Fill based on weather conditions
            series_filled = series.copy()
            
            # Dry conditions: likely 0 or very low rain
            dry_mask = series_filled.isna() & dry_conditions
            series_filled.loc[dry_mask] = np.random.exponential(scale=0.5, size=dry_mask.sum())
            
            # Rainy conditions: use existing rain values as reference
            rain_mask = series_filled.isna() & rain_conditions
            if rain_mask.any():
                rain_values = series_filled[series_filled > 0].dropna()
                if len(rain_values) > 0:
                    # Sample from existing rain distribution
                    replacements = np.random.choice(rain_values, size=rain_mask.sum(), replace=True)
                    series_filled.loc[rain_mask] = replacements
                else:
                    # Use moderate rain values
                    series_filled.loc[rain_mask] = np.random.gamma(2, 3, size=rain_mask.sum())
            
            # Remaining missing: use median of similar conditions
            remaining_missing = series_filled.isna()
            if remaining_missing.any():
                median_rain = series_filled.median()
                series_filled.loc[remaining_missing] = median_rain
                
            return series_filled
        
        # Fallback: Use statistical imputation
        return self._statistical_impute(series, method='median')
    
    def _impute_meteorological(self, series, col_name):
        """Impute meteorological variables using KNN"""
        missing_count = series.isna().sum()
        if missing_count == 0:
            return series
        
        print(f"    🌡️  {col_name} imputation: {missing_count} missing values")
        
        # Use KNN imputation with other meteorological variables
        met_cols = ['TAVG', 'FF_AVG', 'RH_AVG', 'SS', 'DDD_X']
        available_cols = [col for col in met_cols if col in self.df.columns]
        
        if len(available_cols) >= 2:
            # Create temporary dataframe with only meteorological data
            temp_df = self.df[available_cols].copy()
            
            # KNN imputation
            imputer = KNNImputer(n_neighbors=min(3, len(temp_df)-1), weights='distance')
            temp_imputed = imputer.fit_transform(temp_df)
            
            # Get the column index
            col_idx = available_cols.index(col_name)
            return pd.Series(temp_imputed[:, col_idx], index=series.index)
        
        # Fallback to statistical method
        return self._statistical_impute(series, method='mean')
    
    def _impute_wind_direction(self, series):
        """Impute categorical wind direction"""
        missing_count = series.isna().sum()
        if missing_count == 0:
            return series
        
        print(f"    💨 Wind direction imputation: {missing_count} missing values")
        
        # Use mode (most frequent direction)
        mode_direction = series.mode()
        if len(mode_direction) > 0:
            return series.fillna(mode_direction[0])
        else:
            return series.fillna('C')  # Calm as default
    
    def _impute_wind_degrees(self, series):
        """Impute numerical wind direction degrees"""
        missing_count = series.isna().sum()
        if missing_count == 0:
            return series
        
        print(f"    🧭 Wind degrees imputation: {missing_count} missing values")
        
        # For wind direction, use circular mean
        # Convert to radians, compute mean, convert back
        valid_values = series.dropna()
        if len(valid_values) > 0:
            # Circular statistics for wind direction
            rad_values = np.radians(valid_values)
            sin_mean = np.mean(np.sin(rad_values))
            cos_mean = np.mean(np.cos(rad_values))
            circular_mean = np.degrees(np.arctan2(sin_mean, cos_mean))
            if circular_mean < 0:
                circular_mean += 360
            
            return series.fillna(circular_mean)
        
        return series.fillna(0)  # Default to North
    
    def _statistical_impute(self, series, method='median'):
        """Statistical imputation methods"""
        if method == 'median':
            return series.fillna(series.median())
        elif method == 'mean':
            return series.fillna(series.mean())
        elif method == 'mode':
            mode_val = series.mode()
            return series.fillna(mode_val[0] if len(mode_val) > 0 else 0)
        else:
            return series.fillna(0)
    
    def detect_and_handle_outliers(self, method='iqr', factor=2.5):
        """Detect and handle outliers in rainfall data"""
        print(f"\n🎯 OUTLIER DETECTION ({method.upper()})")
        print("-" * 25)
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col == 'DDD_X':  # Skip wind direction degrees (0-360 is valid range)
                continue
                
            original_count = len(self.df)
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    print(f"📊 {col}: {outlier_count} outliers detected")
                    print(f"    Valid range: {lower_bound:.2f} to {upper_bound:.2f}")
                    
                    # For rainfall, cap extreme values instead of removing
                    if col == 'RR':
                        # Cap at reasonable maximum (e.g., 200mm for daily rainfall)
                        reasonable_max = min(200, upper_bound)
                        extreme_mask = self.df[col] > reasonable_max
                        if extreme_mask.any():
                            print(f"    🧢 Capping {extreme_mask.sum()} extreme values at {reasonable_max}")
                            self.df.loc[extreme_mask, col] = reasonable_max
                    else:
                        # For other variables, use winsorization (replace with boundary values)
                        self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                        self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                        print(f"    🔧 Winsorized {outlier_count} values")
                else:
                    print(f"✅ {col}: No outliers detected")
    
    def validate_data_quality(self):
        """Final validation of cleaned data"""
        print(f"\n✅ DATA QUALITY VALIDATION")
        print("-" * 25)
        
        # Check for remaining missing values
        missing_summary = self.df.isnull().sum()
        total_missing = missing_summary.sum()
        
        if total_missing > 0:
            print(f"⚠️  Warning: {total_missing} missing values remain:")
            for col, count in missing_summary[missing_summary > 0].items():
                print(f"    {col}: {count}")
        else:
            print("✅ No missing values remaining")
        
        # Check data ranges
        print("\n📊 Data ranges after cleaning:")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            print(f"    {col}: {self.df[col].min():.2f} to {self.df[col].max():.2f}")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n🔄 Found {duplicates} duplicate rows")
            self.df = self.df.drop_duplicates()
            print(f"    Removed duplicates, new shape: {self.df.shape}")
        
        print(f"\n📈 Final data shape: {self.df.shape}")
        print(f"📉 Data retention: {len(self.df)/self.original_shape[0]*100:.1f}%")
    
    def save_cleaned_data(self):
        """Save the cleaned dataset"""
        self.df.to_csv(self.output_file, index=False)
        print(f"\n💾 Cleaned data saved to: {self.output_file}")
        
        # Save cleaning report
        report_file = self.output_file.replace('.csv', '_cleaning_report.txt')
        with open(report_file, 'w') as f:
            f.write("ADVANCED CLEANING REPORT\n")
            f.write("========================\n\n")
            f.write(f"Original shape: {self.original_shape}\n")
            f.write(f"Final shape: {self.df.shape}\n")
            f.write(f"Data retention: {len(self.df)/self.original_shape[0]*100:.1f}%\n\n")
            f.write("Missing value patterns identified:\n")
            for col, patterns in self.missing_patterns.items():
                f.write(f"  {col}: {patterns}\n")
        
        print(f"📋 Cleaning report saved to: {report_file}")
        
        return self.df
    
    def display_before_after(self):
        """Display before/after comparison"""
        print(f"\n📊 BEFORE vs AFTER COMPARISON")
        print("-" * 30)
        
        # Load original data for comparison
        original_df = pd.read_csv(self.input_file)
        
        print("BEFORE CLEANING:")
        print(f"  Shape: {original_df.shape}")
        print(f"  Missing values: {original_df.isnull().sum().sum()}")
        print(f"  Missing indicators found: {sum([(original_df == indicator).sum().sum() for indicator in ['-', '8888', '9999']])}")
        
        print("\nAFTER CLEANING:")
        print(f"  Shape: {self.df.shape}")
        print(f"  Missing values: {self.df.isnull().sum().sum()}")
        print(f"  Data retention: {len(self.df)/len(original_df)*100:.1f}%")
        
        print(f"\n📋 Sample of cleaned data:")
        print(self.df.head())


def main():
    """Main function to run advanced cleaning"""
    print("🚀 ADVANCED CSV CLEANING FOR RAINFALL DATA")
    print("=" * 45)
    
    # Configuration
    input_file = "data.csv"  # Change to your file path
    output_file = "data_advanced_cleaned.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please ensure the file exists in the current directory.")
        return
    
    # Initialize cleaner
    cleaner = AdvancedCSVCleaner(input_file, output_file)
    
    # Run cleaning pipeline
    try:
        # Step 1: Load and analyze
        cleaner.load_data()
        cleaner.identify_missing_patterns()
        
        # Step 2: Clean missing values
        cleaner.clean_missing_values()
        
        # Step 3: Handle outliers
        cleaner.detect_and_handle_outliers(method='iqr', factor=2.5)
        
        # Step 4: Validate and save
        cleaner.validate_data_quality()
        cleaned_df = cleaner.save_cleaned_data()
        
        # Step 5: Show comparison
        cleaner.display_before_after()
        
        print("\n🎉 ADVANCED CLEANING COMPLETED SUCCESSFULLY!")
        print("=" * 45)
        print(f"📁 Input: {input_file}")
        print(f"📁 Output: {output_file}")
        print(f"📈 Ready for scaling and modeling!")
        
        return cleaned_df
        
    except Exception as e:
        print(f"❌ Error during cleaning: {str(e)}")
        return None

if __name__ == "__main__":
    cleaned_data = main()