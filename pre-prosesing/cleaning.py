import pandas as pd
import os

def clean_csv_data(input_file, output_file=None, remove_values=None):
    """
    Clean CSV data by removing rows containing specific values
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    remove_values (list): List of values that trigger row removal (default: ['-', '8888', '9999'])
    """
    
    # Default values to remove
    if remove_values is None:
        remove_values = ['-', '8888', '9999']
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned.csv"
    
    try:
        # Read the CSV file
        print(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        
        # Convert all data to string for comparison
        df_str = df.astype(str)
        
        # Create boolean mask for rows to remove
        rows_to_remove = pd.Series([False] * len(df))
        
        for value in remove_values:
            # Check each column for the specific value
            for col in df_str.columns:
                rows_to_remove |= (df_str[col] == str(value))
        
        # Keep rows that don't contain any of the specified values
        df_cleaned = df[~rows_to_remove].copy()
        df_cleaned.reset_index(drop=True, inplace=True)
        
        print(f"Rows removed: {rows_to_remove.sum()}")
        print(f"Cleaned data shape: {df_cleaned.shape}")
        
        # Show which values were found
        if rows_to_remove.sum() > 0:
            print("\nRemoved rows contained:")
            removed_df = df[rows_to_remove]
            for value in remove_values:
                count = 0
                for col in df_str.columns:
                    count += (df_str[col] == str(value)).sum()
                if count > 0:
                    print(f"  - '{value}': {count} cells")
        
        # Save cleaned data
        print(f"\nSaving cleaned data to: {output_file}")
        df_cleaned.to_csv(output_file, index=False)
        
        print("Cleaning completed successfully!")
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except Exception as e:
        print(f"Error during cleaning: {str(e)}")
        return None

def batch_clean(input_folder, output_folder=None, remove_values=None):
    """
    Clean multiple CSV files in a folder
    
    Parameters:
    input_folder (str): Path to folder containing CSV files
    output_folder (str): Path to output folder (optional)
    remove_values (list): List of values that trigger row removal
    """
    
    if output_folder is None:
        output_folder = input_folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files to clean")
    
    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_filename = os.path.splitext(csv_file)[0] + "_cleaned.csv"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"\n--- Cleaning {csv_file} ---")
        clean_csv_data(input_path, output_path, remove_values)

def main():
    """Main function"""
    print("CSV Data Cleaner")
    print("=" * 20)
    
    input_file = "./pre-prosesing/data.csv"  # Change this to your input file
    values_to_remove = ['-', '8888', '9999']
    
    if os.path.exists(input_file):
        result = clean_csv_data(input_file, remove_values=values_to_remove)
        if result is not None:
            print(f"\nPreview of cleaned data:")
            print(result.head())
    else:
        print(f"Please make sure '{input_file}' exists")
        print("\nUsage examples:")
        print("clean_csv_data('your_file.csv')")
        print("clean_csv_data('data.csv', remove_values=['-', 'N/A'])")
        print("batch_clean('input_folder/')")

if __name__ == "__main__":
    main()