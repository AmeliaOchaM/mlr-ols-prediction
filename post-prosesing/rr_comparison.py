"""
RR Comparison Script
Compares predicted RR vs actual RR from test data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_test_data():
    """Load test data with actual RR values"""
    try:
        test_data = pd.read_csv("./pre-prosesing/output_data/test_scaled.csv")
        print(f"Test data loaded: {test_data.shape}")
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def load_predicted_data():
    """Load MLR prediction results"""
    try:
        pred_data = pd.read_csv("./prosesing/test_mlr.csv")
        print(f"Prediction data loaded: {pred_data.shape}")
        return pred_data
    except Exception as e:
        print(f"Error loading prediction data: {e}")
        return None

def compare_rr_values(test_data, pred_data):
    """Compare actual vs predicted RR values row by row"""
    
    # Extract RR values
    actual_rr = test_data['RR'].values
    predicted_rr = pred_data['Predicted'].values if 'Predicted' in pred_data.columns else pred_data.iloc[:, -1].values
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Row_Index': range(len(actual_rr)),
        'Actual_RR': actual_rr,
        'Predicted_RR': predicted_rr,
        'Difference': actual_rr - predicted_rr,
        'Abs_Difference': np.abs(actual_rr - predicted_rr),
        'Percent_Error': np.abs(actual_rr - predicted_rr) / np.maximum(actual_rr, 0.1) * 100
    })
    
    return comparison_df

def calculate_metrics(actual, predicted):
    """Calculate error metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 0.1))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def create_comparison_plot(comparison_df):
    """Create visualization comparing actual vs predicted"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Row-by-row comparison
    axes[0, 0].plot(comparison_df['Row_Index'], comparison_df['Actual_RR'], 
                   label='Actual RR', color='blue', alpha=0.7)
    axes[0, 0].plot(comparison_df['Row_Index'], comparison_df['Predicted_RR'], 
                   label='Predicted RR', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Row Index')
    axes[0, 0].set_ylabel('RR Value')
    axes[0, 0].set_title('Actual vs Predicted RR by Row')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot
    axes[0, 1].scatter(comparison_df['Actual_RR'], comparison_df['Predicted_RR'], 
                      alpha=0.6, color='green')
    min_val = min(comparison_df['Actual_RR'].min(), comparison_df['Predicted_RR'].min())
    max_val = max(comparison_df['Actual_RR'].max(), comparison_df['Predicted_RR'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual RR')
    axes[0, 1].set_ylabel('Predicted RR')
    axes[0, 1].set_title('Actual vs Predicted Scatter Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error distribution
    axes[1, 0].hist(comparison_df['Difference'], bins=20, alpha=0.7, color='orange')
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error (Actual - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Prediction Errors')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Absolute error by row
    axes[1, 1].bar(comparison_df['Row_Index'], comparison_df['Abs_Difference'], 
                  alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Row Index')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Absolute Error by Row')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rr_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def display_worst_predictions(comparison_df, n=10):
    """Display rows with worst predictions"""
    worst_errors = comparison_df.nlargest(n, 'Abs_Difference')
    print(f"\n{n} Worst Predictions:")
    print("-" * 80)
    for _, row in worst_errors.iterrows():
        print(f"Row {row['Row_Index']:3d}: Actual={row['Actual_RR']:6.1f}, "
              f"Predicted={row['Predicted_RR']:6.1f}, Error={row['Difference']:6.1f}")

def main():
    """Main comparison function"""
    print("="*60)
    print("RR PREDICTION COMPARISON")
    print("="*60)
    
    # Load data
    print("Loading data...")
    test_data = load_test_data()
    pred_data = load_predicted_data()
    
    if test_data is None or pred_data is None:
        print("Failed to load required data files")
        return
    
    # Check data alignment
    if len(test_data) != len(pred_data):
        print(f"Warning: Data length mismatch! Test: {len(test_data)}, Pred: {len(pred_data)}")
        min_len = min(len(test_data), len(pred_data))
        test_data = test_data.iloc[:min_len]
        pred_data = pred_data.iloc[:min_len]
        print(f"Using first {min_len} rows for comparison")
    
    # Compare values
    print("\nComparing RR values...")
    comparison_df = compare_rr_values(test_data, pred_data)
    
    # Calculate metrics
    metrics = calculate_metrics(comparison_df['Actual_RR'], comparison_df['Predicted_RR'])
    
    # Display results
    print(f"\nComparison Summary ({len(comparison_df)} rows):")
    print("-" * 40)
    print(f"Mean Actual RR:    {comparison_df['Actual_RR'].mean():.2f}")
    print(f"Mean Predicted RR: {comparison_df['Predicted_RR'].mean():.2f}")
    print(f"Mean Error:        {comparison_df['Difference'].mean():.2f}")
    print(f"Mean Abs Error:    {comparison_df['Abs_Difference'].mean():.2f}")
    print(f"Max Abs Error:     {comparison_df['Abs_Difference'].max():.2f}")
    
    print(f"\nError Metrics:")
    print("-" * 20)
    for metric, value in metrics.items():
        print(f"{metric:5s}: {value:8.4f}")
    
    # Show worst predictions
    display_worst_predictions(comparison_df)
    
    # Save detailed comparison
    output_file = "rr_comparison_detailed.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nDetailed comparison saved to: {output_file}")
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plot(comparison_df)
    
    # Summary statistics by ranges
    print(f"\nPrediction Accuracy by RR Range:")
    print("-" * 40)
    ranges = [(0, 1), (1, 10), (10, 50), (50, float('inf'))]
    for low, high in ranges:
        mask = (comparison_df['Actual_RR'] >= low) & (comparison_df['Actual_RR'] < high)
        subset = comparison_df[mask]
        if len(subset) > 0:
            mae_range = subset['Abs_Difference'].mean()
            count = len(subset)
            print(f"RR {low:3.0f}-{high if high != float('inf') else '∞':>3s}: "
                  f"{count:3d} samples, MAE = {mae_range:.2f}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()