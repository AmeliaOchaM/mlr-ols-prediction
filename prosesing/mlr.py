"""
MLR.py - Multiple Linear Regression Prediction and Error Analysis
Input: 1) OLS coefficients from ols_coefficients.csv
       2) Test data from data_test.csv (X1-X6, RR)
Output: Predictions and error metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class MLRPredictor:
    def __init__(self):
        self.coefficients = None
        self.feature_names = None
        
    def load_coefficients(self, coef_file):
        """Load OLS coefficient estimates"""
        try:
            coef_data = pd.read_csv(coef_file)
            print(f"Loading coefficients from: {coef_file}")
            print(f"Coefficients loaded: {len(coef_data)} parameters")
            
            self.coefficients = coef_data['Coefficient_Estimate'].values
            self.feature_names = coef_data['Feature'].tolist()
            
            # Display loaded coefficients
            print("\nLoaded Coefficient Estimates:")
            for feature, coef in zip(self.feature_names, self.coefficients):
                print(f"  {feature}: {coef:.6f}")
                
            return True
            
        except Exception as e:
            print(f"Error loading coefficients: {e}")
            return False
    
    def predict(self, X):
        """Make predictions using MLR equation: ŷ = β₀ + β₁x₁ + ... + βₖxₖ"""
        if self.coefficients is None:
            raise ValueError("Coefficients not loaded!")
        
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # MLR prediction
        predictions = X_with_intercept @ self.coefficients
        
        return predictions
    
    def calculate_errors(self, y_true, y_pred):
        """Calculate comprehensive error metrics"""
        # Basic errors
        residuals = y_true - y_pred
        
        # Error metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        # Improved MAPE calculation handling zero values
        if np.any(y_true == 0):
            non_zero_mask = y_true != 0
            zero_count = np.sum(y_true == 0)
            print(f"Warning: {zero_count} zero values found in target variable")
            
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs(residuals[non_zero_mask] / y_true[non_zero_mask])) * 100
            else:
                mape = float('nan')
        else:
            mape = np.mean(np.abs(residuals / y_true)) * 100

        bias = np.mean(residuals)  # Bias
        
        error_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Bias': bias,
            'Residuals': residuals
        }
        
        return error_metrics

def load_test_data(file_path, target_column='RR'):
    """Load test data from CSV"""
    try:
        print(f"Loading test data from: {file_path}")
        data = pd.read_csv(file_path)
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found!")
        
        # Clean data
        clean_data = data.dropna()
        if len(clean_data) < len(data):
            print(f"Removed {len(data) - len(clean_data)} rows with missing values")
        
        X_test = clean_data.drop(columns=[target_column]).values
        y_test = clean_data[target_column].values
        feature_names = list(clean_data.drop(columns=[target_column]).columns)
        
        print(f"Test data shape: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        return X_test, y_test, feature_names
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None, None

def display_error_analysis(error_metrics):
    """Display comprehensive error analysis"""
    print("\n📊 ERROR ANALYSIS RESULTS")
    print("-" * 30)
    
    # Main metrics
    for metric in ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Bias']:
        value = error_metrics[metric]
        print(f"{metric:6s}: {value:10.4f}")
    
    # Residual statistics
    residuals = error_metrics['Residuals']
    print(f"\nResidual Statistics:")
    print(f"  Mean:     {np.mean(residuals):8.4f}")
    print(f"  Std:      {np.std(residuals):8.4f}")
    print(f"  Min:      {np.min(residuals):8.4f}")
    print(f"  Max:      {np.max(residuals):8.4f}")
    print(f"  Range:    {np.max(residuals) - np.min(residuals):8.4f}")

def create_error_visualizations(y_test, y_pred, error_metrics, save_path="./post-prosesing/mlr_error_analysis.png"):
    """Create error analysis visualizations"""
    residuals = error_metrics['Residuals']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MLR Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add R² annotation
    r2 = error_metrics['R²']
    axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Residual plot
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Time series of errors
    axes[1, 1].plot(y_test, label='Actual', alpha=0.8, linewidth=2)
    axes[1, 1].plot(y_pred, label='Predicted', alpha=0.8, linewidth=2)
    axes[1, 1].fill_between(range(len(residuals)), y_test, y_pred, 
                           alpha=0.3, color='red', label='Error')
    axes[1, 1].set_xlabel('Data Points')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Prediction vs Actual with Errors')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(y_test, y_pred, error_metrics, output_file="mlr_results.csv"):
    """Save MLR results to CSV"""
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': error_metrics['Residuals'],
        'Abs_Error': np.abs(error_metrics['Residuals']),
        'Squared_Error': error_metrics['Residuals']**2
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    # Save error summary
    summary_file = output_file.replace('.csv', '_summary.csv')
    summary_df = pd.DataFrame([{
        'MSE': error_metrics['MSE'],
        'RMSE': error_metrics['RMSE'], 
        'MAE': error_metrics['MAE'],
        'R²': error_metrics['R²'],
        'MAPE': error_metrics['MAPE'],
        'Bias': error_metrics['Bias']
    }])
    
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Error summary saved to: {summary_file}")

def main():
    """Main MLR prediction and error analysis function"""
    print("="*60)
    print("MLR PREDICTION & ERROR ANALYSIS")
    print("="*60)
    
    # Configuration
    coef_file = "./prosesing/train_olsCoefficients.csv"    # OLS coefficient estimates
    test_file = "./pre-prosesing/output_data/test_scaled.csv"           # Test data
    target_col = "RR"                     # Dependent variable
    results_file = "test_mlr.csv"     # Output results
    
    # Initialize MLR predictor
    mlr = MLRPredictor()
    
    # Load OLS coefficients
    print("📊 LOADING OLS COEFFICIENTS")
    print("-" * 30)
    if not mlr.load_coefficients(coef_file):
        print(f"❌ Failed to load coefficients from {coef_file}")
        return None
    
    # Load test data
    print(f"\n📁 LOADING TEST DATA")
    print("-" * 30)
    X_test, y_test, test_features = load_test_data(test_file, target_col)
    
    if X_test is None:
        print(f"❌ Failed to load test data from {test_file}")
        return None
    
    # Verify feature consistency
    expected_features = mlr.feature_names[1:]  # Exclude intercept
    if test_features != expected_features:
        print(f"⚠️ Warning: Feature mismatch!")
        print(f"  Expected: {expected_features}")
        print(f"  Found:    {test_features}")
    
    # Make predictions
    print(f"\n🔮 MAKING PREDICTIONS")
    print("-" * 30)
    y_pred = mlr.predict(X_test)
    print(f"Predictions made for {len(y_pred)} samples")
    
    # Calculate errors
    print(f"\n📈 CALCULATING ERRORS")
    print("-" * 30)
    error_metrics = mlr.calculate_errors(y_test, y_pred)
    
    # Display results
    display_error_analysis(error_metrics)
    
    # Show equation used
    print(f"\n📐 MLR EQUATION USED")
    print("-" * 30)
    equation = f"ŷ = {mlr.coefficients[0]:.4f}"
    for i, feature in enumerate(expected_features):
        coef = mlr.coefficients[i+1]
        if coef >= 0:
            equation += f" + {coef:.4f}×{feature}"
        else:
            equation += f" - {abs(coef):.4f}×{feature}"
    print(equation)
    
    # Create visualizations
    print(f"\n📊 GENERATING VISUALIZATIONS")
    print("-" * 30)
    create_error_visualizations(y_test, y_pred, error_metrics)
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    print("-" * 30)
    save_results(y_test, y_pred, error_metrics, results_file)
    
    print("\n" + "="*60)
    print("✅ MLR ANALYSIS COMPLETED!")
    print("="*60)
    print(f"📊 Processed {len(y_test)} test samples")
    print(f"🎯 R² Score: {error_metrics['R²']:.4f}")
    print(f"📉 RMSE: {error_metrics['RMSE']:.4f}")
    print(f"📄 Results saved to: {results_file}")
    
    return mlr, error_metrics

if __name__ == "__main__":
    mlr_model, metrics = main()