"""
OLS.py - Ordinary Least Squares Coefficient Estimation
Input: data_train.csv (X1-X6, RR)
Output: Coefficient estimates (β̂) menggunakan rumus OLS
"""

import pandas as pd
import numpy as np
from scipy import stats

class OLSEstimator:
    def __init__(self):
        self.beta_hat = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.std_errors = None
        self.t_stats = None
        self.p_values = None
        
    def fit(self, X, y, feature_names=None):
        """
        Estimate coefficients using OLS formula: β̂ = (X^T * X)^(-1) * X^T * y
        """
        self.feature_names = feature_names
        n, p = X.shape
        
        # Add intercept column (β₀)
        X_with_intercept = np.column_stack([np.ones(n), X])
        self.X_train = X_with_intercept
        self.y_train = y
        
        print("Calculating OLS estimates using matrix algebra...")
        print(f"X matrix shape: {X_with_intercept.shape}")
        print(f"y vector shape: {y.shape}")
        
        # Step 1: X^T (transpose)
        X_T = X_with_intercept.T
        print(f"X^T shape: {X_T.shape}")
        
        # Step 2: X^T * X
        XTX = X_T @ X_with_intercept
        print(f"X^T*X shape: {XTX.shape}")
        
        # Step 3: (X^T * X)^(-1)
        XTX_inv = np.linalg.inv(XTX)
        print(f"(X^T*X)^(-1) calculated")
        
        # Step 4: X^T * y
        XTy = X_T @ y
        print(f"X^T*y shape: {XTy.shape}")
        
        # Step 5: β̂ = (X^T * X)^(-1) * X^T * y
        self.beta_hat = XTX_inv @ XTy
        print(f"β̂ (beta hat) calculated: {self.beta_hat.shape}")
        
        # Calculate additional statistics
        self._calculate_statistics(X_with_intercept, y, XTX_inv)
        
        return self
    
    def _calculate_statistics(self, X, y, XTX_inv):
        """Calculate standard errors and significance tests"""
        n, p = X.shape
        
        # Residuals
        y_pred = X @ self.beta_hat
        residuals = y - y_pred
        
        # Sum of squared residuals
        SSR = np.sum(residuals**2)
        
        # Degrees of freedom
        df = n - p
        
        # Mean squared error (σ²)
        mse = SSR / df
        
        # Variance-covariance matrix
        var_covar_matrix = mse * XTX_inv
        
        # Standard errors
        self.std_errors = np.sqrt(np.diag(var_covar_matrix))
        
        # t-statistics
        self.t_stats = self.beta_hat / self.std_errors
        
        # p-values (two-tailed)
        self.p_values = 2 * (1 - stats.t.cdf(np.abs(self.t_stats), df))
        
        print(f"Statistical calculations completed (n={n}, p={p}, df={df})")
    
    def get_coefficients(self):
        """Return coefficient estimates"""
        if self.beta_hat is None:
            raise ValueError("Model belum di-fit!")
        return self.beta_hat
    
    def get_summary(self):
        """Get detailed coefficient summary"""
        if self.beta_hat is None:
            raise ValueError("Model belum di-fit!")
        
        # Feature names
        feature_names = ['Intercept']
        if self.feature_names:
            feature_names.extend(self.feature_names)
        else:
            feature_names.extend([f'X{i+1}' for i in range(len(self.beta_hat)-1)])
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient_Estimate': self.beta_hat,
            'Std_Error': self.std_errors,
            't_statistic': self.t_stats,
            'p_value': self.p_values,
            'Significant_5%': self.p_values < 0.05
        })
        
        return summary_df
    
    def save_coefficients(self, filename="ols_trainCoefficients.csv"):
        """Save coefficient estimates to CSV"""
        summary = self.get_summary()
        summary.to_csv(filename, index=False)
        print(f"✓ Coefficients saved to: {filename}")
        return filename

def load_training_data(file_path, target_column='RR'):
    """Load training data from CSV"""
    print(f"Loading training data from: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found!")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Remove missing values
        clean_data = data.dropna()
        if len(clean_data) < len(data):
            print(f"Removed {len(data) - len(clean_data)} rows with missing values")
        
        X_clean = clean_data.drop(columns=[target_column]).values
        y_clean = clean_data[target_column].values
        feature_names = list(data.drop(columns=[target_column]).columns)
        
        print(f"Final training data: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        print(f"Features: {feature_names}")
        
        return X_clean, y_clean, feature_names
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def display_ols_theory():
    """Display OLS mathematical theory"""
    print("="*60)
    print("ORDINARY LEAST SQUARES (OLS) THEORY")
    print("="*60)
    print("Goal: Find coefficient estimates β̂ that minimize sum of squared residuals")
    print()
    print("Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε")
    print("Matrix form: y = Xβ + ε")
    print()
    print("OLS Estimator Formula:")
    print("β̂ = (X^T * X)^(-1) * X^T * y")
    print()
    print("Where:")
    print("- β̂ = vector of coefficient estimates")
    print("- X = design matrix (includes intercept column)")
    print("- y = response vector")
    print("- X^T = transpose of X")
    print("="*60)

def main():
    """Main OLS estimation function"""
    print("="*60)
    print("OLS COEFFICIENT ESTIMATION")
    print("="*60)
    
    # Configuration
    train_file = "./pre-prosesing/output_data/train_polynomial_scaled.csv"  # Input training data
    target_col = "RR"              # Dependent variable
    output_file = "./prosesing/train_olsCoefficients.csv"  # Output coefficients

    # Display theory
    display_ols_theory()
    
    # Load training data
    print("\n📁 LOADING TRAINING DATA")
    print("-" * 30)
    X_train, y_train, feature_names = load_training_data(train_file, target_col)
    
    if X_train is None:
        print(f"❌ Failed to load {train_file}")
        return None
    
    # Fit OLS model
    print("\n🧮 ESTIMATING COEFFICIENTS")
    print("-" * 30)
    
    ols_model = OLSEstimator()
    ols_model.fit(X_train, y_train, feature_names)
    
    # Get results
    coefficients = ols_model.get_coefficients()
    summary = ols_model.get_summary()
    
    # Display results
    print("\n📊 COEFFICIENT ESTIMATES (β̂)")
    print("-" * 30)
    print(summary.round(6))
    
    # Model equation
    print("\n📐 ESTIMATED MODEL EQUATION")
    print("-" * 30)
    equation = f"ŷ = {coefficients[0]:.6f}"
    for i, feature in enumerate(feature_names):
        if coefficients[i+1] >= 0:
            equation += f" + {coefficients[i+1]:.6f}×{feature}"
        else:
            equation += f" - {abs(coefficients[i+1]):.6f}×{feature}"
    print(equation)
    
    # Significance summary
    significant_features = summary[summary['Significant_5%'] == True]
    print(f"\n📈 SIGNIFICANT FEATURES (α = 0.05): {len(significant_features)}/{len(feature_names)+1}")
    print("-" * 30)
    for _, row in significant_features.iterrows():
        print(f"✓ {row['Feature']}: β̂ = {row['Coefficient_Estimate']:.6f}, p = {row['p_value']:.6f}")
    
    # Save coefficients
    print(f"\n💾 SAVING RESULTS")
    print("-" * 30)
    output_path = ols_model.save_coefficients(output_file)
    
    print("\n" + "="*60)
    print("✅ OLS COEFFICIENT ESTIMATION COMPLETED!")
    print("="*60)
    print(f"📈 Estimated {len(coefficients)} coefficients (including intercept)")
    print(f"📄 Results saved to: {output_path}")
    print(f"🔄 Use coefficients in MLR for predictions and testing")
    
    return ols_model, coefficients

if __name__ == "__main__":
    model, coefficients = main()