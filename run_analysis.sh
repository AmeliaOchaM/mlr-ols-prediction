#!/bin/bash

# Script untuk menjalankan analisis regresi lengkap
# Menjalankan: cleaning -> splitting -> scaling -> ols -> mlr

echo "=========================================="
echo "Memulai Analisis Regresi Lengkap"
echo "=========================================="

# Aktivasi virtual environment
echo "Mengaktifkan virtual environment..."
source ./regresi_env/bin/activate

# 1. Data Cleaning
echo "1. Menjalankan data cleaning..."
python ./pre-prosesing/cleaning.py
if [ $? -ne 0 ]; then
    echo "Error: Data cleaning gagal!"
    exit 1
fi
echo "✓ Data cleaning selesai"

# 2. Data Splitting
echo "2. Menjalankan data splitting..."
python ./pre-prosesing/spliting.py
if [ $? -ne 0 ]; then
    echo "Error: Data splitting gagal!"
    exit 1
fi
echo "✓ Data splitting selesai"

# 3. Polynomial Features
echo "3. Menjalankan polynomial features..."
python ./pre-prosesing/polynomial_features.py
if [ $? -ne 0 ]; then
    echo "Error: Polynomial features gagal!"
    exit 1
fi
echo "✓ Polynomial features selesai"

# 4. Data Scaling
echo "4. Menjalankan data scaling..."
python ./pre-prosesing/scaleing.py
if [ $? -ne 0 ]; then
    echo "Error: Data scaling gagal!"
    exit 1
fi
echo "✓ Data scaling selesai"

# 5. OLS Analysis
echo "5. Menjalankan analisis OLS..."
python ./prosesing/ols.py
if [ $? -ne 0 ]; then
    echo "Error: Analisis OLS gagal!"
    exit 1
fi
echo "✓ Analisis OLS selesai"

# 6. MLR Analysis
echo "6. Menjalankan analisis MLR..."
python ./prosesing/mlr.py
if [ $? -ne 0 ]; then
    echo "Error: Analisis MLR gagal!"
    exit 1
fi
echo "✓ Analisis MLR selesai"

echo "=========================================="
echo "Semua tahapan analisis selesai!"
echo "=========================================="
echo "File output yang dihasilkan:"
echo "- ./pre-prosesing/data_cleaned.csv"
echo "- ./pre-prosesing/input_data/train.csv"
echo "- ./pre-prosesing/input_data/test.csv"
echo "- ./pre-prosesing/polynomial_data/train_polynomial.csv"
echo "- ./pre-prosesing/polynomial_data/test_polynomial.csv"
echo "- ./pre-prosesing/output_data/train_scaled.csv"
echo "- ./pre-prosesing/output_data/test_scaled.csv"
echo "- ./prosesing/train_olsCoefficients.csv"
echo "- ./prosesing/mlr_error_analysis.png"
echo "=========================================="
