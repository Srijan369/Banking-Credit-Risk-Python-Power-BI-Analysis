# check_dataset.py
import pandas as pd
import os

# Update this path to your actual file
dataset_path = 'data\Loan_Default.csv'  # CHANGE THIS TO YOUR FILE NAME

# Check if file exists
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    print("=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"\nFile: {dataset_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumn Names:")
    print(list(df.columns))
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
else:
    print(f"File not found: {dataset_path}")
    print("\nPlease check:")
    print("1. The file name is correct")
    print("2. The file is in the 'data' folder")
    print("\nAvailable files in data folder:")
    if os.path.exists('data'):
        print(os.listdir('data'))