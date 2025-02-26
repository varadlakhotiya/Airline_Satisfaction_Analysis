import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
file_path = "Cleaned_Data.csv"  # Change this to the correct file path
df = pd.read_csv(file_path)

# === 1. Basic Dataset Information ===
print("\nDataset Overview:")
print(df.info())

# === 2. Checking for Missing Values ===
missing_values = df.isnull().sum()
print("\nMissing Values per Column:\n", missing_values)

# === 3. Checking for Duplicates ===
duplicate_rows = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicate_rows)

# === 4. Checking Unique Values per Column (For Encoding Issues) ===
print("\nUnique Values per Column:\n", df.nunique())

# === 5. Detecting Outliers ===
## Select only numeric columns (excluding boolean and categorical)
numeric_cols = df.select_dtypes(include=['int64', 'float64'])  

## (A) Boxplot for Visual Inspection
plt.figure(figsize=(15, 8))
numeric_cols.boxplot(rot=90)
plt.title("Boxplot of Numerical Columns")
plt.show()

## (B) Z-Score Method
z_scores = numeric_cols.apply(zscore)  # Apply Z-score to numeric columns
outliers_zscore = (z_scores.abs() > 3).sum()
print("\nOutliers Detected Using Z-Score per Column:\n", outliers_zscore)

## (C) IQR Method
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

outliers_iqr = ((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).sum()
print("\nOutliers Detected Using IQR per Column:\n", outliers_iqr)

# === 6. Checking for Incorrect Encodings in Categorical Columns ===
print("\nFirst 10 Unique Values in Each Column:")
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # Checking numeric columns
        print(f"\nUnique values in {col}: {df[col].unique()[:10]}")  # Show first 10 unique values

print("\nData Analysis Completed! ✅")
