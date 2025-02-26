import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the preprocessed dataset
file_path = "Cleaned_Data.csv"  # Change path if necessary
df = pd.read_csv(file_path)

# Display basic info
print("\n🔍 Dataset Overview:")
print(df.info())

# Check for missing values
print("\n🛑 Missing Values Check:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Check for duplicate records
duplicates = df.duplicated().sum()
print(f"\n🔄 Duplicate Records: {duplicates}")

# Summary statistics
print("\n📊 Summary Statistics:")
print(df.describe())

# Check class balance (for classification task)
if 'Overall_Satisfaction' in df.columns:
    print("\n⚖️ Class Balance in Target Variable:")
    print(df['Overall_Satisfaction'].value_counts(normalize=True) * 100)

# Outlier Detection using Z-score
def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(zscore(data))
    return np.where(z_scores > threshold)[0]

# Outlier Detection using IQR
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))[0]

# Columns to check for outliers
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
outlier_report = {}

for col in numerical_columns:
    outliers_z = detect_outliers_zscore(df[col])
    outliers_iqr = detect_outliers_iqr(df[col])
    outlier_report[col] = {
        'Z-score outliers': len(outliers_z),
        'IQR outliers': len(outliers_iqr)
    }

print("\n🚨 Outlier Report:")
for col, report in outlier_report.items():
    print(f"{col}: {report}")

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔗 Feature Correlation Heatmap")
plt.show()

# Distribution plots for key numerical columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_columns[:6], 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"📈 Distribution of {col}")

plt.tight_layout()
plt.show()
