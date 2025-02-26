import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Encoded_Flight_Data_Processed.csv")

# Display basic statistics
print("Basic Statistics:")
print(df.describe())

# Mode for categorical columns
print("\nMode for Categorical Columns:")
categorical_columns = df.select_dtypes(include=['object']).columns

# Check and print mode for each categorical column
for column in categorical_columns:
    mode_value = df[column].mode()
    if not mode_value.empty:
        print(f"{column}: {mode_value[0]}")
    else:
        print(f"{column}: No mode found")

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Satisfaction distribution (ensure correct categorization)
df['Satisfaction_Status'] = df['Overall_Satisfaction'].apply(lambda x: 'Satisfied' if x == 1 else 'Unsatisfied')

# Calculate satisfaction counts
satisfaction_counts = df['Satisfaction_Status'].value_counts()

# Print satisfaction distribution
print("\nSatisfaction Distribution:")
for satisfaction, count in satisfaction_counts.items():
    print(f"   - {count} passengers are {satisfaction.lower()}.")

# Plot the barplot for overall satisfaction based on 'Class'
plt.figure(figsize=(8, 6))
sns.barplot(x='Class', y='Overall_Satisfaction', data=df, ci=None, palette='viridis')
plt.title("Overall Satisfaction by Class")
plt.show()

# Plot satisfaction counts as a pie chart
plt.figure(figsize=(8, 6))
satisfaction_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title("Satisfaction Distribution")
plt.ylabel('')  # Hide the 'Satisfaction_Status' label
plt.show()

# Additional visualizations can be added here as needed
