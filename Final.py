import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
# Replace 'cleaned_dataset.csv' with the actual file path
df = pd.read_csv('cleaned_dataset.csv')

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# ------------------------------------------------------------------------------------
# 1. Bar Chart: Satisfaction by Travel Class
# ------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Overall_Satisfaction', data=df, palette='viridis')
plt.title('Satisfaction by Travel Class', fontsize=16)
plt.xlabel('Travel Class', fontsize=14)
plt.ylabel('Average Satisfaction', fontsize=14)
plt.xticks(ticks=[0, 1, 2], labels=['Economy', 'Business', 'First Class'])
plt.show()

# ------------------------------------------------------------------------------------
# 2. Pie Chart: Percentage of Satisfied vs. Unsatisfied Passengers
# ------------------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
satisfied_counts = df['Satisfied'].value_counts()
plt.pie(satisfied_counts, labels=['Not Satisfied', 'Satisfied'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
plt.title('Percentage of Satisfied vs. Unsatisfied Passengers', fontsize=16)
plt.show()

# ------------------------------------------------------------------------------------
# 3. Heatmap: Correlation Between Service Ratings and Satisfaction
# ------------------------------------------------------------------------------------
service_columns = [
    'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 'Cabin_Staff_Service',
    'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 'WiFi_Service'
]

# Calculate correlation matrix
correlation_matrix = df[service_columns + ['Overall_Satisfaction']].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Service Ratings and Satisfaction', fontsize=16)
plt.show()

corr_matrix = df[
    [
      "Seat_Comfort", 
      "InFlight_Entertainment", 
      "Food_Quality",
      "Cleanliness",
      "Cabin_Staff_Service",
      "Legroom",
      "Baggage_Handling",
      "CheckIn_Service",
      "Boarding_Process",
      "Overall_Satisfaction"
    ]
].corr()
print(corr_matrix)


# ------------------------------------------------------------------------------------
# 4. Box Plot: Flight Duration vs. Satisfaction
# ------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='Satisfied', y='Flight_Duration', data=df, palette='Set2')
plt.title('Flight Duration vs. Satisfaction', fontsize=16)
plt.xlabel('Satisfaction (0 = Not Satisfied, 1 = Satisfied)', fontsize=14)
plt.ylabel('Flight Duration (Minutes)', fontsize=14)
plt.show()

# ------------------------------------------------------------------------------------
# 5. Count Plot: Satisfaction by Gender
# ------------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Satisfied', data=df, palette='Set1')
plt.title('Satisfaction by Gender', fontsize=16)
plt.xlabel('Gender (0 = Female, 1 = Male)', fontsize=14)
plt.ylabel('Number of Passengers', fontsize=14)
plt.legend(title='Satisfaction', loc='upper right', labels=['Not Satisfied', 'Satisfied'])
plt.show()

# ------------------------------------------------------------------------------------
# 6. Scatter Plot: Flight Distance vs. Total Delay
# ------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Flight_Distance', y='Total_Delay', hue='Satisfied', data=df, palette='Set2', alpha=0.7)
plt.title('Flight Distance vs. Total Delay', fontsize=16)
plt.xlabel('Flight Distance (Miles)', fontsize=14)
plt.ylabel('Total Delay (Minutes)', fontsize=14)
plt.legend(title='Satisfaction', loc='upper right', labels=['Not Satisfied', 'Satisfied'])
plt.show()

# ------------------------------------------------------------------------------------
# 7. Bar Chart: Average Satisfaction by Airline
# ------------------------------------------------------------------------------------
# Extract airline columns
airline_columns = [col for col in df.columns if 'Airline_Name_' in col]

# Calculate average satisfaction for each airline
airline_satisfaction = {}
for col in airline_columns:
    airline_name = col.replace('Airline_Name_', '').replace('_True', '')
    avg_satisfaction = df[df[col] == 1]['Overall_Satisfaction'].mean()
    airline_satisfaction[airline_name] = avg_satisfaction

# Convert to DataFrame for plotting
airline_satisfaction_df = pd.DataFrame(list(airline_satisfaction.items()), columns=['Airline', 'Average_Satisfaction'])

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Airline', y='Average_Satisfaction', data=airline_satisfaction_df, palette='viridis')
plt.title('Average Satisfaction by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Average Satisfaction', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# ------------------------------------------------------------------------------------
# 8. Dashboard: Combine Multiple Visualizations
# ------------------------------------------------------------------------------------
# Create a dashboard using subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: Satisfaction by Travel Class
sns.barplot(x='Class', y='Overall_Satisfaction', data=df, palette='viridis', ax=axes[0, 0])
axes[0, 0].set_title('Satisfaction by Travel Class', fontsize=14)
axes[0, 0].set_xlabel('Travel Class', fontsize=12)
axes[0, 0].set_ylabel('Average Satisfaction', fontsize=12)
axes[0, 0].set_xticks([0, 1, 2])
axes[0, 0].set_xticklabels(['Economy', 'Business', 'First Class'])

# Plot 2: Percentage of Satisfied vs. Unsatisfied Passengers
satisfied_counts = df['Satisfied'].value_counts()
axes[0, 1].pie(satisfied_counts, labels=['Not Satisfied', 'Satisfied'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
axes[0, 1].set_title('Percentage of Satisfied vs. Unsatisfied Passengers', fontsize=14)

# Plot 3: Correlation Between Service Ratings and Satisfaction
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axes[1, 0])
axes[1, 0].set_title('Correlation Between Service Ratings and Satisfaction', fontsize=14)

# Plot 4: Flight Duration vs. Satisfaction
sns.boxplot(x='Satisfied', y='Flight_Duration', data=df, palette='Set2', ax=axes[1, 1])
axes[1, 1].set_title('Flight Duration vs. Satisfaction', fontsize=14)
axes[1, 1].set_xlabel('Satisfaction (0 = Not Satisfied, 1 = Satisfied)', fontsize=12)
axes[1, 1].set_ylabel('Flight Duration (Minutes)', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()