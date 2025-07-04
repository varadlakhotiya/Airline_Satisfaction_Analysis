import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
# Replace 'cleaned_dataset.csv' with the actual file path
df = pd.read_csv('cleaned_dataset.csv')

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# 1. Bar Chart: Satisfaction by Travel Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Class', y='Overall_Satisfaction', data=df, ci=None, palette='viridis')
plt.title('Satisfaction by Travel Class', fontsize=16)
plt.xlabel('Travel Class', fontsize=14)
plt.ylabel('Average Satisfaction', fontsize=14)
plt.xticks(ticks=[0, 1, 2], labels=['Economy', 'Business', 'First Class'])
plt.show()

# 2. Pie Chart: Percentage of Satisfied vs. Unsatisfied Passengers
plt.figure(figsize=(6, 6))
satisfied_counts = df['Satisfied'].value_counts()
plt.pie(satisfied_counts, labels=['Not Satisfied', 'Satisfied'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
plt.title('Percentage of Satisfied vs. Unsatisfied Passengers', fontsize=16)
plt.show()

# 3. Heatmap: Correlation Between Service Ratings and Satisfaction
# Select service ratings and overall satisfaction columns
service_columns = [
    'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 'Cabin_Staff_Service',
    'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 'WiFi_Service'
]
correlation_matrix = df[service_columns + ['Overall_Satisfaction']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Service Ratings and Satisfaction', fontsize=16)
plt.show()

# 4. Distribution of Age
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# 5. Distribution of Flight Distance
plt.figure(figsize=(8, 6))
sns.histplot(df['Flight_Distance'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Flight Distance', fontsize=16)
plt.xlabel('Flight Distance (miles)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

# 6. Boxplot: Satisfaction by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Overall_Satisfaction', data=df, palette='pastel')
plt.title('Satisfaction by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Satisfaction', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
plt.show()

# 7. Countplot: Travel Purpose
plt.figure(figsize=(10, 6))
travel_purpose_columns = [
    'Travel_Purpose_Family_True', 'Travel_Purpose_Leisure_True', 'Travel_Purpose_Medical_True',
    'Travel_Purpose_Study_True', 'Travel_Purpose_Transit_True'
]
travel_purpose_counts = df[travel_purpose_columns].sum()
sns.barplot(x=travel_purpose_counts.index, y=travel_purpose_counts.values, palette='magma')
plt.title('Travel Purpose Distribution', fontsize=16)
plt.xlabel('Travel Purpose', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(ticks=range(len(travel_purpose_columns)), labels=['Family', 'Leisure', 'Medical', 'Study', 'Transit'], rotation=45)
plt.show()

# 8. Pairplot: Relationship Between Numeric Features
numeric_columns = ['Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay', 'Flight_Duration', 'Overall_Satisfaction']
sns.pairplot(df[numeric_columns], diag_kind='kde', palette='husl')
plt.suptitle('Pairplot of Numeric Features', y=1.02, fontsize=16)
plt.show()

# 9. Violin Plot: Satisfaction by Frequent Flyer Status
plt.figure(figsize=(8, 6))
sns.violinplot(x='Frequent_Flyer', y='Overall_Satisfaction', data=df, palette='Set2')
plt.title('Satisfaction by Frequent Flyer Status', fontsize=16)
plt.xlabel('Frequent Flyer', fontsize=14)
plt.ylabel('Satisfaction', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.show()

# 10. Stacked Bar Chart: Satisfaction by Travel Class and Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', hue='Gender', data=df, palette='coolwarm')
plt.title('Satisfaction by Travel Class and Gender', fontsize=16)
plt.xlabel('Travel Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(ticks=[0, 1, 2], labels=['Economy', 'Business', 'First Class'])
plt.legend(title='Gender', labels=['Female', 'Male'])
plt.show()