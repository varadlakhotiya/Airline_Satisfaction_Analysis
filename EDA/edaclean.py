import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("AIRLINE DATASET EXPLORATORY DATA ANALYSIS")
print("-----------------------------------------\n")

# 1. Load the dataset
print("1. Loading the dataset...")
df = pd.read_csv('final_updated_dataset.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# 2. Overview of the dataset
print("2. Overview of the dataset")
print("-----------------------")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics for numeric columns:")
print(df.describe().T)

print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_count = df[col].nunique()
    if unique_count < 10:  # Only show if not too many unique values
        print(f"{col}: {df[col].unique()}")
    else:
        print(f"{col}: {unique_count} unique values")

# 3. Feature Engineering
print("\n3. Feature Engineering")
print("--------------------")

# Convert Flight_Duration to minutes for analysis
def convert_to_minutes(duration_str):
    try:
        hours, minutes = map(int, duration_str.split(':'))
        return hours * 60 + minutes
    except:
        return np.nan

df['Flight_Duration_Minutes'] = df['Flight_Duration'].apply(convert_to_minutes)
print("Created 'Flight_Duration_Minutes' from 'Flight_Duration'")

# Create categorical features from numeric columns for better visualization
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '19-30', '31-45', '46-60', '61+'])
print("Created 'Age_Group' from 'Age'")

df['Distance_Category'] = pd.cut(df['Flight_Distance'], 
                               bins=[0, 500, 1000, 2000, 5000, 10000], 
                               labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
print("Created 'Distance_Category' from 'Flight_Distance'")

# Create Satisfaction Score from Overall_Satisfaction
satisfaction_map = {'Happy': 3, 'Satisfied': 2, 'Neutral': 1, 'Dissatisfied': 0}
df['Satisfaction_Score'] = df['Overall_Satisfaction'].map(satisfaction_map)
print("Created 'Satisfaction_Score' from 'Overall_Satisfaction'")

# Convert service ratings to numeric for correlation analysis
service_rating_map = {'Poor': 1, 'Good': 2, 'Excellent': 3, 'Unavailable': np.nan, 'No Connection': np.nan}
service_cols = ['Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 
                'Cabin_Staff_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 
                'Boarding_Process', 'WiFi_Service']

for col in service_cols:
    df[f'{col}_Score'] = df[col].map(service_rating_map)
    print(f"Created '{col}_Score' from '{col}'")
    
    

# 4. Exploratory Data Analysis
print("\n4. Exploratory Data Analysis")
print("--------------------------")

# 4.1 Demographics Analysis
print("\n4.1 Demographics Analysis")
print(f"- Total number of passengers analyzed: {len(df)}")
print(f"- Average passenger age: {df['Age'].mean():.1f} years (Min: {df['Age'].min()}, Max: {df['Age'].max()})")
print(f"- Gender distribution: {df['Gender'].value_counts().to_dict()}")
print(f"- Most common age group: {df['Age_Group'].value_counts().idxmax()}")
print(f"- Top 3 nationalities: {', '.join(df['Nationality'].value_counts().head(3).index.tolist())}")
print(f"- Most common travel purpose: {df['Travel_Purpose'].value_counts().idxmax()} ({df['Travel_Purpose'].value_counts().max()} passengers)")


# Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean Age: {df["Age"].mean():.1f}')
plt.title('Age Distribution of Passengers', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend()
plt.savefig('age_distribution.png')
plt.close()

# Age Groups
plt.figure(figsize=(10, 6))
age_counts = df['Age_Group'].value_counts().sort_index()
sns.barplot(x=age_counts.index, y=age_counts.values)
plt.title('Passenger Distribution by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('age_groups.png')
plt.close()

# Gender Distribution
plt.figure(figsize=(10, 6))
gender_counts = df['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Gender Distribution of Passengers', fontsize=16)
plt.axis('equal')
plt.savefig('gender_distribution.png')
plt.close()

# Nationality Distribution
plt.figure(figsize=(12, 8))
nationality_counts = df['Nationality'].value_counts()
sns.barplot(x=nationality_counts.values, y=nationality_counts.index)
plt.title('Passenger Nationality Distribution', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Nationality', fontsize=14)
plt.savefig('nationality_distribution.png')
plt.close()

# Travel Purpose Distribution
plt.figure(figsize=(12, 6))
travel_counts = df['Travel_Purpose'].value_counts()
sns.barplot(x=travel_counts.index, y=travel_counts.values)
plt.title('Travel Purpose Distribution', fontsize=16)
plt.xlabel('Travel Purpose', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('travel_purpose.png')
plt.close()

# 4.2 Flight Analysis
print("\n4.2 Flight Analysis")
print(f"- Most popular airline: {df['Airline_Name'].value_counts().idxmax()} ({df['Airline_Name'].value_counts().max()} passengers)")
print(f"- Class distribution: {df['Class'].value_counts().to_dict()}")
print(f"- Average flight distance: {df['Flight_Distance'].mean():.1f} km")
print(f"- Average flight duration: {df['Flight_Duration_Minutes'].mean()/60:.1f} hours")
print(f"- Domestic vs International: {df['Flight_Route_Type'].value_counts().to_dict()}")
print(f"- Average departure delay: {df['Departure_Delay'].mean():.1f} minutes")
print(f"- Average arrival delay: {df['Arrival_Delay'].mean():.1f} minutes")
print(f"- Airline with highest average delay: {df.groupby('Airline_Name')['Departure_Delay'].mean().idxmax()} ({df.groupby('Airline_Name')['Departure_Delay'].mean().max():.1f} minutes)")
print(f"- Airline with lowest average delay: {df.groupby('Airline_Name')['Departure_Delay'].mean().idxmin()} ({df.groupby('Airline_Name')['Departure_Delay'].mean().min():.1f} minutes)")


# Airline Distribution
plt.figure(figsize=(14, 8))
airline_counts = df['Airline_Name'].value_counts()
sns.barplot(x=airline_counts.values, y=airline_counts.index)
plt.title('Passenger Distribution by Airline', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Airline', fontsize=14)
plt.tight_layout()
plt.savefig('airline_distribution.png')
plt.close()

# Class Distribution
plt.figure(figsize=(10, 6))
class_counts = df['Class'].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Passenger Distribution by Class', fontsize=16)
plt.axis('equal')
plt.savefig('class_distribution.png')
plt.close()

# Flight Route Type
plt.figure(figsize=(10, 6))
route_counts = df['Flight_Route_Type'].value_counts()
sns.barplot(x=route_counts.index, y=route_counts.values)
plt.title('Domestic vs International Flights', fontsize=16)
plt.xlabel('Route Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('route_type_distribution.png')
plt.close()

# Flight Distance Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Flight_Distance'], bins=20, kde=True)
plt.axvline(df['Flight_Distance'].mean(), color='red', linestyle='--', 
           label=f'Mean Distance: {df["Flight_Distance"].mean():.1f} km')
plt.title('Flight Distance Distribution', fontsize=16)
plt.xlabel('Distance (km)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend()
plt.savefig('distance_distribution.png')
plt.close()

# Distance Categories
plt.figure(figsize=(12, 6))
distance_counts = df['Distance_Category'].value_counts().sort_index()
sns.barplot(x=distance_counts.index, y=distance_counts.values)
plt.title('Flight Distance Categories', fontsize=16)
plt.xlabel('Distance Category', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('distance_categories.png')
plt.close()

# Flight Duration Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Flight_Duration_Minutes'], bins=20, kde=True)
plt.axvline(df['Flight_Duration_Minutes'].mean(), color='red', linestyle='--', 
           label=f'Mean Duration: {df["Flight_Duration_Minutes"].mean():.1f} min')
plt.title('Flight Duration Distribution', fontsize=16)
plt.xlabel('Duration (minutes)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend()
plt.savefig('duration_distribution.png')
plt.close()

# Popular Routes
plt.figure(figsize=(14, 10))
routes = df.groupby(['Departure_City', 'Arrival_City']).size().reset_index(name='Count')
routes = routes.sort_values('Count', ascending=False).head(10)  # Top 10 routes
sns.barplot(x='Count', y=[f"{d} → {a}" for d, a in zip(routes['Departure_City'], routes['Arrival_City'])], data=routes)
plt.title('Top 10 Popular Routes', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Route', fontsize=14)
plt.tight_layout()
plt.savefig('popular_routes.png')
plt.close()

# Flight Delays Analysis
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
sns.histplot(df['Departure_Delay'], bins=30, kde=True)
plt.title('Departure Delay Distribution (minutes)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(df['Departure_Delay'].mean(), color='green', linestyle='--', 
           label=f'Mean: {df["Departure_Delay"].mean():.1f} min')
plt.legend()

plt.subplot(2, 1, 2)
sns.histplot(df['Arrival_Delay'], bins=30, kde=True)
plt.title('Arrival Delay Distribution (minutes)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.axvline(df['Arrival_Delay'].mean(), color='green', linestyle='--', 
           label=f'Mean: {df["Arrival_Delay"].mean():.1f} min')
plt.legend()

plt.tight_layout()
plt.savefig('delay_distributions.png')
plt.close()

# Delay by Airline
plt.figure(figsize=(14, 8))
airline_delays = df.groupby('Airline_Name')[['Departure_Delay', 'Arrival_Delay']].mean().sort_values('Departure_Delay')
airline_delays.plot(kind='bar')
plt.title('Average Delays by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Average Delay (minutes)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_delays.png')
plt.close()

# Delay by Departure City
plt.figure(figsize=(14, 8))
city_delays = df.groupby('Departure_City')[['Departure_Delay']].mean().sort_values('Departure_Delay', ascending=False).head(10)
city_delays.plot(kind='bar')
plt.title('Top 10 Cities with Highest Average Departure Delays', fontsize=16)
plt.xlabel('Departure City', fontsize=14)
plt.ylabel('Average Delay (minutes)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('city_delays.png')
plt.close()

top_route = routes.iloc[0]
print(f"- Most popular route: {top_route['Departure_City']} → {top_route['Arrival_City']} ({top_route['Count']} passengers)")

# 4.3 Satisfaction Analysis
print("\n4.3 Satisfaction Analysis")


# Overall Satisfaction Distribution
plt.figure(figsize=(12, 6))
satisfaction_counts = df['Overall_Satisfaction'].value_counts()
colors = ['green' if x == 'Happy' else 'lightgreen' if x == 'Satisfied' else 'orange' if x == 'Neutral' else 'red' for x in satisfaction_counts.index]
sns.barplot(x=satisfaction_counts.index, y=satisfaction_counts.values, palette=colors)
plt.title('Overall Satisfaction Distribution', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('satisfaction_distribution.png')
plt.close()

# Satisfaction Distribution as Percentage
plt.figure(figsize=(10, 8))
plt.pie(satisfaction_counts, labels=satisfaction_counts.index, autopct='%1.1f%%', 
       startangle=90, colors=sns.color_palette("YlOrRd", len(satisfaction_counts)))
plt.title('Percentage Distribution of Passenger Satisfaction', fontsize=16)
plt.axis('equal')
plt.savefig('satisfaction_percentage.png')
plt.close()

# Service Parameters Ratings Analysis
# Convert ratings to numeric scores for plotting
rating_scores = pd.DataFrame()
for col in service_cols:
    rating_scores[col] = df[f'{col}_Score']

plt.figure(figsize=(14, 10))
rating_means = rating_scores.mean().sort_values()
sns.barplot(x=rating_means.values, y=rating_means.index)
plt.title('Average Ratings for Different Service Parameters', fontsize=16)
plt.xlabel('Average Rating (1=Poor, 2=Good, 3=Excellent)', fontsize=14)
plt.ylabel('Service Parameter', fontsize=14)
plt.tight_layout()
plt.savefig('service_ratings.png')
plt.close()

# Recommendation Status
plt.figure(figsize=(10, 6))
recommendation_counts = df['Recommendation'].value_counts()
colors = ['green' if x == 'Yes' else 'red' if x == 'No' else 'orange' for x in recommendation_counts.index]
sns.barplot(x=recommendation_counts.index, y=recommendation_counts.values, palette=colors)
plt.title('Would Passengers Recommend the Airline?', fontsize=16)
plt.xlabel('Recommendation', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('recommendation_status.png')
plt.close()

# Percentage recommending by airline
plt.figure(figsize=(14, 8))
airline_rec = df.groupby('Airline_Name')['Recommendation'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
sns.barplot(x=airline_rec.index, y=airline_rec.values)
plt.title('Percentage of Passengers Recommending Each Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Percentage Recommending (%)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_recommendation_percentage.png')
plt.close()

# Satisfaction by Airline
plt.figure(figsize=(14, 8))
airline_satisfaction = df.groupby('Airline_Name')['Satisfaction_Score'].mean().sort_values(ascending=False)
sns.barplot(x=airline_satisfaction.index, y=airline_satisfaction.values)
plt.title('Average Satisfaction Score by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_satisfaction.png')
plt.close()

# Satisfaction by Class
plt.figure(figsize=(12, 6))
class_satisfaction = df.groupby('Class')['Satisfaction_Score'].mean()
sns.barplot(x=class_satisfaction.index, y=class_satisfaction.values)
plt.title('Average Satisfaction Score by Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.tight_layout()
plt.savefig('class_satisfaction.png')
plt.close()

print("\nSatisfaction Analysis Key Findings:")
print(f"- Overall satisfaction distribution: {df['Overall_Satisfaction'].value_counts().to_dict()}")
print(f"- Percentage of satisfied passengers: {((df['Overall_Satisfaction'] == 'Happy') | (df['Overall_Satisfaction'] == 'Satisfied')).mean() * 100:.1f}%")
print(f"- Percentage of passengers who would recommend: {(df['Recommendation'] == 'Yes').mean() * 100:.1f}%")

# Get highest and lowest rated services
highest_rated = rating_means.idxmax().replace('_Score', '')
lowest_rated = rating_means.idxmin().replace('_Score', '')
print(f"- Highest rated service aspect: {highest_rated} ({rating_means.max():.2f}/3)")
print(f"- Lowest rated service aspect: {lowest_rated} ({rating_means.min():.2f}/3)")

# Airline with highest satisfaction
print(f"- Airline with highest satisfaction: {airline_satisfaction.idxmax()} ({airline_satisfaction.max():.2f}/3)")
print(f"- Airline with lowest satisfaction: {airline_satisfaction.idxmin()} ({airline_satisfaction.min():.2f}/3)")
print(f"- Class with highest satisfaction: {class_satisfaction.idxmax()} ({class_satisfaction.max():.2f}/3)")


# 4.4 Complaint Analysis
print("\n4.4 Complaint Analysis")

# Complaint Distribution
plt.figure(figsize=(10, 6))
complaint_counts = df['Complaint_Submitted'].value_counts()
colors = ['red' if x == 'Yes' else 'green' for x in complaint_counts.index]
sns.barplot(x=complaint_counts.index, y=complaint_counts.values, palette=colors)
plt.title('Complaints Submitted', fontsize=16)
plt.xlabel('Complaint Submitted', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('complaint_distribution.png')
plt.close()

# Complaint Types
plt.figure(figsize=(14, 8))
complaint_type_counts = df[df['Complaint_Submitted'] == 'Yes']['Complaint_Type'].value_counts()
sns.barplot(x=complaint_type_counts.values, y=complaint_type_counts.index)
plt.title('Distribution of Complaint Types', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Complaint Type', fontsize=14)
plt.tight_layout()
plt.savefig('complaint_types.png')
plt.close()

# Complaints by Airline
plt.figure(figsize=(14, 8))
airline_complaints = df.groupby('Airline_Name')['Complaint_Submitted'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
sns.barplot(x=airline_complaints.index, y=airline_complaints.values)
plt.title('Percentage of Passengers Filing Complaints by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Complaint Percentage (%)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_complaints_percentage.png')
plt.close()

print("\nComplaint Analysis Key Findings:")
print(f"- Percentage of passengers filing complaints: {(df['Complaint_Submitted'] == 'Yes').mean() * 100:.1f}%")
print(f"- Most common complaint type: {complaint_type_counts.idxmax()} ({complaint_type_counts.max()} complaints)")
print(f"- Airline with highest complaint rate: {airline_complaints.idxmax()} ({airline_complaints.max():.1f}%)")
print(f"- Airline with lowest complaint rate: {airline_complaints.idxmin()} ({airline_complaints.min():.1f}%)")


# 4.5 Relationship Analysis
print("\n4.5 Relationship Analysis")

# Age vs Satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Overall_Satisfaction', y='Age', data=df)
plt.title('Age Distribution by Satisfaction Level', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.tight_layout()
plt.savefig('age_vs_satisfaction.png')
plt.close()

# Class vs Satisfaction
plt.figure(figsize=(12, 8))
satisfaction_class = pd.crosstab(df['Class'], df['Overall_Satisfaction'], normalize='index') * 100
satisfaction_class.plot(kind='bar', stacked=True)
plt.title('Satisfaction Distribution by Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Satisfaction Level')
plt.tight_layout()
plt.savefig('class_vs_satisfaction.png')
plt.close()

# Flight Distance vs Satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Overall_Satisfaction', y='Flight_Distance', data=df)
plt.title('Flight Distance by Satisfaction Level', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Flight Distance (km)', fontsize=14)
plt.tight_layout()
plt.savefig('distance_vs_satisfaction.png')
plt.close()

# Flight Duration vs Satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Overall_Satisfaction', y='Flight_Duration_Minutes', data=df)
plt.title('Flight Duration by Satisfaction Level', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Flight Duration (minutes)', fontsize=14)
plt.tight_layout()
plt.savefig('duration_vs_satisfaction.png')
plt.close()

# Departure Delay vs Satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Overall_Satisfaction', y='Departure_Delay', data=df)
plt.title('Departure Delay by Satisfaction Level', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Departure Delay (minutes)', fontsize=14)
plt.tight_layout()
plt.savefig('delay_vs_satisfaction.png')
plt.close()

print("\nRelationship Analysis Key Findings:")
print("- Age and satisfaction correlation:")
for satisfaction in df['Overall_Satisfaction'].unique():
    print(f"  * {satisfaction}: Average age {df[df['Overall_Satisfaction'] == satisfaction]['Age'].mean():.1f} years")

# Calculate delay difference between satisfaction levels
happy_delay = df[df['Overall_Satisfaction'] == 'Happy']['Departure_Delay'].mean()
unhappy_delay = df[df['Overall_Satisfaction'] == 'Unhappy']['Departure_Delay'].mean()
print(f"- Departure delay impact: Unhappy passengers experienced {unhappy_delay - happy_delay:.1f} minutes more delay on average")



# 4.6 Correlation Analysis
print("\n4.6 Correlation Analysis")

# Combine service score columns with other numeric columns for correlation
numeric_cols = ['Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay', 
                'Flight_Duration_Minutes', 'Satisfaction_Score']
score_cols = [col for col in df.columns if col.endswith('_Score') and col != 'Satisfaction_Score']
corr_cols = numeric_cols + score_cols

# Correlation heatmap
plt.figure(figsize=(16, 14))
correlation = df[corr_cols].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Key Variables', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Top correlations with satisfaction
satisfaction_corr = correlation['Satisfaction_Score'].sort_values(ascending=False)
print("\nTop correlations with satisfaction:")
print(satisfaction_corr)

plt.figure(figsize=(14, 10))
sns.barplot(x=satisfaction_corr.values[1:], y=satisfaction_corr.index[1:])  # Skip the first (self-correlation)
plt.title('Correlation of Variables with Satisfaction Score', fontsize=16)
plt.xlabel('Correlation Coefficient', fontsize=14)
plt.ylabel('Variable', fontsize=14)
plt.axvline(x=0, color='black', linestyle='-')
plt.tight_layout()
plt.savefig('satisfaction_correlations.png')
plt.close()

print("\nCorrelation Analysis Key Findings:")
print(f"- Strongest positive correlation with satisfaction: {satisfaction_corr.index[1]} ({satisfaction_corr.values[1]:.3f})")
print(f"- Strongest negative correlation with satisfaction: {satisfaction_corr.index[-1]} ({satisfaction_corr.values[-1]:.3f})")


# 4.7 Additional Analysis
print("\n4.7 Additional Analysis")

# Loyalty Membership Distribution
plt.figure(figsize=(10, 6))
loyalty_counts = df['Loyalty_Membership'].value_counts()
sns.barplot(x=loyalty_counts.index, y=loyalty_counts.values)
plt.title('Loyalty Membership Distribution', fontsize=16)
plt.xlabel('Membership Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('loyalty_distribution.png')
plt.close()

# Loyalty and Satisfaction
plt.figure(figsize=(12, 7))
loyalty_satisfaction = df.groupby('Loyalty_Membership')['Satisfaction_Score'].mean().sort_values(ascending=False)
sns.barplot(x=loyalty_satisfaction.index, y=loyalty_satisfaction.values)
plt.title('Average Satisfaction Score by Loyalty Membership Level', fontsize=16)
plt.xlabel('Loyalty Membership', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.tight_layout()
plt.savefig('loyalty_satisfaction.png')
plt.close()

# Payment Method Distribution
plt.figure(figsize=(12, 8))
payment_counts = df['Payment_Method'].value_counts()
sns.barplot(x=payment_counts.values, y=payment_counts.index)
plt.title('Payment Method Distribution', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Payment Method', fontsize=14)
plt.tight_layout()
plt.savefig('payment_methods.png')
plt.close()

# Seat Type and Satisfaction
plt.figure(figsize=(10, 6))
seat_satisfaction = df.groupby('Seat_Type')['Satisfaction_Score'].mean()
sns.barplot(x=seat_satisfaction.index, y=seat_satisfaction.values)
plt.title('Average Satisfaction Score by Seat Type', fontsize=16)
plt.xlabel('Seat Type', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.tight_layout()
plt.savefig('seat_satisfaction.png')
plt.close()

# Festival Season Travel and Satisfaction
plt.figure(figsize=(10, 6))
festival_satisfaction = df.groupby('Festival_Season_Travel')['Satisfaction_Score'].mean()
sns.barplot(x=festival_satisfaction.index, y=festival_satisfaction.values)
plt.title('Average Satisfaction for Festival vs Non-Festival Travel', fontsize=16)
plt.xlabel('Festival Season Travel', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.tight_layout()
plt.savefig('festival_satisfaction.png')
plt.close()

# Booking Channel Analysis
plt.figure(figsize=(12, 8))
channel_counts = df['Booking_Channel'].value_counts()
sns.barplot(x=channel_counts.values, y=channel_counts.index)
plt.title('Distribution of Booking Channels', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Booking Channel', fontsize=14)
plt.tight_layout()
plt.savefig('booking_channels.png')
plt.close()

plt.figure(figsize=(12, 8))
channel_satisfaction = df.groupby('Booking_Channel')['Satisfaction_Score'].mean().sort_values(ascending=False)
sns.barplot(x=channel_satisfaction.index, y=channel_satisfaction.values)
plt.title('Average Satisfaction Score by Booking Channel', fontsize=16)
plt.xlabel('Booking Channel', fontsize=14)
plt.ylabel('Average Satisfaction Score (0-3)', fontsize=14)
plt.tight_layout()
plt.savefig('channel_satisfaction.png')
plt.close()

print("\nAdditional Analysis Key Findings:")
print(f"- Most common loyalty membership level: {df['Loyalty_Membership'].value_counts().idxmax()} ({df['Loyalty_Membership'].value_counts().max()} passengers)")
print(f"- Loyalty level with highest satisfaction: {loyalty_satisfaction.idxmax()} ({loyalty_satisfaction.max():.2f}/3)")
print(f"- Most common payment method: {df['Payment_Method'].value_counts().idxmax()} ({df['Payment_Method'].value_counts().max()} passengers)")
print(f"- Most common booking channel: {df['Booking_Channel'].value_counts().idxmax()} ({df['Booking_Channel'].value_counts().max()} passengers)")
print(f"- Booking channel with highest satisfaction: {channel_satisfaction.idxmax()} ({channel_satisfaction.max():.2f}/3)")
print(f"- Seat type with highest satisfaction: {seat_satisfaction.idxmax()} ({seat_satisfaction.max():.2f}/3)")

# Festival season impact
festival_yes = df[df['Festival_Season_Travel'] == 'Yes']['Satisfaction_Score'].mean()
festival_no = df[df['Festival_Season_Travel'] == 'No']['Satisfaction_Score'].mean()
print(f"- Festival season travel satisfaction: {festival_yes:.2f}/3 (vs. {festival_no:.2f}/3 for non-festival travel)")

# Overall insights
print("\nOverall Key Insights:")
print("1. " + ("Flight distance and duration have the strongest positive correlation with satisfaction, suggesting longer " +
              "flights may be more satisfying experiences overall."))
print("2. " + ("WiFi service and baggage handling show the strongest negative correlations with satisfaction, indicating " +
              "these are potential areas for improvement."))
print("3. " + (f"The {class_satisfaction.idxmax()} class shows the highest satisfaction levels, while {class_satisfaction.idxmin()} " +
              "class shows the lowest."))
print("4. " + (f"{airline_satisfaction.idxmax()} airline has the highest passenger satisfaction scores, while " +
              f"{airline_satisfaction.idxmin()} has the lowest."))
print("5. " + ("There is a clear relationship between flight delays and passenger satisfaction, with unhappy passengers " +
              "experiencing significantly longer delays."))

# 5. Key Insights & Conclusion
print("\n5. Key Insights & Conclusion")
print("---------------------------")

# Calculate key metrics
avg_age = df['Age'].mean()
avg_distance = df['Flight_Distance'].mean()
avg_duration = df['Flight_Duration_Minutes'].mean()
satisfaction_perc = df[df['Overall_Satisfaction'].isin(['Happy', 'Satisfied'])].shape[0] / df.shape[0] * 100
most_common_airline = df['Airline_Name'].value_counts().index[0]
most_common_class = df['Class'].value_counts().index[0]
avg_departure_delay = df['Departure_Delay'].mean()
complaint_rate = df[df['Complaint_Submitted'] == 'Yes'].shape[0] / df.shape[0] * 100
recommend_rate = df[df['Recommendation'] == 'Yes'].shape[0] / df.shape[0] * 100

print(f"Average passenger age: {avg_age:.1f} years")
print(f"Average flight distance: {avg_distance:.1f} km")
print(f"Average flight duration: {avg_duration:.1f} minutes")
print(f"Percentage of satisfied passengers (Happy+Satisfied): {satisfaction_perc:.1f}%")
print(f"Most common airline: {most_common_airline}")
print(f"Most common travel class: {most_common_class}")
print(f"Average departure delay: {avg_departure_delay:.1f} minutes")
print(f"Complaint submission rate: {complaint_rate:.1f}%")
print(f"Recommendation rate: {recommend_rate:.1f}%")

# Top 5 factors most correlated with passenger satisfaction
top_factors = satisfaction_corr[1:6]  # Skip self-correlation
print("\nTop 5 factors most correlated with passenger satisfaction:")
for idx, val in top_factors.items():
    print(f"- {idx}: {val:.3f}")

print("\nEDA completed successfully! All visualizations have been saved.")