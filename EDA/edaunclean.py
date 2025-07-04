import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots - updated style name for newer versions
plt.style.use('seaborn-v0_8-whitegrid')  # Updated from 'seaborn-whitegrid'
# Alternative fix: just use a basic style that's always available
# plt.style.use('ggplot')  # Uncomment this and comment the line above if still having issues
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("AIRLINE DATASET EXPLORATORY DATA ANALYSIS")
print("-----------------------------------------\n")

# 1. Load the dataset
print("1. Loading the dataset...")
df = pd.read_csv('airline_passenger_satisfaction.csv')
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# 2. Overview of the dataset
print("2. Overview of the dataset")
print("-----------------------")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe().T)

# 3. Data Cleaning
print("\n3. Data Cleaning")
print("--------------")

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 
                          'Percentage': missing_percentage})
print(missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False))

# Visualize missing values
plt.figure(figsize=(15, 10))
msno.matrix(df)
plt.title('Missing Values in Airline Dataset', fontsize=16)
plt.tight_layout()
plt.savefig('missing_values_matrix.png')
plt.close()

# Check for inconsistent and invalid values
print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df[col].unique()
    if len(unique_vals) < 20:  # Only show if not too many unique values
        print(f"{col}: {unique_vals}")

# Handle numeric inconsistencies
print("\nChecking for numeric inconsistencies...")
# Age validation - negative or extremely high values
invalid_age = df[(df['Age'] < 0) | (df['Age'] > 120)].shape[0]
print(f"Invalid age values (< 0 or > 120): {invalid_age}")

# Flight Distance validation - negative values
invalid_distance = df[df['Flight_Distance'] < 0].shape[0]
print(f"Invalid flight distance values (< 0): {invalid_distance}")

# Clean the data
print("\nCleaning the data...")
df_clean = df.copy()

# Handling negative and extreme values
df_clean.loc[df_clean['Age'] < 0, 'Age'] = np.nan
df_clean.loc[df_clean['Age'] > 120, 'Age'] = np.nan
df_clean.loc[df_clean['Flight_Distance'] < 0, 'Flight_Distance'] = np.nan

# Handling missing values for important numeric columns
numeric_cols = ['Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay', 'Flight_Duration']
imputer = SimpleImputer(strategy='median')
df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])

# Standardize some categorical values
if 'Gender' in df_clean.columns:
    df_clean['Gender'] = df_clean['Gender'].replace({'Unknown': 'Other', 'X': 'Other'})

if 'Nationality' in df_clean.columns:
    df_clean['Nationality'] = df_clean['Nationality'].replace({'Ind': 'Indian', 'Indien': 'Indian'})

print("Data cleaning completed.\n")

# 4. Exploratory Data Analysis
print("4. Exploratory Data Analysis")
print("--------------------------")

# 4.1 Demographics Analysis
print("\n4.1 Demographics Analysis")

# Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df_clean['Age'].dropna(), bins=20, kde=True)
plt.title('Age Distribution of Passengers', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('age_distribution.png')
plt.close()

# Print age distribution findings
print(f"Age distribution findings:")
print(f"- Average passenger age: {df_clean['Age'].mean():.1f} years")
print(f"- Youngest passenger: {df_clean['Age'].min():.0f} years")
print(f"- Oldest passenger: {df_clean['Age'].max():.0f} years")
print(f"- Age groups: {df_clean['Age'].value_counts(bins=5).sort_index().to_dict()}")

# Gender Distribution
if 'Gender' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    gender_counts = df_clean['Gender'].value_counts()
    sns.barplot(x=gender_counts.index, y=gender_counts.values)
    plt.title('Gender Distribution of Passengers', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.savefig('gender_distribution.png')
    plt.close()
    
    print("\nGender distribution findings:")
    for gender, count in gender_counts.items():
        percentage = (count / gender_counts.sum()) * 100
        print(f"- {gender}: {count} passengers ({percentage:.1f}%)")

# Nationality Distribution
if 'Nationality' in df_clean.columns:
    plt.figure(figsize=(12, 8))
    nationality_counts = df_clean['Nationality'].value_counts().head(10)  # Top 10 nationalities
    sns.barplot(x=nationality_counts.values, y=nationality_counts.index)
    plt.title('Top 10 Passenger Nationalities', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Nationality', fontsize=14)
    plt.savefig('nationality_distribution.png')
    plt.close()
    
    print("\nNationality distribution findings:")
    print(f"- Most common nationality: {nationality_counts.index[0]} ({nationality_counts.values[0]} passengers)")
    for nationality, count in nationality_counts.items():
        percentage = (count / nationality_counts.sum()) * 100
        print(f"- {nationality}: {count} passengers ({percentage:.1f}%)")

# 4.2 Flight Analysis
print("\n4.2 Flight Analysis")

# Airline Distribution
plt.figure(figsize=(12, 8))
airline_counts = df_clean['Airline_Name'].value_counts()
sns.barplot(x=airline_counts.values, y=airline_counts.index)
plt.title('Passenger Distribution by Airline', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Airline', fontsize=14)
plt.savefig('airline_distribution.png')
plt.close()

print("\nAirline distribution findings:")
print(f"- Total airlines: {len(airline_counts)}")
print(f"- Most popular airline: {airline_counts.index[0]} with {airline_counts.values[0]} passengers ({(airline_counts.values[0]/sum(airline_counts.values))*100:.1f}%)")
print(f"- Least popular airline: {airline_counts.index[-1]} with {airline_counts.values[-1]} passengers ({(airline_counts.values[-1]/sum(airline_counts.values))*100:.1f}%)")
for airline, count in airline_counts.items():
    percentage = (count / airline_counts.sum()) * 100
    print(f"- {airline}: {count} passengers ({percentage:.1f}%)")

# Class Distribution
plt.figure(figsize=(10, 6))
class_counts = df_clean['Class'].value_counts()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Passenger Distribution by Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('class_distribution.png')
plt.close()

print("\nClass distribution findings:")
for travel_class, count in class_counts.items():
    percentage = (count / class_counts.sum()) * 100
    print(f"- {travel_class}: {count} passengers ({percentage:.1f}%)")

# Flight Route Type
plt.figure(figsize=(10, 6))
route_counts = df_clean['Flight_Route_Type'].value_counts()
sns.barplot(x=route_counts.index, y=route_counts.values)
plt.title('Domestic vs International Flights', fontsize=16)
plt.xlabel('Route Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('route_type_distribution.png')
plt.close()

print("\nFlight route type findings:")
for route_type, count in route_counts.items():
    percentage = (count / route_counts.sum()) * 100
    print(f"- {route_type}: {count} flights ({percentage:.1f}%)")

# Popular Routes
plt.figure(figsize=(14, 10))
routes = df_clean.groupby(['Departure_City', 'Arrival_City']).size().reset_index(name='Count')
routes = routes.sort_values('Count', ascending=False).head(10)  # Top 10 routes
sns.barplot(x='Count', y=[f"{d} → {a}" for d, a in zip(routes['Departure_City'], routes['Arrival_City'])], data=routes)
plt.title('Top 10 Popular Routes', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Route', fontsize=14)
plt.savefig('popular_routes.png')
plt.close()

print("\nPopular routes findings:")
print(f"- Total unique routes: {len(df_clean.groupby(['Departure_City', 'Arrival_City']).size())}")
print(f"- Top 5 most popular routes:")
for i, (_, row) in enumerate(routes.head(5).iterrows()):
    print(f"  {i+1}. {row['Departure_City']} → {row['Arrival_City']}: {row['Count']} flights")

# Flight Delays Analysis
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
sns.histplot(df_clean['Departure_Delay'].dropna(), bins=30, kde=True)
plt.title('Departure Delay Distribution (minutes)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')

plt.subplot(2, 1, 2)
sns.histplot(df_clean['Arrival_Delay'].dropna(), bins=30, kde=True)
plt.title('Arrival Delay Distribution (minutes)', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('delay_distributions.png')
plt.close()

print("\nFlight delay findings:")
print(f"- Average departure delay: {df_clean['Departure_Delay'].mean():.1f} minutes")
print(f"- Median departure delay: {df_clean['Departure_Delay'].median():.1f} minutes")
print(f"- Maximum departure delay: {df_clean['Departure_Delay'].max():.1f} minutes")
print(f"- Percentage of flights with departure delays > 15 minutes: {(df_clean['Departure_Delay'] > 15).mean() * 100:.1f}%")
print(f"- Average arrival delay: {df_clean['Arrival_Delay'].mean():.1f} minutes")
print(f"- Flights with negative delays (early arrivals): {(df_clean['Arrival_Delay'] < 0).sum()} flights")

# Delay by Airline
plt.figure(figsize=(14, 8))
airline_delays = df_clean.groupby('Airline_Name')[['Departure_Delay', 'Arrival_Delay']].mean().sort_values('Departure_Delay')
airline_delays.plot(kind='bar')
plt.title('Average Delays by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Average Delay (minutes)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_delays.png')
plt.close()

print("\nAirline delay comparison findings:")
print("- Average delays by airline (in minutes):")
for airline, row in airline_delays.iterrows():
    print(f"  {airline}: Departure = {row['Departure_Delay']:.1f}, Arrival = {row['Arrival_Delay']:.1f}")
print(f"- Airline with best on-time departure performance: {airline_delays.index[0]} ({airline_delays['Departure_Delay'].iloc[0]:.1f} min)")
print(f"- Airline with worst on-time departure performance: {airline_delays.index[-1]} ({airline_delays['Departure_Delay'].iloc[-1]:.1f} min)")

# 4.3 Satisfaction Analysis
print("\n4.3 Satisfaction Analysis")

# Overall Satisfaction Distribution
plt.figure(figsize=(12, 6))
satisfaction_counts = df_clean['Overall_Satisfaction'].value_counts()
sns.barplot(x=satisfaction_counts.index, y=satisfaction_counts.values)
plt.title('Overall Satisfaction Distribution', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('satisfaction_distribution.png')
plt.close()

print("\nOverall satisfaction findings:")
total_rated = satisfaction_counts.sum()
for rating, count in satisfaction_counts.items():
    percentage = (count / total_rated) * 100
    print(f"- {rating}: {count} passengers ({percentage:.1f}%)")
print(f"- Overall positive ratings (Happy + Satisfied): {satisfaction_counts.get('Happy', 0) + satisfaction_counts.get('Satisfied', 0)} ({(satisfaction_counts.get('Happy', 0) + satisfaction_counts.get('Satisfied', 0))/total_rated*100:.1f}%)")
print(f"- Overall negative ratings (Dissatisfied): {satisfaction_counts.get('Dissatisfied', 0)} ({satisfaction_counts.get('Dissatisfied', 0)/total_rated*100:.1f}%)")

# Service Parameters Ratings Analysis
service_cols = ['Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 
                'Cabin_Staff_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service']

# Convert text ratings to numeric where possible
for col in service_cols:
    if col in df_clean.columns:
        # Try to convert 'Good', 'Poor' etc. to numeric values
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        except:
            # If column contains mixed numeric and text values, handle accordingly
            # For simplicity, treat as categorical here
            pass

# Service Ratings (numeric columns only)
numeric_service_cols = [col for col in service_cols if pd.api.types.is_numeric_dtype(df_clean[col])]
if numeric_service_cols:
    plt.figure(figsize=(14, 10))
    service_means = df_clean[numeric_service_cols].mean().sort_values()
    service_means.plot(kind='barh')
    plt.title('Average Ratings for Different Service Parameters', fontsize=16)
    plt.xlabel('Average Rating', fontsize=14)
    plt.ylabel('Service Parameter', fontsize=14)
    plt.tight_layout()
    plt.savefig('service_ratings.png')
    plt.close()
    
    print("\nService ratings findings:")
    print("- Average ratings for service parameters (0-10 scale):")
    for service, mean in service_means.items():
        print(f"  {service}: {mean:.2f}/10")
    print(f"- Highest rated service: {service_means.index[-1]} ({service_means.iloc[-1]:.2f}/10)")
    print(f"- Lowest rated service: {service_means.index[0]} ({service_means.iloc[0]:.2f}/10)")

# Recommendation Status
if 'Recommendation' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    recommendation_counts = df_clean['Recommendation'].value_counts()
    colors = ['green' if x == 'Yes' else 'red' if x == 'No' else 'gray' for x in recommendation_counts.index]
    sns.barplot(x=recommendation_counts.index, y=recommendation_counts.values, palette=colors)
    plt.title('Would Passengers Recommend the Airline?', fontsize=16)
    plt.xlabel('Recommendation', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig('recommendation_status.png')
    plt.close()
    
    print("\nRecommendation status findings:")
    total = recommendation_counts.sum()
    for status, count in recommendation_counts.items():
        percentage = (count / total) * 100
        print(f"- {status}: {count} passengers ({percentage:.1f}%)")
    recommend_pct = (recommendation_counts.get('Yes', 0) / total) * 100
    print(f"- Net recommendation rate: {recommend_pct:.1f}%")

# Satisfaction by Airline
plt.figure(figsize=(14, 8))
# Create a satisfaction score (Happy=3, Satisfied=2, Neutral=1, Dissatisfied=0)
satisfaction_map = {'Happy': 3, 'Satisfied': 2, 'Neutral': 1, 'Dissatisfied': 0}
df_clean['Satisfaction_Score'] = df_clean['Overall_Satisfaction'].map(satisfaction_map)
airline_satisfaction = df_clean.groupby('Airline_Name')['Satisfaction_Score'].mean().sort_values(ascending=False)
sns.barplot(x=airline_satisfaction.index, y=airline_satisfaction.values)
plt.title('Average Satisfaction Score by Airline', fontsize=16)
plt.xlabel('Airline', fontsize=14)
plt.ylabel('Average Satisfaction Score', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('airline_satisfaction.png')
plt.close()

print("\nAirline satisfaction comparison findings:")
print("- Average satisfaction score by airline (scale 0-3):")
for airline, score in airline_satisfaction.items():
    print(f"  {airline}: {score:.2f}")
print(f"- Airline with highest satisfaction: {airline_satisfaction.index[0]} ({airline_satisfaction.iloc[0]:.2f}/3)")
print(f"- Airline with lowest satisfaction: {airline_satisfaction.index[-1]} ({airline_satisfaction.iloc[-1]:.2f}/3)")

# 4.4 Complaint Analysis
print("\n4.4 Complaint Analysis")

# Complaint Distribution
if 'Complaint_Submitted' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    complaint_counts = df_clean['Complaint_Submitted'].value_counts()
    colors = ['red' if x == 'Yes' else 'green' for x in complaint_counts.index]
    sns.barplot(x=complaint_counts.index, y=complaint_counts.values, palette=colors)
    plt.title('Complaints Submitted', fontsize=16)
    plt.xlabel('Complaint Submitted', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig('complaint_distribution.png')
    plt.close()
    
    print("\nComplaint submission findings:")
    yes_complaints = complaint_counts.get('Yes', 0)
    total = complaint_counts.sum()
    complaint_rate = (yes_complaints / total) * 100
    print(f"- Complaint rate: {complaint_rate:.1f}% ({yes_complaints} out of {total} passengers)")
    print(f"- No complaints: {complaint_counts.get('No', 0)} passengers ({(complaint_counts.get('No', 0) / total) * 100:.1f}%)")

# Complaint Types
if 'Complaint_Type' in df_clean.columns:
    plt.figure(figsize=(14, 10))
    complaint_type_counts = df_clean['Complaint_Type'].value_counts().head(10)  # Top 10 complaint types
    sns.barplot(x=complaint_type_counts.values, y=complaint_type_counts.index)
    plt.title('Top 10 Complaint Types', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Complaint Type', fontsize=14)
    plt.tight_layout()
    plt.savefig('complaint_types.png')
    plt.close()
    
    print("\nComplaint type findings:")
    print(f"- Most common complaint: {complaint_type_counts.index[0]} ({complaint_type_counts.values[0]} instances)")
    print(f"- Top 5 most common complaints:")
    for i, (complaint, count) in enumerate(complaint_type_counts.head(5).items()):
        print(f"  {i+1}. {complaint}: {count} instances ({count/complaint_type_counts.sum()*100:.1f}%)")

# 4.5 Relationship Analysis
print("\n4.5 Relationship Analysis")

# Age vs Satisfaction
plt.figure(figsize=(12, 8))
sns.boxplot(x='Overall_Satisfaction', y='Age', data=df_clean)
plt.title('Age Distribution by Satisfaction Level', fontsize=16)
plt.xlabel('Satisfaction Level', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.tight_layout()
plt.savefig('age_vs_satisfaction.png')
plt.close()

print("\nAge vs. satisfaction findings:")
satisfaction_levels = df_clean['Overall_Satisfaction'].unique()
for level in satisfaction_levels:
    if pd.notna(level):  # Check if level is not NaN
        avg_age = df_clean[df_clean['Overall_Satisfaction'] == level]['Age'].mean()
        print(f"- Average age for '{level}' passengers: {avg_age:.1f} years")

# Class vs Satisfaction
plt.figure(figsize=(12, 8))
satisfaction_class = pd.crosstab(df_clean['Class'], df_clean['Overall_Satisfaction'], normalize='index') * 100
satisfaction_class.plot(kind='bar', stacked=True)
plt.title('Satisfaction Distribution by Class', fontsize=16)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Percentage', fontsize=14)
plt.legend(title='Satisfaction Level')
plt.tight_layout()
plt.savefig('class_vs_satisfaction.png')
plt.close()

print("\nClass vs. satisfaction findings:")
# Calculate percentage of "Happy" and "Satisfied" for each class
class_satisfaction = {}
for travel_class in df_clean['Class'].unique():
    class_data = df_clean[df_clean['Class'] == travel_class]
    positive = class_data[class_data['Overall_Satisfaction'].isin(['Happy', 'Satisfied'])].shape[0]
    total = class_data.shape[0]
    if total > 0:
        class_satisfaction[travel_class] = (positive / total) * 100

print("- Positive satisfaction percentages by class:")
for travel_class, percentage in class_satisfaction.items():
    print(f"  {travel_class}: {percentage:.1f}%")

best_class = max(class_satisfaction.items(), key=lambda x: x[1])[0]
print(f"- Class with highest satisfaction: {best_class} ({class_satisfaction[best_class]:.1f}%)")

# Flight Duration vs Satisfaction (if Flight_Duration is available and numeric)
if 'Flight_Duration' in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean['Flight_Duration']):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Overall_Satisfaction', y='Flight_Duration', data=df_clean)
    plt.title('Flight Duration by Satisfaction Level', fontsize=16)
    plt.xlabel('Satisfaction Level', fontsize=14)
    plt.ylabel('Flight Duration (hours)', fontsize=14)
    plt.tight_layout()
    plt.savefig('duration_vs_satisfaction.png')
    plt.close()
    
    print("\nFlight duration vs. satisfaction findings:")
    for level in satisfaction_levels:
        if pd.notna(level):  # Check if level is not NaN
            avg_duration = df_clean[df_clean['Overall_Satisfaction'] == level]['Flight_Duration'].mean()
            if not pd.isna(avg_duration):
                print(f"- Average flight duration for '{level}' passengers: {avg_duration:.2f} hours")

# 4.6 Correlation Analysis
print("\n4.6 Correlation Analysis")

# Get only numeric columns
numeric_df = df_clean.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(16, 14))
correlation = numeric_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Variables', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("\nKey correlation findings:")
# If satisfaction score is available
if 'Satisfaction_Score' in numeric_df.columns:
    satisfaction_corr = numeric_df.corr()['Satisfaction_Score'].sort_values(ascending=False)
    print("- Factors most positively correlated with satisfaction:")
    for i, (factor, corr) in enumerate(satisfaction_corr.head(5).items()):
        if factor != 'Satisfaction_Score':
            print(f"  {i+1}. {factor}: {corr:.3f}")
    
    print("- Factors most negatively correlated with satisfaction:")
    for i, (factor, corr) in enumerate(satisfaction_corr.tail(5).items()):
        if factor != 'Satisfaction_Score':
            print(f"  {i+1}. {factor}: {corr:.3f}")

# Other important correlations
print("- Other notable correlations:")
# Find high correlations (absolute value > 0.3) excluding self-correlations
high_correlations = []
for i in range(len(correlation.columns)):
    for j in range(i+1, len(correlation.columns)):
        value = correlation.iloc[i, j]
        if abs(value) > 0.3:
            high_correlations.append((correlation.columns[i], correlation.columns[j], value))

# Sort by absolute correlation value (highest first)
high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

# Print top 5 high correlations
for i, (var1, var2, value) in enumerate(high_correlations[:5]):
    print(f"  {i+1}. {var1} vs {var2}: {value:.3f}")

# 4.7 Additional Analysis - Loyalty and Payment Methods
print("\n4.7 Additional Analysis - Loyalty and Payment Methods")

# Loyalty Membership Distribution
if 'Loyalty_Membership' in df_clean.columns:
    plt.figure(figsize=(10, 6))
    loyalty_counts = df_clean['Loyalty_Membership'].value_counts()
    sns.barplot(x=loyalty_counts.index, y=loyalty_counts.values)
    plt.title('Loyalty Membership Distribution', fontsize=16)
    plt.xlabel('Membership Level', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig('loyalty_distribution.png')
    plt.close()
    
    print("\nLoyalty membership findings:")
    total_members = loyalty_counts.sum()
    for level, count in loyalty_counts.items():
        if pd.notna(level):  # Check if level is not NaN
            percentage = (count / total_members) * 100
            print(f"- {level}: {count} passengers ({percentage:.1f}%)")
    
    # Calculate average satisfaction by loyalty level
    if 'Satisfaction_Score' in df_clean.columns:
        loyalty_satisfaction = df_clean.groupby('Loyalty_Membership')['Satisfaction_Score'].mean()
        print("\n- Average satisfaction score by loyalty level:")
        for level, score in loyalty_satisfaction.items():
            if pd.notna(level) and pd.notna(score):
                print(f"  {level}: {score:.2f}/3")

# Payment Method Distribution
if 'Payment_Method' in df_clean.columns:
    plt.figure(figsize=(12, 8))
    payment_counts = df_clean['Payment_Method'].value_counts()
    sns.barplot(x=payment_counts.values, y=payment_counts.index)
    plt.title('Payment Method Distribution', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Payment Method', fontsize=14)
    plt.tight_layout()
    plt.savefig('payment_methods.png')
    plt.close()
    
    print("\nPayment method findings:")
    total_payments = payment_counts.sum()
    for method, count in payment_counts.items():
        percentage = (count / total_payments) * 100
        print(f"- {method}: {count} bookings ({percentage:.1f}%)")
    print(f"- Most popular payment method: {payment_counts.index[0]} ({payment_counts.values[0]} bookings, {payment_counts.values[0]/total_payments*100:.1f}%)")
    
    
    
# 5. Key Insights & Conclusion
print("\n5. Key Insights & Conclusion")
print("---------------------------")

# Calculate some key metrics
try:
    avg_age = df_clean['Age'].mean()
    satisfaction_perc = df_clean[df_clean['Overall_Satisfaction'].isin(['Happy', 'Satisfied'])].shape[0] / df_clean.shape[0] * 100
    most_common_airline = df_clean['Airline_Name'].value_counts().index[0]
    avg_departure_delay = df_clean['Departure_Delay'].mean()
    complaint_rate = df_clean[df_clean['Complaint_Submitted'] == 'Yes'].shape[0] / df_clean.shape[0] * 100
    
    print(f"Average passenger age: {avg_age:.1f} years")
    print(f"Percentage of satisfied passengers: {satisfaction_perc:.1f}%")
    print(f"Most common airline: {most_common_airline}")
    print(f"Average departure delay: {avg_departure_delay:.1f} minutes")
    print(f"Complaint submission rate: {complaint_rate:.1f}%")
except:
    print("Could not calculate some key metrics due to data issues.")

print("\nTop factors correlated with passenger satisfaction:")
try:
    # Try to find correlations with satisfaction score
    if 'Satisfaction_Score' in numeric_df.columns:
        satisfaction_corr = numeric_df.corr()['Satisfaction_Score'].sort_values(ascending=False)
        for idx, val in satisfaction_corr.items():
            if idx != 'Satisfaction_Score' and not pd.isna(val):
                print(f"- {idx}: {val:.3f}")
except:
    print("Could not calculate satisfaction correlations.")

print("\nEDA completed successfully! All visualizations have been saved.")