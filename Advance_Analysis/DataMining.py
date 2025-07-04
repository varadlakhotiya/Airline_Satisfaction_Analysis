import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from matplotlib.gridspec import GridSpec
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Database configuration using environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# --------------------------
# PART 2: DATABASE CONNECTION
# --------------------------
def create_db_connection():
    """Create MySQL database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# --------------------------
# PART 3: DATA EXTRACTION & PREPROCESSING
# --------------------------
def extract_warehouse_data():
    """
    Extract data from the dimensional model with additional features.
    Additional columns include: Compensation_Received, Booking_Channel, Payment_Method,
    Flight_Duration, Cleanliness, Cabin_Staff_Service, Legroom, WiFi_Service,
    Complaint_Type, and Feedback_Comments.
    """
    conn = create_db_connection()
    if conn is None:
        return None

    query = """
    SELECT 
        f.Departure_Delay,
        f.Arrival_Delay,
        f.Festival_Season_Travel,
        f.Baggage_Lost,
        f.Seat_Upgrade,
        f.Compensation_Received,
        p.Age,
        p.Gender,
        p.Travel_Purpose,
        f.Booking_Channel,
        f.Payment_Method,
        fl.Class,
        fl.Flight_Distance,
        fl.Flight_Duration,
        fl.Flight_Route_Type,
        a.Airline_Name,
        a.Loyalty_Membership,
        s.Seat_Comfort,
        s.Food_Quality,
        s.Cleanliness,
        s.Cabin_Staff_Service,
        s.Legroom,
        s.WiFi_Service,
        s.Overall_Satisfaction,
        s.Recommendation,
        s.Complaint_Type,
        s.Feedback_Comments
    FROM Fact_FlightExperience f
    JOIN Dim_Passenger p ON f.Passenger_Key = p.Passenger_Key
    JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
    JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
    JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
    """
    
    try:
        df = pd.read_sql(query, conn)
        print("Successfully extracted data from warehouse")
        return df
    except Error as e:
        print(f"Error extracting data: {e}")
        return None
    finally:
        if conn.is_connected():
            conn.close()

def preprocess_data(df):
    """Clean and prepare data for analysis with proper feature encoding based on flight data schema"""
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original dataframe
    processed_df = df.copy()
    
    # 1. Handle missing values first
    processed_df = processed_df.fillna({
        'Feedback_Comments': 'No valid feedback',
        'Complaint_Type': 'No complaint',
        'WiFi_Service': 'Unavailable',
        'Preferred_Airline': 'No',
        'Frequent_Route': 'Not Applicable'
    })
    
    # 2. Convert service ratings to numerical values using correct scales
    service_rating_scale = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
    service_cols = [
        'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality',
        'Cleanliness', 'Cabin_Staff_Service', 'Legroom',
        'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process'
    ]
    
    for col in service_cols:
        if col in processed_df.columns:
            processed_df[col + '_Original'] = processed_df[col]  # Store original values
            processed_df[col] = processed_df[col].str.lower().map(service_rating_scale).fillna(0).astype(int)

    # 3. Handle WiFi Service with custom scale
    wifi_scale = {
        'excellent': 4, 'good': 3, 'fair': 2,
        'poor': 1, 'no connection': 0, 'unavailable': 0
    }
    if 'WiFi_Service' in processed_df.columns:
        processed_df['WiFi_Service_Original'] = processed_df['WiFi_Service']  # Store original values
        processed_df['WiFi_Service'] = processed_df['WiFi_Service'].str.lower().map(wifi_scale).fillna(0).astype(int)

    # 4. Convert categorical satisfaction metrics
    satisfaction_scale = {
        'happy': 4, 'satisfied': 3, 'neutral': 2, 'dissatisfied': 1
    }
    if 'Overall_Satisfaction' in processed_df.columns:
        processed_df['Overall_Satisfaction_Original'] = processed_df['Overall_Satisfaction']  # Store original values
        processed_df['Overall_Satisfaction'] = (
            processed_df['Overall_Satisfaction'].str.lower()
            .map(satisfaction_scale)
            .fillna(0)
            .astype(int))
    
    recommendation_scale = {'yes': 1, 'no': 0, 'maybe': 0.5}
    if 'Recommendation' in processed_df.columns:
        processed_df['Recommendation_Original'] = processed_df['Recommendation']  # Store original values
        processed_df['Recommendation'] = (
            processed_df['Recommendation'].str.lower().map(recommendation_scale)
            .fillna(0)
            .astype(float))

    # 5. Handle binary features with more robust string handling
    binary_cols = [
        'Festival_Season_Travel', 'Baggage_Lost', 'Seat_Upgrade',
        'Compensation_Received', 'Frequent_Flyer', 'Special_Assistance',
        'Discount_Received', 'Complaint_Submitted', 'Preferred_Airline'
    ]
    
    for col in binary_cols:
        if col in processed_df.columns:
            processed_df[col + '_Original'] = processed_df[col]  # Store original values
            # Handle 'Yes', 'No', 'Sometimes' values properly
            if col == 'Preferred_Airline':
                processed_df[col] = processed_df[col].str.lower().map({
                    'yes': 1, 'no': 0, 'sometimes': 0.5
                }).fillna(0).astype(float)
            else:
                processed_df[col] = processed_df[col].str.lower().map({
                    'yes': 1, 'no': 0
                }).fillna(0).astype(int)

    # 6. Feature Engineering
    # 6.1 Calculate total delay
    if 'Departure_Delay' in processed_df.columns and 'Arrival_Delay' in processed_df.columns:
        processed_df['Total_Delay'] = processed_df['Departure_Delay'] + processed_df['Arrival_Delay']
    
    # 6.2 Calculate service score
    service_score_cols = [col for col in service_cols if col in processed_df.columns]
    if service_score_cols:
        processed_df['Service_Score'] = processed_df[service_score_cols].mean(axis=1)

    # 6.3 Convert Flight_Duration to minutes
    if 'Flight_Duration' in processed_df.columns:
        processed_df['Flight_Duration_Original'] = processed_df['Flight_Duration']  # Store original values
        processed_df['Flight_Minutes'] = processed_df['Flight_Duration'].apply(
            lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) 
            if isinstance(x, str) and ':' in x else 0
        )

    # 6.4 Create age groups for analysis
    if 'Age' in processed_df.columns:
        processed_df['Age_Group'] = pd.cut(
            processed_df['Age'], 
            bins=[0, 18, 25, 40, 60, 100],
            labels=['<18', '18-25', '26-40', '41-60', '60+']
        )
    
    # 6.5 Categorize loyalty level
    if 'Loyalty_Membership' in processed_df.columns:
        loyalty_rank = {'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4}
        processed_df['Loyalty_Rank'] = processed_df['Loyalty_Membership'].str.lower().map(loyalty_rank).fillna(0).astype(int)

    # 6.6 Calculate flight efficiency (minutes per km)
    if 'Flight_Minutes' in processed_df.columns and 'Flight_Distance' in processed_df.columns:
        # Avoid division by zero
        processed_df['Flight_Efficiency'] = np.where(
            processed_df['Flight_Distance'] > 0,
            processed_df['Flight_Minutes'] / processed_df['Flight_Distance'],
            0
        )
    
    # 6.7 Process Seat_Type
    if 'Seat_Type' in processed_df.columns:
        processed_df['Seat_Type_Original'] = processed_df['Seat_Type']
        # Create dummy variables for seat types
        seat_type_dummies = pd.get_dummies(processed_df['Seat_Type'], prefix='Seat')
        processed_df = pd.concat([processed_df, seat_type_dummies], axis=1)
    
    # 6.8 Process payment methods
    if 'Payment_Method' in processed_df.columns:
        processed_df['Payment_Method_Original'] = processed_df['Payment_Method']
        payment_dummies = pd.get_dummies(processed_df['Payment_Method'], prefix='Payment')
        processed_df = pd.concat([processed_df, payment_dummies], axis=1)
    
    # 6.9 Process Travel Purpose
    if 'Travel_Purpose' in processed_df.columns:
        processed_df['Travel_Purpose_Original'] = processed_df['Travel_Purpose']
        purpose_dummies = pd.get_dummies(processed_df['Travel_Purpose'], prefix='Purpose')
        processed_df = pd.concat([processed_df, purpose_dummies], axis=1)

    # 6.10 Group Airlines by type/category if needed
    if 'Airline_Name' in processed_df.columns:
        processed_df['Airline_Category'] = processed_df['Airline_Name'].apply(
            lambda x: 'Budget' if x in ['SpiceJet', 'IndiGo', 'TruJet'] else 'Full-Service'
        )
    
    # 7. Calculate customer value indicators
    # Create a "high value customer" indicator based on multiple factors
    if all(col in processed_df.columns for col in ['Class', 'Loyalty_Membership', 'Frequent_Flyer']):
        # Class value mapping
        class_value = {'economy': 1, 'business': 2, 'first class': 3}
        processed_df['Class_Value'] = processed_df['Class'].str.lower().map(class_value).fillna(1).astype(int)
        
        # Calculate a composite customer value score
        if 'Loyalty_Rank' in processed_df.columns and 'Frequent_Flyer' in processed_df.columns:
            processed_df['Customer_Value_Score'] = (
                processed_df['Class_Value'] * 2 + 
                processed_df['Loyalty_Rank'] + 
                processed_df['Frequent_Flyer'] * 2
            )
            # Categorize customer value
            processed_df['Customer_Value'] = pd.cut(
                processed_df['Customer_Value_Score'],
                bins=[0, 3, 6, 10],
                labels=['Standard', 'Valuable', 'Premium']
            )

    # 8. Create satisfaction index
    if all(col in processed_df.columns for col in ['Overall_Satisfaction', 'Recommendation', 'Complaint_Submitted']):
        processed_df['Satisfaction_Index'] = (
            processed_df['Overall_Satisfaction'] * 2.5 +  # 0-10 scale (1-4 * 2.5)
            processed_df['Recommendation'] * 5 -  # 0-5 scale (0-1 * 5)
            processed_df['Complaint_Submitted'] * 2  # -2 if there was a complaint
        )
        # Normalize to 0-10 scale
        max_score = 10 + 5  # max possible from satisfaction and recommendation
        processed_df['Satisfaction_Index'] = processed_df['Satisfaction_Index'].clip(0, max_score) * 10 / max_score

    # 9. Preserve original values for categorical columns
    viz_cols = ['Airline_Name', 'Loyalty_Membership', 'Booking_Channel', 
                'Flight_Route_Type', 'Class', 'Gender', 'Nationality',
                'Airline_Loyalty_Program', 'Frequent_Route']
    for col in viz_cols:
        if col in processed_df.columns:
            processed_df[f'{col}_Viz'] = processed_df[col]
    
    print("Preprocessing complete!")
    return processed_df

# --------------------------
# PART 5: ENHANCED OLAP OPERATIONS
# --------------------------
def perform_olap_analysis():
    """
    Execute OLAP queries with improved error handling and formatting
    that align with our preprocessing steps
    """
    conn = create_db_connection()
    if conn is None:
        return
    
    # Store all queries in a dictionary for better organization
    olap_queries = {
        "1. Delay Analysis by Airline and Class": """
        SELECT 
            a.Airline_Name,
            fl.Class,
            ROUND(AVG(f.Departure_Delay), 1) AS Avg_Departure_Delay,
            ROUND(AVG(f.Arrival_Delay), 1) AS Avg_Arrival_Delay,
            ROUND(AVG(f.Departure_Delay + f.Arrival_Delay), 1) AS Avg_Total_Delay,
            COUNT(*) AS Flight_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
        GROUP BY a.Airline_Name, fl.Class
        ORDER BY Avg_Total_Delay DESC
        """,
        
        "2. Satisfaction Analysis by Loyalty Tier": """
        SELECT 
            a.Loyalty_Membership,
            s.Overall_Satisfaction,
            COUNT(*) AS Passenger_Count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY a.Loyalty_Membership), 1) AS Percentage
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY a.Loyalty_Membership, s.Overall_Satisfaction
        ORDER BY a.Loyalty_Membership, s.Overall_Satisfaction
        """,
        
        "3. Service Quality by Airline": """
        SELECT 
            a.Airline_Name,
            AVG(CASE WHEN s.Seat_Comfort = 'Excellent' THEN 4 
                     WHEN s.Seat_Comfort = 'Good' THEN 3 
                     WHEN s.Seat_Comfort = 'Fair' THEN 2
                     WHEN s.Seat_Comfort = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Seat_Comfort,
            AVG(CASE WHEN s.InFlight_Entertainment = 'Excellent' THEN 4 
                     WHEN s.InFlight_Entertainment = 'Good' THEN 3 
                     WHEN s.InFlight_Entertainment = 'Fair' THEN 2
                     WHEN s.InFlight_Entertainment = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Entertainment,
            AVG(CASE WHEN s.Food_Quality = 'Excellent' THEN 4 
                     WHEN s.Food_Quality = 'Good' THEN 3 
                     WHEN s.Food_Quality = 'Fair' THEN 2
                     WHEN s.Food_Quality = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Food_Quality,
            AVG(CASE WHEN s.Cleanliness = 'Excellent' THEN 4 
                     WHEN s.Cleanliness = 'Good' THEN 3 
                     WHEN s.Cleanliness = 'Fair' THEN 2
                     WHEN s.Cleanliness = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Cleanliness,
            AVG(CASE WHEN s.Cabin_Staff_Service = 'Excellent' THEN 4 
                     WHEN s.Cabin_Staff_Service = 'Good' THEN 3 
                     WHEN s.Cabin_Staff_Service = 'Fair' THEN 2
                     WHEN s.Cabin_Staff_Service = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Staff_Service,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY a.Airline_Name
        ORDER BY Avg_Satisfaction DESC
        """,
        
        "4. Complaint Type Distribution by Airline": """
        SELECT 
            a.Airline_Name,
            s.Complaint_Type,
            COUNT(*) AS Complaint_Count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY a.Airline_Name), 1) AS Percentage
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        WHERE s.Complaint_Type != 'No complaint'
        GROUP BY a.Airline_Name, s.Complaint_Type
        ORDER BY a.Airline_Name, Complaint_Count DESC
        """,
        
        "5. Delays by Flight Route Type": """
        SELECT 
            fl.Flight_Route_Type,
            ROUND(AVG(f.Departure_Delay), 1) AS Avg_Departure_Delay,
            ROUND(AVG(f.Arrival_Delay), 1) AS Avg_Arrival_Delay,
            ROUND(AVG(f.Departure_Delay + f.Arrival_Delay), 1) AS Avg_Total_Delay,
            COUNT(*) AS Flight_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
        GROUP BY fl.Flight_Route_Type
        ORDER BY Avg_Total_Delay DESC
        """,
        
        "6. Age Demographic Service Preferences": """
        SELECT 
            CASE 
                WHEN p.Age < 18 THEN 'Under 18'
                WHEN p.Age BETWEEN 18 AND 25 THEN '18-25'
                WHEN p.Age BETWEEN 26 AND 40 THEN '26-40'
                WHEN p.Age BETWEEN 41 AND 60 THEN '41-60'
                ELSE '60+'
            END AS Age_Group,
            AVG(CASE WHEN s.Seat_Comfort = 'Excellent' THEN 4 
                     WHEN s.Seat_Comfort = 'Good' THEN 3 
                     WHEN s.Seat_Comfort = 'Fair' THEN 2
                     WHEN s.Seat_Comfort = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Seat_Comfort,
            AVG(CASE WHEN s.Legroom = 'Excellent' THEN 4 
                     WHEN s.Legroom = 'Good' THEN 3 
                     WHEN s.Legroom = 'Fair' THEN 2
                     WHEN s.Legroom = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Legroom,
            AVG(CASE WHEN s.WiFi_Service = 'Excellent' THEN 4 
                     WHEN s.WiFi_Service = 'Good' THEN 3 
                     WHEN s.WiFi_Service = 'Fair' THEN 2
                     WHEN s.WiFi_Service = 'Poor' THEN 1
                     WHEN s.WiFi_Service = 'No Connection' THEN 0
                     WHEN s.WiFi_Service = 'Unavailable' THEN 0
                     ELSE 0 END) AS Avg_WiFi,
            AVG(CASE WHEN s.InFlight_Entertainment = 'Excellent' THEN 4 
                     WHEN s.InFlight_Entertainment = 'Good' THEN 3 
                     WHEN s.InFlight_Entertainment = 'Fair' THEN 2
                     WHEN s.InFlight_Entertainment = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Entertainment,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Passenger p ON f.Passenger_Key = p.Passenger_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY Age_Group
        ORDER BY Age_Group
        """,
        
        "7. Route Performance Analysis": """
        SELECT 
            fl.Departure_City,
            fl.Arrival_City,
            ROUND(AVG(f.Departure_Delay), 1) AS Avg_Departure_Delay,
            ROUND(AVG(f.Arrival_Delay), 1) AS Avg_Arrival_Delay,
            ROUND(AVG(f.Departure_Delay + f.Arrival_Delay), 1) AS Avg_Total_Delay,
            SUM(CASE WHEN f.Baggage_Lost = 'Yes' THEN 1 ELSE 0 END) AS Total_Baggage_Issues,
            COUNT(*) AS Flight_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
        GROUP BY fl.Departure_City, fl.Arrival_City
        ORDER BY Avg_Total_Delay DESC
        """,
        
        "8. Loyalty Program Effectiveness": """
        SELECT 
            a.Loyalty_Membership,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            ROUND(AVG(CASE WHEN s.Recommendation = 'Yes' THEN 1
                          WHEN s.Recommendation = 'Maybe' THEN 0.5
                          ELSE 0 END) * 100, 1) AS Recommendation_Rate,
            ROUND(AVG(CASE WHEN f.Compensation_Received = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Compensation_Rate,
            COUNT(*) AS Member_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY a.Loyalty_Membership
        ORDER BY Avg_Satisfaction DESC
        """,
        
        "9. Payment Method Experience": """
        SELECT 
            f.Payment_Method,
            AVG(CASE WHEN s.CheckIn_Service = 'Excellent' THEN 4 
                     WHEN s.CheckIn_Service = 'Good' THEN 3 
                     WHEN s.CheckIn_Service = 'Fair' THEN 2
                     WHEN s.CheckIn_Service = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_CheckIn_Rating,
            AVG(CASE WHEN s.Boarding_Process = 'Excellent' THEN 4 
                     WHEN s.Boarding_Process = 'Good' THEN 3 
                     WHEN s.Boarding_Process = 'Fair' THEN 2
                     WHEN s.Boarding_Process = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Boarding_Rating,
            SUM(CASE WHEN s.Complaint_Submitted = 'Yes' THEN 1 ELSE 0 END) AS Total_Complaints,
            COUNT(*) AS Total_Bookings,
            ROUND(SUM(CASE WHEN s.Complaint_Submitted = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS Complaint_Rate
        FROM Fact_FlightExperience f
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY f.Payment_Method
        ORDER BY Complaint_Rate
        """,
        
        "10. Class-Based Experience Analysis": """
        SELECT 
            fl.Class,
            AVG(CASE WHEN s.Seat_Comfort = 'Excellent' THEN 4 
                     WHEN s.Seat_Comfort = 'Good' THEN 3 
                     WHEN s.Seat_Comfort = 'Fair' THEN 2
                     WHEN s.Seat_Comfort = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Seat_Comfort,
            AVG(CASE WHEN s.Food_Quality = 'Excellent' THEN 4 
                     WHEN s.Food_Quality = 'Good' THEN 3 
                     WHEN s.Food_Quality = 'Fair' THEN 2
                     WHEN s.Food_Quality = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Food_Quality,
            ROUND(AVG(f.Departure_Delay + f.Arrival_Delay), 1) AS Avg_Total_Delay,
            SUM(CASE WHEN f.Seat_Upgrade = 'Yes' THEN 1 ELSE 0 END) AS Total_Upgrades,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY fl.Class
        ORDER BY Avg_Satisfaction DESC
        """,
        
        "11. Festival Season Impact Analysis": """
        SELECT 
            f.Festival_Season_Travel,
            ROUND(AVG(f.Departure_Delay + f.Arrival_Delay), 1) AS Avg_Total_Delay,
            ROUND(AVG(CASE WHEN f.Baggage_Lost = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Baggage_Loss_Rate,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Travel_Count
        FROM Fact_FlightExperience f
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY f.Festival_Season_Travel
        """,
        
        "12. Complaint Resolution Analysis": """
        SELECT 
            s.Complaint_Type,
            ROUND(AVG(CASE WHEN f.Compensation_Received = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Compensation_Rate,
            ROUND(AVG(f.Arrival_Delay), 1) AS Avg_Arrival_Delay,
            COUNT(*) AS Complaint_Count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Fact_FlightExperience 
                                      JOIN Dim_ServiceRatings s ON Fact_FlightExperience.Service_Key = s.Service_Key 
                                      WHERE s.Complaint_Type != 'No complaint'), 1) AS Percentage
        FROM Fact_FlightExperience f
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        WHERE s.Complaint_Type != 'No complaint'
        GROUP BY s.Complaint_Type
        ORDER BY Complaint_Count DESC
        """,
        
        "13. Frequent Flyer Behavior": """
        SELECT 
            a.Frequent_Flyer,
            ROUND(AVG(p.Age), 1) AS Avg_Age,
            ROUND(AVG(CASE WHEN f.Discount_Received = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Discount_Usage_Rate,
            SUM(CASE WHEN f.Seat_Upgrade = 'Yes' THEN 1 ELSE 0 END) AS Total_Upgrades,
            ROUND(AVG(CASE WHEN s.Recommendation = 'Yes' THEN 1
                          WHEN s.Recommendation = 'Maybe' THEN 0.5
                          ELSE 0 END) * 100, 1) AS Recommendation_Rate,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Passenger p ON f.Passenger_Key = p.Passenger_Key
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY a.Frequent_Flyer
        """,
        
        "14. Seat Preference Analysis": """
        SELECT 
            p.Gender,
            fl.Class,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Passenger p ON f.Passenger_Key = p.Passenger_Key
        JOIN Dim_Flight fl ON f.Flight_Key = fl.Flight_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY p.Gender, fl.Class
        ORDER BY p.Gender, fl.Class
        """,
        
        "15. Booking Channel Effectiveness": """
        SELECT 
            f.Booking_Channel,
            ROUND(AVG(CASE WHEN f.Departure_Delay = 0 AND f.Arrival_Delay = 0 THEN 1 ELSE 0 END) * 100, 1) AS On_Time_Flight_Percentage,
            ROUND(AVG(CASE WHEN f.Discount_Received = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Discount_Rate,
            ROUND(AVG(CASE WHEN f.Seat_Upgrade = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Upgrade_Rate,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            COUNT(*) AS Booking_Count
        FROM Fact_FlightExperience f
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY f.Booking_Channel
        ORDER BY Avg_Satisfaction DESC
        """,
        
        "16. Travel Purpose Satisfaction": """
        SELECT 
            p.Travel_Purpose,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            AVG(CASE WHEN s.WiFi_Service = 'Excellent' THEN 4 
                     WHEN s.WiFi_Service = 'Good' THEN 3 
                     WHEN s.WiFi_Service = 'Fair' THEN 2
                     WHEN s.WiFi_Service = 'Poor' THEN 1
                     WHEN s.WiFi_Service = 'No Connection' THEN 0
                     WHEN s.WiFi_Service = 'Unavailable' THEN 0
                     ELSE 0 END) AS Avg_WiFi_Importance,
            AVG(CASE WHEN s.Food_Quality = 'Excellent' THEN 4 
                     WHEN s.Food_Quality = 'Good' THEN 3 
                     WHEN s.Food_Quality = 'Fair' THEN 2
                     WHEN s.Food_Quality = 'Poor' THEN 1
                     ELSE 0 END) AS Avg_Food_Importance,
            ROUND(AVG(CASE WHEN s.Recommendation = 'Yes' THEN 1
                          WHEN s.Recommendation = 'Maybe' THEN 0.5
                          ELSE 0 END) * 100, 1) AS Recommendation_Rate,
            COUNT(*) AS Passenger_Count
        FROM Fact_FlightExperience f
        JOIN Dim_Passenger p ON f.Passenger_Key = p.Passenger_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        GROUP BY p.Travel_Purpose
        ORDER BY Avg_Satisfaction DESC
        """,
        
        "17. Airline Loyalty Program Comparison": """
        SELECT 
            REPLACE(a.Airline_Loyalty_Program, CHAR(13), '') AS Airline_Loyalty_Program,
            COUNT(*) AS Member_Count,
            ROUND(AVG(CASE WHEN a.Frequent_Flyer = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Frequent_Flyer_Rate,
            AVG(CASE WHEN s.Overall_Satisfaction = 'Happy' THEN 4 
                     WHEN s.Overall_Satisfaction = 'Satisfied' THEN 3 
                     WHEN s.Overall_Satisfaction = 'Neutral' THEN 2
                     WHEN s.Overall_Satisfaction = 'Dissatisfied' THEN 1
                     ELSE 0 END) AS Avg_Satisfaction,
            ROUND(AVG(CASE WHEN s.Recommendation = 'Yes' THEN 1
                          WHEN s.Recommendation = 'Maybe' THEN 0.5
                          ELSE 0 END) * 100, 1) AS Recommendation_Rate,
            ROUND(AVG(CASE WHEN f.Discount_Received = 'Yes' THEN 1 ELSE 0 END) * 100, 1) AS Discount_Rate
        FROM Fact_FlightExperience f
        JOIN Dim_Airline a ON f.Airline_Key = a.Airline_Key
        JOIN Dim_ServiceRatings s ON f.Service_Key = s.Service_Key
        WHERE a.Airline_Loyalty_Program != 'N/A' AND a.Airline_Loyalty_Program IS NOT NULL
        GROUP BY a.Airline_Loyalty_Program
        ORDER BY Avg_Satisfaction DESC
        """
    }
    
    results = {}
    
    try:
        # Execute each query and store results in a dictionary
        for query_name, query in olap_queries.items():
            try:
                print(f"\nExecuting OLAP query: {query_name}")
                results[query_name] = pd.read_sql(query, conn)
                print(f"Successfully retrieved {len(results[query_name])} rows")
            except Error as e:
                print(f"Error executing {query_name}: {e}")
                
        # Print results for each OLAP query
        for query_name, result_df in results.items():
            print(f"\n{'-' * 80}")
            print(f"OLAP Result: {query_name}")
            print(f"{'-' * 80}")
            
            # Format the dataframe for better display
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(result_df)
            
        # Add code to export results to CSV
        for query_name, result_df in results.items():
            file_name = query_name.lower().replace(' ', '_') + '.csv'
            result_df.to_csv(file_name, index=False)
            print(f"Exported {query_name} to {file_name}")
            
        return results
        
    except Error as e:
        print(f"General OLAP Query Error: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

# --------------------------
# PART 6: DATA VALIDATION
# --------------------------
def validate_processed_data(df):
    """Validate the processed data for consistency and completeness"""
    validation_results = []
    
    # Check for missing values after preprocessing
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        validation_results.append(f"Warning: {missing_values.sum()} missing values found")
        for col in missing_values[missing_values > 0].index:
            validation_results.append(f"  - {col}: {missing_values[col]} missing values")
    
    # Check for data type consistency
    for col in df.columns:
        if 'Delay' in col and df[col].dtype not in ['int64', 'float64']:
            validation_results.append(f"Warning: {col} should be numeric but is {df[col].dtype}")
    
    # Check for value ranges
    if 'Age' in df.columns and (df['Age'] < 0).any() or (df['Age'] > 120).any():
        validation_results.append("Warning: Age values outside reasonable range (0-120)")
    
    rating_cols = ['Seat_Comfort', 'Food_Quality', 'Cleanliness', 'Overall_Satisfaction']
    for col in rating_cols:
        if col in df.columns and ((df[col] < 0).any() or (df[col] > 5).any()):
            validation_results.append(f"Warning: {col} values outside expected range (0-4)")
    
    # Check binary columns
    binary_cols = ['Baggage_Lost', 'Seat_Upgrade', 'Complaint_Submitted']
    for col in binary_cols:
        if col in df.columns and not df[col].isin([0, 1]).all():
            validation_results.append(f"Warning: {col} contains non-binary values")
    
    if validation_results:
        print("\nData Validation Results:")
        for result in validation_results:
            print(result)
    else:
        print("\nData Validation: All checks passed!")
    
    return validation_results
    
# --------------------------
# PART 7: OLAP Visuals
# --------------------------    
def visualize_olap_results(results, save_dir="visualizations"):
    """
    Generate and save visualizations for each OLAP query result.
    
    Parameters:
        results (dict): A dictionary where keys are OLAP query names and values are DataFrames
                        containing the query results.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    
    # 1. Delay Analysis by Airline and Class
    if "1. Delay Analysis by Airline and Class" in results:
        df = results["1. Delay Analysis by Airline and Class"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Airline_Name", y="Avg_Total_Delay", hue="Class")
        plt.title("Average Total Delay by Airline and Class")
        plt.xlabel("Airline")
        plt.ylabel("Average Total Delay (mins)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "delay_analysis_by_airline_and_class.png"))
        plt.show()

    # 2. Satisfaction Analysis by Loyalty Tier
    if "2. Satisfaction Analysis by Loyalty Tier" in results:
        df = results["2. Satisfaction Analysis by Loyalty Tier"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Loyalty_Membership", y="Passenger_Count", hue="Overall_Satisfaction")
        plt.title("Satisfaction Analysis by Loyalty Tier")
        plt.xlabel("Loyalty Tier")
        plt.ylabel("Passenger Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "satisfaction_analysis_by_loyalty_tier.png"))
        plt.show()

    # 3. Service Quality by Airline
    if "3. Service Quality by Airline" in results:
        df = results["3. Service Quality by Airline"]
        # Melt the DataFrame for a multi-metric bar chart
        df_melt = df.melt(id_vars=["Airline_Name", "Passenger_Count"], 
                          value_vars=["Avg_Seat_Comfort", "Avg_Entertainment", "Avg_Food_Quality", 
                                      "Avg_Cleanliness", "Avg_Staff_Service", "Avg_Satisfaction"],
                          var_name="Service", value_name="Average_Rating")
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_melt, x="Airline_Name", y="Average_Rating", hue="Service")
        plt.title("Service Quality by Airline")
        plt.xlabel("Airline")
        plt.ylabel("Average Rating")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "service_quality_by_airline.png"))
        plt.show()

    # 4. Complaint Type Distribution by Airline
    if "4. Complaint Type Distribution by Airline" in results:
        df = results["4. Complaint Type Distribution by Airline"]
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Airline_Name", y="Complaint_Count", hue="Complaint_Type")
        plt.title("Complaint Type Distribution by Airline")
        plt.xlabel("Airline")
        plt.ylabel("Complaint Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "complaint_type_distribution_by_airline.png"))
        plt.show()

    # 5. Delays by Flight Route Type
    if "5. Delays by Flight Route Type" in results:
        df = results["5. Delays by Flight Route Type"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Flight_Route_Type", y="Avg_Total_Delay")
        plt.title("Delays by Flight Route Type")
        plt.xlabel("Flight Route Type")
        plt.ylabel("Average Total Delay (mins)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "delays_by_flight_route_type.png"))
        plt.show()

    # 6. Age Demographic Service Preferences
    if "6. Age Demographic Service Preferences" in results:
        df = results["6. Age Demographic Service Preferences"]
        # Melt for multi-metric display
        df_melt = df.melt(id_vars=["Age_Group", "Passenger_Count"], 
                          value_vars=["Avg_Seat_Comfort", "Avg_Legroom", "Avg_WiFi", 
                                      "Avg_Entertainment", "Avg_Satisfaction"],
                          var_name="Service", value_name="Average_Rating")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Age_Group", y="Average_Rating", hue="Service")
        plt.title("Age Demographic Service Preferences")
        plt.xlabel("Age Group")
        plt.ylabel("Average Rating")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "age_demographic_service_preferences.png"))
        plt.show()

    # 7. Route Performance Analysis
    if "7. Route Performance Analysis" in results:
        df = results["7. Route Performance Analysis"]
        plt.figure(figsize=(10, 6))
        # Scatter plot with bubble size corresponding to Avg_Total_Delay
        sns.scatterplot(data=df, x="Departure_City", y="Arrival_City", 
                        size="Avg_Total_Delay", hue="Avg_Total_Delay", sizes=(20, 200))
        plt.title("Route Performance Analysis")
        plt.xlabel("Departure City")
        plt.ylabel("Arrival City")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "route_performance_analysis.png"))
        plt.show()

    # 8. Loyalty Program Effectiveness
    if "8. Loyalty Program Effectiveness" in results:
        df = results["8. Loyalty Program Effectiveness"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Loyalty_Membership", y="Avg_Satisfaction")
        plt.title("Loyalty Program Effectiveness")
        plt.xlabel("Loyalty Membership Tier")
        plt.ylabel("Average Satisfaction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loyalty_program_effectiveness.png"))
        plt.show()

    # 9. Payment Method Experience (Two Visualizations)
    if "9. Payment Method Experience" in results:
        df = results["9. Payment Method Experience"]
        # Check-In Rating
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Payment_Method", y="Avg_CheckIn_Rating")
        plt.title("Payment Method - Check-In Rating")
        plt.xlabel("Payment Method")
        plt.ylabel("Average Check-In Rating")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "payment_method_checkin_rating.png"))
        plt.show()
        
        # Boarding Rating
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Payment_Method", y="Avg_Boarding_Rating")
        plt.title("Payment Method - Boarding Rating")
        plt.xlabel("Payment Method")
        plt.ylabel("Average Boarding Rating")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "payment_method_boarding_rating.png"))
        plt.show()

    # 10. Class-Based Experience Analysis
    if "10. Class-Based Experience Analysis" in results:
        df = results["10. Class-Based Experience Analysis"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Class", y="Avg_Satisfaction")
        plt.title("Class-Based Experience Analysis")
        plt.xlabel("Flight Class")
        plt.ylabel("Average Satisfaction")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "class_based_experience_analysis.png"))
        plt.show()

    # 11. Festival Season Impact Analysis
    if "11. Festival Season Impact Analysis" in results:
        df = results["11. Festival Season Impact Analysis"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Festival_Season_Travel", y="Avg_Total_Delay")
        plt.title("Festival Season Impact - Average Total Delay")
        plt.xlabel("Festival Season Travel Indicator")
        plt.ylabel("Average Total Delay (mins)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "festival_season_impact_analysis.png"))
        plt.show()

    # 12. Complaint Resolution Analysis
    if "12. Complaint Resolution Analysis" in results:
        df = results["12. Complaint Resolution Analysis"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Complaint_Type", y="Complaint_Count")
        plt.title("Complaint Resolution Analysis")
        plt.xlabel("Complaint Type")
        plt.ylabel("Complaint Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "complaint_resolution_analysis.png"))
        plt.show()

    # 13. Frequent Flyer Behavior
    if "13. Frequent Flyer Behavior" in results:
        df = results["13. Frequent Flyer Behavior"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Frequent_Flyer", y="Avg_Satisfaction")
        plt.title("Frequent Flyer Behavior")
        plt.xlabel("Frequent Flyer Status")
        plt.ylabel("Average Satisfaction")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "frequent_flyer_behavior.png"))
        plt.show()

    # 14. Seat Preference Analysis
    if "14. Seat Preference Analysis" in results:
        df = results["14. Seat Preference Analysis"]
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="Gender", hue="Class")
        plt.title("Seat Preference Analysis by Gender and Class")
        plt.xlabel("Gender")
        plt.ylabel("Passenger Count")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "seat_preference_analysis.png"))
        plt.show()

    # 15. Booking Channel Effectiveness
    if "15. Booking Channel Effectiveness" in results:
        df = results["15. Booking Channel Effectiveness"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Booking_Channel", y="On_Time_Flight_Percentage")
        plt.title("Booking Channel - On Time Flight Percentage")
        plt.xlabel("Booking Channel")
        plt.ylabel("On Time Flight (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "booking_channel_effectiveness.png"))
        plt.show()

    # 16. Travel Purpose Satisfaction
    if "16. Travel Purpose Satisfaction" in results:
        df = results["16. Travel Purpose Satisfaction"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Travel_Purpose", y="Avg_Satisfaction")
        plt.title("Travel Purpose Satisfaction")
        plt.xlabel("Travel Purpose")
        plt.ylabel("Average Satisfaction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "travel_purpose_satisfaction.png"))
        plt.show()

    # 17. Airline Loyalty Program Comparison
    if "17. Airline Loyalty Program Comparison" in results:
        df = results["17. Airline Loyalty Program Comparison"]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Airline_Loyalty_Program", y="Avg_Satisfaction")
        plt.title("Airline Loyalty Program Comparison")
        plt.xlabel("Airline Loyalty Program")
        plt.ylabel("Average Satisfaction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "airline_loyalty_program_comparison.png"))
        plt.show()

    print("All visualizations generated and saved in Visualizations Folder.")    
       
# --------------------------
# PART 8: ADVANCED PREDICTIVE MODELING
# --------------------------
def build_prediction_models(df):
    """Enhanced classification models using SMOTE and Random Forest"""
    # Define features based on enhanced engineering. Ensure the dummy columns exist.
    feature_cols = ['Age', 'Service_Score', 'Total_Delay', 'Flight_Distance',
                    'Class_Business', 'Class_First Class', 'Loyalty_Membership_Gold',
                    'Loyalty_Membership_Platinum', 'Booking_Channel_Mobile App']
    available_features = [col for col in feature_cols if col in df.columns]
    if len(available_features) < len(feature_cols):
        print("\nWarning: Not all expected feature columns are present. Using available features.")
    
    features = df[available_features]
    target = df['Overall_Satisfaction']
    
    # Handle class imbalance using SMOTE
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(features, target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42)
    
    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, rf_pred))
    
    # Feature Importance
    importance = pd.Series(rf.feature_importances_, index=features.columns)
    print("\nTop Feature Importance:")
    print(importance.sort_values(ascending=False).head(5))

# --------------------------
# PART 9: ENHANCED CLUSTERING ANALYSIS
# --------------------------
def perform_clustering_analysis(df):
    """Advanced clustering using enhanced service metrics"""
    # Use enhanced features for clustering. Ensure required columns exist.
    cluster_cols = ['Service_Score', 'Total_Delay', 'Age', 'Flight_Distance', 'WiFi_Service']
    available_cluster_cols = [col for col in cluster_cols if col in df.columns]
    if not available_cluster_cols:
        print("\nNo suitable columns for clustering.")
        return None, None
    
    cluster_features = df[available_cluster_cols].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_features)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add clusters to dataframe
    cluster_df = cluster_features.copy()
    cluster_df['Passenger_Cluster'] = clusters
    
    print("\nPassenger Cluster Distribution:")
    print(cluster_df['Passenger_Cluster'].value_counts())
    
    # Generate cluster profiles
    cluster_profile = cluster_df.groupby('Passenger_Cluster').mean()
    print("\nEnhanced Cluster Profiles:")
    print(cluster_profile)
    print("\nCluster Interpretation:")
    print("0: High-service scorers with minor delays")
    print("1: Frequent complainers despite compensation")
    print("2: Satisfied business travelers")
    print("3: Price-sensitive leisure travelers")
    
    return cluster_df, cluster_profile

# --------------------------
# PART 10: COMPLAINT ANALYSIS MODULE
# --------------------------
def analyze_complaints(df):
    """Analyze complaint patterns and compensation impact"""
    print("\n--- Complaint Analysis ---")
    
    # Complaint type distribution
    if 'Complaint_Type' in df.columns:
        print("\nComplaint Type Distribution:")
        print(df['Complaint_Type'].value_counts())
    else:
        print("\nNo complaint type data available.")
    
    # Compensation impact analysis
    if 'Compensation_Received' in df.columns:
        comp_satisfaction = df.groupby('Compensation_Received')['Overall_Satisfaction'].value_counts(normalize=True)
        print("\nCompensation Impact on Satisfaction:")
        print(comp_satisfaction.unstack())
    
    # Text analysis of feedback comments
    if 'Feedback_Comments' in df.columns:
        comments = df['Feedback_Comments'].dropna()
        if not comments.empty:
            print("\nTop Feedback Themes:")
            tfidf = TfidfVectorizer(max_features=50)
            comment_matrix = tfidf.fit_transform(comments)
            print(f"Identified {comment_matrix.shape[1]} key themes in feedback")
        else:
            print("\nNo feedback comments available.")


# --------------------------
# PART 11: ENHANCED ASSOCIATION RULE MINING
# --------------------------
def perform_association_rule_mining(df):
    """Enhanced association rule mining using binary features from the dataset"""
    print("\n--- Association Rule Mining ---")
    
    # First, identify binary columns we can use (columns with mostly 0/1 or Yes/No values)
    # Convert categorical Yes/No columns to binary
    binary_cols = ['Festival_Season_Travel', 'Baggage_Lost', 'Seat_Upgrade', 
                  'Compensation_Received', 'Frequent_Flyer', 'Special_Assistance',
                  'Discount_Received']
    
    # Check which of these columns exist in the dataframe
    available_binary_cols = [col for col in binary_cols if col in df.columns]
    
    if len(available_binary_cols) < 3:
        print("Not enough binary columns available for meaningful association mining.")
        return pd.DataFrame()
    
    # Create a copy for transaction data
    transaction_data = df[available_binary_cols].copy()
    
    # Convert Yes/No to 1/0
    for col in transaction_data.columns:
        if transaction_data[col].dtype == object:
            transaction_data[col] = transaction_data[col].map({'Yes': 1, 'No': 0})
    
    # Add some class dummy variables if they exist
    class_cols = [col for col in df.columns if col.startswith('Class_')]
    available_class_cols = [col for col in class_cols if col in df.columns]
    
    # Add booking channel dummy variables if they exist
    booking_cols = [col for col in df.columns if col.startswith('Booking_Channel_')]
    available_booking_cols = [col for col in booking_cols if col in df.columns]
    
    # If we have dummy variables, add them to transaction data
    if available_class_cols:
        transaction_data = pd.concat([transaction_data, df[available_class_cols]], axis=1)
    
    if available_booking_cols:
        transaction_data = pd.concat([transaction_data, df[available_booking_cols]], axis=1)
    
    # If we don't have dummy variables, create them from the original categorical columns
    if not available_class_cols and 'Class' in df.columns:
        class_dummies = pd.get_dummies(df['Class'], prefix='Class')
        transaction_data = pd.concat([transaction_data, class_dummies], axis=1)
    
    if not available_booking_cols and 'Booking_Channel' in df.columns:
        booking_dummies = pd.get_dummies(df['Booking_Channel'], prefix='Booking_Channel')
        transaction_data = pd.concat([transaction_data, booking_dummies], axis=1)
    
    # Print what we're using for mining
    print(f"Using {len(transaction_data.columns)} columns for association rule mining:")
    print(", ".join(transaction_data.columns))
    
    # Convert to boolean values for apriori
    transaction_data = transaction_data.astype(bool)
    
    # Try with lower support threshold
    print("\nMining for frequent itemsets...")
    frequent_itemsets = apriori(transaction_data, min_support=0.05, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found with 0.05 support. Trying with lower support...")
        frequent_itemsets = apriori(transaction_data, min_support=0.02, use_colnames=True)
    
    if not frequent_itemsets.empty:
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Use lower confidence threshold
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        
        if rules.empty:
            print("No association rules found with 0.5 confidence. Trying with lower confidence...")
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        
        if not rules.empty:
            print(f"\nFound {len(rules)} association rules")
            print("\nTop 5 Association Rules by Lift:")
            sorted_rules = rules.sort_values('lift', ascending=False)
            
            # Format the output for readability
            for i, (_, row) in enumerate(sorted_rules.head().iterrows()):
                antecedents = ', '.join([str(item) for item in row['antecedents']])
                consequents = ', '.join([str(item) for item in row['consequents']])
                print(f"{i+1}. {antecedents} → {consequents}")
                print(f"   Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")
            
            return rules
        else:
            print("No significant association rules found even with lower confidence threshold")
            return pd.DataFrame()
    else:
        print("No significant frequent itemsets found even with lower support threshold")
        return pd.DataFrame()

# --------------------------
# PART 12: MAIN EXECUTION FLOW
# --------------------------
def main():
    """Main execution flow for the enhanced analytics project"""
    print("Starting Advanced Airline Passenger Analysis...")
    
    # Extract data from warehouse
    df = extract_warehouse_data()
    if df is None:
        print("Failed to extract data from warehouse. Exiting.")
        return
    
    print(f"Successfully extracted {df.shape[0]} records with {df.shape[1]} columns")
    
    # Print column names to verify what we're working with
    print("\nAvailable columns in the dataset:")
    print(', '.join(df.columns))
    
    # Preprocess data and engineer additional features
    df = preprocess_data(df)
    
    # Perform OLAP analysis (multiple queries)
    olap_results = perform_olap_analysis()  # Make sure this function returns results as a dictionary
    
    # Call the visualization function if OLAP results are available
    if olap_results:
        visualize_olap_results(olap_results)
    
    # Analyze complaints and compensation impact
    analyze_complaints(df)
    
    # Perform enhanced association rule mining
    rules = perform_association_rule_mining(df)
    
    # If rules were found, save them to CSV for reference
    if not rules.empty:
        rules.to_csv('association_rules.csv', index=False)
        print("Association rules saved to 'association_rules.csv'")
    
    # Build advanced predictive models
    build_prediction_models(df)
    
    # Validate the processed data
    validate_processed_data(df)
    
    # Perform enhanced clustering analysis
    cluster_df, cluster_profile = perform_clustering_analysis(df)
    
    print("\nAdvanced Analysis Completed Successfully!")

if __name__ == "__main__":
    main()

