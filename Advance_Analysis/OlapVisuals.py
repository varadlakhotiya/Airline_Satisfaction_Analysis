# app.py (Flask Backend)
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
import urllib.parse

app = Flask(__name__)
CORS(app)

# Database connection setup
username = "root"
password = "Varad@2004"
encoded_password = urllib.parse.quote_plus(password)
host = "localhost"
database = "DataWarehouse"
engine = create_engine(f"mysql+mysqlconnector://{username}:{encoded_password}@{host}/{database}")

@app.route('/')
def index():
    return render_template('index.html')

# ------------------------------
# OLAP Endpoints (API Routes)
# ------------------------------

# 1. Roll-Up / Drill-Down Operation
@app.route('/rollup')
def rollup():
    query = """
    SELECT 
        a.Airline_Name, 
        f.Class,
        AVG(ffe.Departure_Delay) AS Avg_Departure_Delay,
        AVG(ffe.Arrival_Delay) AS Avg_Arrival_Delay
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Dim_Flight f ON ffe.Flight_Key = f.Flight_Key
    GROUP BY a.Airline_Name, f.Class WITH ROLLUP;
    """
    df = pd.read_sql(query, engine)
    # Drop subtotal rows (NULL values)
    df = df.dropna(subset=["Airline_Name", "Class"])
    return df.to_json(orient='records')

# 2. Slice Operation
@app.route('/slice')
def slice():
    query = """
    SELECT 
        a.Airline_Name, 
        AVG(ffe.Departure_Delay) AS Avg_Departure_Delay
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    WHERE a.Airline_Name = 'Air India'
    GROUP BY a.Airline_Name;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 3. Dice Operation
@app.route('/dice')
def dice():
    query = """
    SELECT 
        a.Airline_Name, 
        f.Class,
        COUNT(*) AS FlightCount
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Dim_Flight f ON ffe.Flight_Key = f.Flight_Key
    WHERE a.Airline_Name IN ('Air India', 'SpiceJet')
      AND f.Class = 'Economy'
    GROUP BY a.Airline_Name, f.Class;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 4. Pivot Operation using Pandas
@app.route('/pivot')
def pivot():
    query = """
    SELECT 
        a.Airline_Name, 
        f.Class,
        ffe.Departure_Delay
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Dim_Flight f ON ffe.Flight_Key = f.Flight_Key;
    """
    df_source = pd.read_sql(query, engine)
    pivot_table = pd.pivot_table(
        df_source,
        index="Airline_Name",
        columns="Class",
        values="Departure_Delay",
        aggfunc='mean'
    )
    # Resetting the index for a cleaner JSON structure
    pivot_df = pivot_table.reset_index()
    return pivot_df.to_json(orient='records')

# 5. Overall Satisfaction Distribution
@app.route('/satisfaction')
def satisfaction():
    query = """
    SELECT 
        s.Overall_Satisfaction,
        COUNT(*) AS Count
    FROM Fact_FlightExperience ffe
    JOIN Dim_ServiceRatings s ON ffe.Service_Key = s.Service_Key
    GROUP BY s.Overall_Satisfaction;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 6. Payment Method Usage
@app.route('/payment')
def payment():
    query = """
    SELECT 
        ffe.Payment_Method,
        COUNT(*) AS Count
    FROM Fact_FlightExperience ffe
    GROUP BY ffe.Payment_Method;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 7. Average Flight Distance by Departure City and Airline
@app.route('/distance')
def distance():
    query = """
    SELECT 
        ffd.Departure_City,
        a.Airline_Name,
        AVG(ffd.Flight_Distance) AS Avg_Flight_Distance
    FROM Fact_FlightExperience ffe
    JOIN Dim_Flight ffd ON ffe.Flight_Key = ffd.Flight_Key
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    GROUP BY ffd.Departure_City, a.Airline_Name;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 8. Count of Flights by Gender per Airline
@app.route('/gender')
def gender():
    query = """
    SELECT 
        a.Airline_Name,
        sfd.Gender,
        COUNT(*) AS FlightCount
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Staging_FlightData sfd ON ffe.Passenger_Key = sfd.Passenger_ID
    GROUP BY a.Airline_Name, sfd.Gender;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 9. Distribution of Travel Purpose by Nationality
@app.route('/travel_purpose')
def travel_purpose():
    query = """
    SELECT 
        Nationality,
        Travel_Purpose,
        COUNT(*) AS Count
    FROM Staging_FlightData
    GROUP BY Nationality, Travel_Purpose;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 10. Average Age by Travel Purpose
@app.route('/age_travel')
def age_travel():
    query = """
    SELECT 
        Travel_Purpose,
        AVG(Age) AS Avg_Age
    FROM Staging_FlightData
    GROUP BY Travel_Purpose;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 11. Delay Analysis: Flights with Departure Delay > 15 Minutes
@app.route('/delay_analysis')
def delay_analysis():
    query = """
    SELECT 
        a.Airline_Name,
        COUNT(*) AS Total_Flights,
        SUM(CASE WHEN ffe.Departure_Delay > 15 THEN 1 ELSE 0 END) AS Delayed_Flights,
        ROUND(100 * SUM(CASE WHEN ffe.Departure_Delay > 15 THEN 1 ELSE 0 END) / COUNT(*), 2) AS Delay_Percentage
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    GROUP BY a.Airline_Name;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 12. Complaint Analysis: Distribution of Complaint Types by Airline
@app.route('/complaints')
def complaints():
    query = """
    SELECT 
        a.Airline_Name,
        s.Complaint_Type,
        COUNT(*) AS Complaint_Count
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Dim_ServiceRatings s ON ffe.Service_Key = s.Service_Key
    WHERE s.Complaint_Submitted = 'Yes'
    GROUP BY a.Airline_Name, s.Complaint_Type;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 13. Booking Channel Usage
@app.route('/booking')
def booking():
    query = """
    SELECT 
        ffe.Booking_Channel,
        COUNT(*) AS FlightCount
    FROM Fact_FlightExperience ffe
    GROUP BY ffe.Booking_Channel;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 14. Seat Type Distribution by Airline
@app.route('/seat_distribution')
def seat_distribution():
    query = """
    SELECT 
        a.Airline_Name,
        sfd.Seat_Type,
        COUNT(*) AS SeatCount
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    JOIN Staging_FlightData sfd ON ffe.Passenger_Key = sfd.Passenger_ID
    GROUP BY a.Airline_Name, sfd.Seat_Type;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 15. Frequent Flyer Analysis: Count of Frequent Flyer Flights by Airline
@app.route('/frequent_flyer')
def frequent_flyer():
    query = """
    SELECT 
        a.Airline_Name,
        COUNT(*) AS FrequentFlyerCount
    FROM Fact_FlightExperience ffe
    JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
    WHERE a.Frequent_Flyer = 'Yes'
    GROUP BY a.Airline_Name;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 16. Customer Segmentation by Age & Travel Purpose
@app.route('/customer_segmentation')
def customer_segmentation():
    query = """
    SELECT 
        Travel_Purpose,
        CASE 
            WHEN Age < 25 THEN 'Under 25'
            WHEN Age BETWEEN 25 AND 40 THEN '25-40'
            WHEN Age BETWEEN 41 AND 60 THEN '41-60'
            ELSE '60+'
        END AS Age_Group,
        COUNT(*) AS Passenger_Count,
        ROUND(100 * AVG(CASE WHEN Overall_Satisfaction IN ('Happy', 'Satisfied') THEN 1 ELSE 0 END), 2) AS Positive_Satisfaction_Percentage
    FROM Staging_FlightData
    GROUP BY Travel_Purpose, Age_Group;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 17. On-Time Performance by Flight Route Type
@app.route('/on_time')
def on_time():
    query = """
    SELECT 
        Flight_Route_Type,
        COUNT(*) AS Total_Flights,
        SUM(CASE WHEN Departure_Delay <= 0 AND Arrival_Delay <= 0 THEN 1 ELSE 0 END) AS On_Time_Flights,
        ROUND(100 * SUM(CASE WHEN Departure_Delay <= 0 AND Arrival_Delay <= 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS On_Time_Percentage
    FROM Staging_FlightData
    GROUP BY Flight_Route_Type;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 18. Loyalty Impact Analysis
@app.route('/loyalty')
def loyalty():
    query = """
    SELECT 
        Loyalty_Membership,
        COUNT(*) AS Total_Passengers,
        ROUND(100 * AVG(CASE WHEN Overall_Satisfaction IN ('Happy', 'Satisfied') THEN 1 ELSE 0 END), 2) AS Positive_Satisfaction_Percentage
    FROM Staging_FlightData
    GROUP BY Loyalty_Membership;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

# 19. Class Performance Analysis: Delays and Satisfaction by Flight Class
@app.route('/class_performance')
def class_performance():
    query = """
    SELECT 
        Class,
        AVG(Departure_Delay) AS Avg_Departure_Delay,
        AVG(Arrival_Delay) AS Avg_Arrival_Delay,
        ROUND(100 * AVG(CASE WHEN Overall_Satisfaction IN ('Happy', 'Satisfied') THEN 1 ELSE 0 END), 2) AS Positive_Satisfaction_Percentage
    FROM Staging_FlightData
    GROUP BY Class;
    """
    df = pd.read_sql(query, engine)
    return df.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
