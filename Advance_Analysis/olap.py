from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse

# -------------------------------
# 1. Establish Database Connection using SQLAlchemy
# -------------------------------
# Update these values with your own credentials.
# If your password contains special characters (like '@'), URL-encode it.
username = "root"
password = "Varad@2004"  # Example password that contains '@'
encoded_password = urllib.parse.quote_plus(password)  # Encodes special characters
host = "localhost"
database = "DataWarehouse"

# Create the SQLAlchemy engine with the URL-encoded password.
engine = create_engine(f"mysql+mysqlconnector://{username}:{encoded_password}@{host}/{database}")
print("Connected to DataWarehouse")

# -------------------------------
# 2. OLAP Query: Roll-Up / Drill-Down Operation
#    Aggregate average delays grouped by Airline and Flight Class.
# -------------------------------
query_rollup = """
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

df_rollup = pd.read_sql(query_rollup, engine)
print("\nRoll-Up / Drill-Down Results:")
print(df_rollup)

# Visualize (drop subtotal rows with NULL values)
df_rollup_viz = df_rollup.dropna(subset=["Airline_Name", "Class"])
plt.figure(figsize=(10, 6))
sns.barplot(data=df_rollup_viz, x="Airline_Name", y="Avg_Departure_Delay", hue="Class")
plt.title("Average Departure Delay by Airline and Flight Class")
plt.xlabel("Airline")
plt.ylabel("Avg Departure Delay (min)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 3. OLAP Query: Slice Operation
#    Focus on one airline – here, 'Air India'.
# -------------------------------
query_slice = """
SELECT 
    a.Airline_Name, 
    AVG(ffe.Departure_Delay) AS Avg_Departure_Delay
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
WHERE a.Airline_Name = 'Air India'
GROUP BY a.Airline_Name;
"""

df_slice = pd.read_sql(query_slice, engine)
print("\nSlice Operation (Air India):")
print(df_slice)

plt.figure(figsize=(4,4))
plt.bar(df_slice['Airline_Name'], df_slice['Avg_Departure_Delay'], color='skyblue')
plt.title("Air India: Average Departure Delay")
plt.ylabel("Avg Departure Delay (min)")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. OLAP Query: Dice Operation
#    Filter for specific airlines ('Air India' and 'SpiceJet') and only Economy class flights.
# -------------------------------
query_dice = """
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

df_dice = pd.read_sql(query_dice, engine)
print("\nDice Operation (Air India & SpiceJet, Economy Class):")
print(df_dice)

plt.figure(figsize=(6,4))
sns.barplot(data=df_dice, x="Airline_Name", y="FlightCount")
plt.title("Flight Count for Economy Class (Air India & SpiceJet)")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.show()

# -------------------------------
# 5. OLAP Query: Pivot Operation using Pandas
#    Create a pivot table of average departure delays by Airline and Flight Class.
# -------------------------------
query_pivot = """
SELECT 
    a.Airline_Name, 
    f.Class,
    ffe.Departure_Delay
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
JOIN Dim_Flight f ON ffe.Flight_Key = f.Flight_Key;
"""

df_pivot_source = pd.read_sql(query_pivot, engine)
pivot_table = pd.pivot_table(
    df_pivot_source,
    index="Airline_Name",
    columns="Class",
    values="Departure_Delay",
    aggfunc='mean'
)
print("\nPivot Table: Average Departure Delay by Airline and Flight Class:")
print(pivot_table)

plt.figure(figsize=(8,6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Heatmap: Average Departure Delay by Airline & Flight Class")
plt.xlabel("Flight Class")  # Add this
plt.ylabel("Airline")  # Add this
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Additional OLAP Analysis: Overall Satisfaction Distribution
#    Count of feedback by Overall Satisfaction rating.
# -------------------------------
query_satisfaction = """
SELECT 
    s.Overall_Satisfaction,
    COUNT(*) AS Count
FROM Fact_FlightExperience ffe
JOIN Dim_ServiceRatings s ON ffe.Service_Key = s.Service_Key
GROUP BY s.Overall_Satisfaction;
"""

df_satisfaction = pd.read_sql(query_satisfaction, engine)
print("\nOverall Satisfaction Distribution:")
print(df_satisfaction)

plt.figure(figsize=(6,4))
sns.barplot(data=df_satisfaction, x="Overall_Satisfaction", y="Count")
plt.title("Distribution of Overall Satisfaction Ratings")
plt.xlabel("Overall Satisfaction")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. Additional OLAP Analysis: Payment Method Usage
#    Distribution of Payment Methods used across all bookings.
# -------------------------------
query_payment = """
SELECT 
    ffe.Payment_Method,
    COUNT(*) AS Count
FROM Fact_FlightExperience ffe
GROUP BY ffe.Payment_Method;
"""

df_payment = pd.read_sql(query_payment, engine)
print("\nPayment Method Distribution:")
print(df_payment)

plt.figure(figsize=(6,4))
sns.barplot(data=df_payment, x="Payment_Method", y="Count")
plt.title("Distribution of Payment Methods")
plt.xlabel("Payment Method")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===============================
# Additional OLAP Features
# ===============================

# 8. Average Flight Distance by Departure City and Airline
query_distance = """
SELECT 
    ffd.Departure_City,
    a.Airline_Name,
    AVG(ffd.Flight_Distance) AS Avg_Flight_Distance
FROM Fact_FlightExperience ffe
JOIN Dim_Flight ffd ON ffe.Flight_Key = ffd.Flight_Key
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
GROUP BY ffd.Departure_City, a.Airline_Name;
"""

df_distance = pd.read_sql(query_distance, engine)
print("\nAverage Flight Distance by Departure City and Airline:")
print(df_distance)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_distance, x="Departure_City", y="Avg_Flight_Distance", hue="Airline_Name")
plt.title("Average Flight Distance by Departure City and Airline")
plt.xlabel("Departure City")
plt.ylabel("Avg Flight Distance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Count of Flights by Gender per Airline
query_gender = """
SELECT 
    a.Airline_Name,
    sfd.Gender,
    COUNT(*) AS FlightCount
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
JOIN Staging_FlightData sfd ON ffe.Passenger_Key = sfd.Passenger_ID
GROUP BY a.Airline_Name, sfd.Gender;
"""

df_gender = pd.read_sql(query_gender, engine)
print("\nCount of Flights by Gender per Airline:")
print(df_gender)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_gender, x="Airline_Name", y="FlightCount", hue="Gender")
plt.title("Count of Flights by Gender per Airline")
plt.xlabel("Airline")
plt.ylabel("Flight Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 10. Distribution of Travel Purpose by Nationality
query_travel_purpose = """
SELECT 
    Nationality,
    Travel_Purpose,
    COUNT(*) AS Count
FROM Staging_FlightData
GROUP BY Nationality, Travel_Purpose;
"""

df_travel = pd.read_sql(query_travel_purpose, engine)
print("\nDistribution of Travel Purpose by Nationality:")
print(df_travel)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_travel, x="Nationality", y="Count", hue="Travel_Purpose")
plt.title("Distribution of Travel Purpose by Nationality")
plt.xlabel("Nationality")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 11. Average Age by Travel Purpose
query_age_travel = """
SELECT 
    Travel_Purpose,
    AVG(Age) AS Avg_Age
FROM Staging_FlightData
GROUP BY Travel_Purpose;
"""

df_age_travel = pd.read_sql(query_age_travel, engine)
print("\nAverage Age by Travel Purpose:")
print(df_age_travel)

plt.figure(figsize=(6,4))
sns.barplot(data=df_age_travel, x="Travel_Purpose", y="Avg_Age")
plt.title("Average Age by Travel Purpose")
plt.xlabel("Travel Purpose")
plt.ylabel("Average Age")
plt.tight_layout()
plt.show()

# ============================================
# Additional OLAP/Analytics Features
# ============================================

# 12. Delay Analysis: Count and percentage of flights with departure delay > 15 minutes by Airline.
query_delay = """
SELECT 
    a.Airline_Name,
    COUNT(*) AS Total_Flights,
    SUM(CASE WHEN ffe.Departure_Delay > 15 THEN 1 ELSE 0 END) AS Delayed_Flights,
    ROUND(100 * SUM(CASE WHEN ffe.Departure_Delay > 15 THEN 1 ELSE 0 END) / COUNT(*), 2) AS Delay_Percentage
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
GROUP BY a.Airline_Name;
"""

df_delay = pd.read_sql(query_delay, engine)
print("\nDelay Analysis (Flights with Departure Delay > 15 min):")
print(df_delay)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_delay, x="Airline_Name", y="Delay_Percentage")
plt.title("Percentage of Flights with Departure Delay > 15 min by Airline")
plt.xlabel("Airline")
plt.ylabel("Delay Percentage (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Complaint Analysis: Distribution of Complaint Types by Airline.
query_complaints = """
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

df_complaints = pd.read_sql(query_complaints, engine)
print("\nComplaint Analysis (Complaint Types by Airline):")
print(df_complaints)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_complaints, x="Airline_Name", y="Complaint_Count", hue="Complaint_Type")
plt.title("Distribution of Complaint Types by Airline")
plt.xlabel("Airline")
plt.ylabel("Complaint Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 14. Booking Channel Usage: Count of flights by Booking Channel.
query_booking = """
SELECT 
    ffe.Booking_Channel,
    COUNT(*) AS FlightCount
FROM Fact_FlightExperience ffe
GROUP BY ffe.Booking_Channel;
"""

df_booking = pd.read_sql(query_booking, engine)
print("\nBooking Channel Usage:")
print(df_booking)

plt.figure(figsize=(8, 4))
sns.barplot(data=df_booking, x="Booking_Channel", y="FlightCount")
plt.title("Number of Flights by Booking Channel")
plt.xlabel("Booking Channel")
plt.ylabel("Flight Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 15. Seat Type Distribution: How different seat types are distributed across Airlines.
query_seat = """
SELECT 
    a.Airline_Name,
    sfd.Seat_Type,
    COUNT(*) AS SeatCount
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
JOIN Staging_FlightData sfd ON ffe.Passenger_Key = sfd.Passenger_ID
GROUP BY a.Airline_Name, sfd.Seat_Type;
"""

df_seat = pd.read_sql(query_seat, engine)
print("\nSeat Type Distribution by Airline:")
print(df_seat)

plt.figure(figsize=(12, 6))
sns.barplot(data=df_seat, x="Airline_Name", y="SeatCount", hue="Seat_Type")
plt.title("Seat Type Distribution by Airline")
plt.xlabel("Airline")
plt.ylabel("Seat Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 16. Frequent Flyer Analysis: Count of flights where the passenger is a Frequent Flyer by Airline.
query_frequent = """
SELECT 
    a.Airline_Name,
    COUNT(*) AS FrequentFlyerCount
FROM Fact_FlightExperience ffe
JOIN Dim_Airline a ON ffe.Airline_Key = a.Airline_Key
WHERE a.Frequent_Flyer = 'Yes'
GROUP BY a.Airline_Name;
"""

df_frequent = pd.read_sql(query_frequent, engine)
print("\nFrequent Flyer Analysis (Flights with Frequent Flyer = 'Yes'):")
print(df_frequent)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_frequent, x="Airline_Name", y="FrequentFlyerCount")
plt.title("Count of Frequent Flyer Flights by Airline")
plt.xlabel("Airline")
plt.ylabel("Frequent Flyer Flight Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ============================================
# New Business Analytics Features
# ============================================

# 17. Customer Segmentation by Age & Travel Purpose
# Segment ages into ranges and calculate count and percentage of positive satisfaction.
# For simplicity, we consider Overall_Satisfaction as positive if it is 'Happy' or 'Satisfied'.
query_customer_segmentation = """
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

df_segmentation = pd.read_sql(query_customer_segmentation, engine)
print("\nCustomer Segmentation by Age & Travel Purpose:")
print(df_segmentation)

plt.figure(figsize=(10, 6))
sns.barplot(data=df_segmentation, x="Age_Group", y="Passenger_Count", hue="Travel_Purpose")
plt.title("Passenger Count by Age Group and Travel Purpose")
plt.xlabel("Age Group")
plt.ylabel("Passenger Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_segmentation, x="Age_Group", y="Positive_Satisfaction_Percentage", hue="Travel_Purpose")
plt.title("Positive Satisfaction (%) by Age Group and Travel Purpose")
plt.xlabel("Age Group")
plt.ylabel("Positive Satisfaction (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 18. On-Time Performance by Flight Route Type
# Calculate percentage of on-time flights (both departure and arrival delays ≤ 0) per route type.
query_on_time = """
SELECT 
    Flight_Route_Type,
    COUNT(*) AS Total_Flights,
    SUM(CASE WHEN Departure_Delay <= 0 AND Arrival_Delay <= 0 THEN 1 ELSE 0 END) AS On_Time_Flights,
    ROUND(100 * SUM(CASE WHEN Departure_Delay <= 0 AND Arrival_Delay <= 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS On_Time_Percentage
FROM Staging_FlightData
GROUP BY Flight_Route_Type;
"""

df_on_time = pd.read_sql(query_on_time, engine)
print("\nOn-Time Performance by Flight Route Type:")
print(df_on_time)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_on_time, x="Flight_Route_Type", y="On_Time_Percentage", palette="viridis")
plt.title("On-Time Performance by Flight Route Type")
plt.xlabel("Flight Route Type")
plt.ylabel("On-Time Percentage (%)")
plt.tight_layout()
plt.show()

# 19. Loyalty Impact Analysis
# Compare satisfaction percentages for different loyalty membership groups.
query_loyalty = """
SELECT 
    Loyalty_Membership,
    COUNT(*) AS Total_Passengers,
    ROUND(100 * AVG(CASE WHEN Overall_Satisfaction IN ('Happy', 'Satisfied') THEN 1 ELSE 0 END), 2) AS Positive_Satisfaction_Percentage
FROM Staging_FlightData
GROUP BY Loyalty_Membership;
"""

df_loyalty = pd.read_sql(query_loyalty, engine)
print("\nLoyalty Impact Analysis:")
print(df_loyalty)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_loyalty, x="Loyalty_Membership", y="Positive_Satisfaction_Percentage", palette="coolwarm")
plt.title("Positive Satisfaction (%) by Loyalty Membership")
plt.xlabel("Loyalty Membership")
plt.ylabel("Positive Satisfaction (%)")
plt.tight_layout()
plt.show()

# 20. Class Performance Analysis
# Analyze delays and satisfaction across different flight classes.
query_class_performance = """
SELECT 
    Class,
    AVG(Departure_Delay) AS Avg_Departure_Delay,
    AVG(Arrival_Delay) AS Avg_Arrival_Delay,
    ROUND(100 * AVG(CASE WHEN Overall_Satisfaction IN ('Happy', 'Satisfied') THEN 1 ELSE 0 END), 2) AS Positive_Satisfaction_Percentage
FROM Staging_FlightData
GROUP BY Class;
"""

df_class_perf = pd.read_sql(query_class_performance, engine)
print("\nClass Performance Analysis:")
print(df_class_perf)

# Create a grouped bar chart: one set for delays and one for satisfaction.
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot delays (using twin axes)
ax2 = ax1.twinx()

sns.barplot(data=df_class_perf, x="Class", y="Avg_Departure_Delay", color="lightblue", ax=ax1, label="Avg Departure Delay")
sns.barplot(data=df_class_perf, x="Class", y="Avg_Arrival_Delay", color="lightgreen", ax=ax1, label="Avg Arrival Delay")

ax1.set_ylabel("Average Delay (min)")
ax1.set_xlabel("Flight Class")
ax1.set_title("Flight Class Performance: Delays and Positive Satisfaction (%)")
ax1.legend(loc="upper left")

# Plot satisfaction percentage as a line on the secondary axis
sns.pointplot(data=df_class_perf, x="Class", y="Positive_Satisfaction_Percentage", color="red", ax=ax2, markers="o")
ax2.set_ylabel("Positive Satisfaction (%)")
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.show()

print("\nDatabase operations complete.")
