import csv
import mysql.connector
import os

def safe_int(val):
    """
    Converts a value to an integer.
    First, it tries converting using float.
    If that fails, it checks for boolean strings ("true"/"false") and returns 1 or 0.
    """
    try:
        return int(float(val))
    except ValueError:
        lower_val = val.strip().lower()
        if lower_val == 'true':
            return 1
        elif lower_val == 'false':
            return 0
        else:
            raise ValueError(f"Cannot convert {val} to int")

# Establish connection to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user=os.getenv("DB_USER"),       # Get username from environment variable
    password=os.getenv("DB_PASSWORD"),   # Get password from environment variable
    database="AirlineFeedbackDW"
)
cursor = conn.cursor()

# Open and read the CSV file
with open('Encoded_Flight_Data_Processed.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            # === 1. Insert into Passenger Demographics ===
            passenger_id = safe_int(row['Passenger_ID'])
            age = safe_int(row['Age'])
            gender = safe_int(row['Gender'])
            pclass = safe_int(row['Class'])
            sql = """
                INSERT IGNORE INTO dim_passenger_demographics (passenger_id, age, gender, class)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (passenger_id, age, gender, pclass))

            # === 2. Insert into Flight Details ===
            flight_distance = safe_int(row['Flight_Distance'])
            departure_delay = safe_int(row['Departure_Delay'])
            arrival_delay = safe_int(row['Arrival_Delay'])
            flight_duration = safe_int(row['Flight_Duration'])
            flight_id = passenger_id  # Adjust if you have a separate flight identifier
            sql = """
                INSERT IGNORE INTO dim_flight_details 
                    (flight_id, flight_distance, departure_delay, arrival_delay, flight_duration)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (flight_id, flight_distance, departure_delay, arrival_delay, flight_duration))

            # === 3. Insert into Service Ratings ===
            seat_comfort = safe_int(row['Seat_Comfort'])
            inflight_entertainment = safe_int(row['InFlight_Entertainment'])
            food_quality = safe_int(row['Food_Quality'])
            cleanliness = safe_int(row['Cleanliness'])
            cabin_staff_service = safe_int(row['Cabin_Staff_Service'])
            legroom = safe_int(row['Legroom'])
            baggage_handling = safe_int(row['Baggage_Handling'])
            checkin_service = safe_int(row['CheckIn_Service'])
            boarding_process = safe_int(row['Boarding_Process'])
            wifi_service = safe_int(row['WiFi_Service'])
            rating_id = passenger_id  # Assumed mapping
            sql = """
                INSERT IGNORE INTO dim_service_ratings 
                    (rating_id, seat_comfort, inflight_entertainment, food_quality, cleanliness, 
                     cabin_staff_service, legroom, baggage_handling, checkin_service, boarding_process, wifi_service)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (rating_id, seat_comfort, inflight_entertainment, food_quality, cleanliness,
                                   cabin_staff_service, legroom, baggage_handling, checkin_service, boarding_process, wifi_service))

            # === 4. Insert into Complaints (if any) ===
            complaint_type = None
            complaint_columns = [
                'Complaint_Type_Broken Entertainment System_True',
                'Complaint_Type_Delay_True',
                'Complaint_Type_Food_True',
                'Complaint_Type_Long Security Check_True',
                'Complaint_Type_No Complaint_True',
                'Complaint_Type_Other_True',
                'Complaint_Type_Service_True',
                'Complaint_Type_Turbulence_True',
                'Complaint_Type_Unhygienic Toilets_True'
            ]
            for col in complaint_columns:
                if row[col].strip().lower() == 'true':
                    # Remove the prefix and trailing '_True' to get a clean complaint type name.
                    complaint_type = col.replace("Complaint_Type_", "").replace("_True", "")
                    break
            if complaint_type:
                complaint_id = passenger_id  # Assumed mapping
                sql = """
                    INSERT IGNORE INTO dim_complaints (complaint_id, complaint_type)
                    VALUES (%s, %s)
                """
                cursor.execute(sql, (complaint_id, complaint_type))

            # === 5. Insert into Booking Information ===
            # Determine booking channel
            booking_channel = None
            for col in ['Booking_Channel_Mobile App_True', 'Booking_Channel_Online_True', 'Booking_Channel_Travel Agency_True']:
                if row[col].strip().lower() == 'true':
                    booking_channel = col.replace("Booking_Channel_", "").replace("_True", "")
                    break

            # Determine payment method
            payment_method = None
            for col in ['Payment_Method_Credit Card_True', 'Payment_Method_Debit Card_True',
                        'Payment_Method_Net Banking_True', 'Payment_Method_UPI_True']:
                if row[col].strip().lower() == 'true':
                    payment_method = col.replace("Payment_Method_", "").replace("_True", "")
                    break

            # Determine airline loyalty program
            airline_loyalty_program = None
            for col in ['Airline_Loyalty_Program_Flying Returns (Air India)_True',
                        'Airline_Loyalty_Program_IndiGo 6E Rewards_True',
                        'Airline_Loyalty_Program_SpiceClub_True']:
                if row[col].strip().lower() == 'true':
                    airline_loyalty_program = col.replace("Airline_Loyalty_Program_", "").replace("_True", "")
                    break

            booking_id = passenger_id  # Assumed mapping
            sql = """
                INSERT IGNORE INTO dim_booking (booking_id, booking_channel, payment_method, airline_loyalty_program)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (booking_id, booking_channel, payment_method, airline_loyalty_program))

            # === 6. Insert into Passenger Feedback Fact ===
            satisfaction = float(row['Overall_Satisfaction'])
            recommendation = safe_int(row['Recommendation'])
            complaint_submitted = safe_int(row['Complaint_Submitted'])
            loyalty_membership = safe_int(row['Loyalty_Membership'])
            frequent_flyer = safe_int(row['Frequent_Flyer'])
            baggage_lost = safe_int(row['Baggage_Lost'])
            compensation_received = safe_int(row['Compensation_Received'])
            seat_upgrade = safe_int(row['Seat_Upgrade'])
            special_assistance = safe_int(row['Special_Assistance'])
            refund_requested = safe_int(row['Refund_Requested'])
            discount_received = safe_int(row['Discount_Received'])
            preferred_airline = row['Preferred_Airline']

            feedback_comments_food = row['Feedback_Comments_Food was cold.']
            feedback_comments_good = row['Feedback_Comments_Good']
            feedback_comments_great_service = row['Feedback_Comments_Great service!']
            feedback_comments_loved_flight = row['Feedback_Comments_Loved the flight.']
            feedback_comments_no_feedback = row['Feedback_Comments_No Feedback']
            feedback_comments_terrible_experience = row['Feedback_Comments_Terrible experience.']
            feedback_comments_wifi_issues = row['Feedback_Comments_WiFi was not working.']

            # Determine frequent route (pick the first flag that is True)
            frequent_route = None
            for col in ['Frequent_Route_Chennai - Kolkata', 'Frequent_Route_Delhi - Mumbai', 'Frequent_Route_Pune - Delhi']:
                if row[col].strip().lower() == 'true':
                    frequent_route = col.replace("Frequent_Route_", "")
                    break

            travel_purpose_family = safe_int(row['Travel_Purpose_Family_True'])
            travel_purpose_leisure = safe_int(row['Travel_Purpose_Leisure_True'])
            travel_purpose_medical = safe_int(row['Travel_Purpose_Medical_True'])
            travel_purpose_study = safe_int(row['Travel_Purpose_Study_True'])
            travel_purpose_transit = safe_int(row['Travel_Purpose_Transit_True'])

            # Determine airline name
            airline_name = None
            for col in ['Airline_Name_AirAsia India_True', 'Airline_Name_Akasa Air_True', 'Airline_Name_Alliance Air_True',
                        'Airline_Name_GoAir_True', 'Airline_Name_Indigo_True', 'Airline_Name_SpiceJet_True',
                        'Airline_Name_TruJet_True', 'Airline_Name_Vistara_True']:
                if row[col].strip().lower() == 'true':
                    airline_name = col.replace("Airline_Name_", "").replace("_True", "")
                    break

            # Determine departure city
            departure_city = None
            for col in row:
                if col.startswith("Departure_City_") and col.endswith("_True"):
                    if row[col].strip().lower() == 'true':
                        departure_city = col.replace("Departure_City_", "").replace("_True", "")
                        break

            # Determine arrival city
            arrival_city = None
            for col in row:
                if col.startswith("Arrival_City_") and col.endswith("_True"):
                    if row[col].strip().lower() == 'true':
                        arrival_city = col.replace("Arrival_City_", "").replace("_True", "")
                        break

            # Use the complaint type determined earlier (if none, store an empty string)
            complaint_type_fact = complaint_type if complaint_type else ""

            # Determine seat type
            seat_type = None
            for col in ['Seat_Type_Middle_True', 'Seat_Type_Standard_True', 'Seat_Type_Window_True']:
                if row[col].strip().lower() == 'true':
                    seat_type = col.replace("Seat_Type_", "").replace("_True", "")
                    break

            sql = """
                INSERT INTO fact_passenger_feedback (
                    passenger_id, flight_id, satisfaction, recommendation, complaint_submitted,
                    loyalty_membership, frequent_flyer, baggage_lost, compensation_received,
                    seat_upgrade, special_assistance, refund_requested, discount_received,
                    preferred_airline, feedback_comments_food, feedback_comments_good,
                    feedback_comments_great_service, feedback_comments_loved_flight,
                    feedback_comments_no_feedback, feedback_comments_terrible_experience,
                    feedback_comments_wifi_issues, frequent_route, travel_purpose_family,
                    travel_purpose_leisure, travel_purpose_medical, travel_purpose_study,
                    travel_purpose_transit, airline_name, departure_city, arrival_city,
                    complaint_type, seat_type
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                passenger_id, flight_id, satisfaction, recommendation, complaint_submitted,
                loyalty_membership, frequent_flyer, baggage_lost, compensation_received,
                seat_upgrade, special_assistance, refund_requested, discount_received,
                preferred_airline, feedback_comments_food, feedback_comments_good,
                feedback_comments_great_service, feedback_comments_loved_flight,
                feedback_comments_no_feedback, feedback_comments_terrible_experience,
                feedback_comments_wifi_issues, frequent_route, travel_purpose_family,
                travel_purpose_leisure, travel_purpose_medical, travel_purpose_study,
                travel_purpose_transit, airline_name, departure_city, arrival_city,
                complaint_type_fact, seat_type
            ))
            conn.commit()
        except Exception as e:
            print(f"Error processing row {row['Passenger_ID']}: {e}")
            conn.rollback()

# Close the connection
cursor.close()
conn.close()
