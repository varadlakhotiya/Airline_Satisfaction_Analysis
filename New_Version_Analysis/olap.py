import pandas as pd
import numpy as np
import plotly.express as px

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset
df = pd.read_csv('Cleaned_Data.csv')

# Step 2: Data Preprocessing
# Ensure 'Overall_Satisfaction' is properly encoded (e.g., 0 = Unsatisfied, 1 = Satisfied)
df['Overall_Satisfaction'] = df['Overall_Satisfaction'].apply(lambda x: 1 if x > 0 else 0)


# Step 3: Map numerical Class values to class names
class_mapping = {
    0: 'Economy',
    1: 'Business',
    2: 'First Class'
}
df['Class'] = df['Class'].map(class_mapping)


# Handle missing values (if any)
df.fillna(0, inplace=True)  # Replace missing values with 0 or use appropriate imputation

# Step 3: Transform Boolean Flags into Single Categorical Columns
# Extract airline names from boolean flags
airline_columns = [col for col in df.columns if col.startswith('Airline_Name_')]
df['Airline_Name'] = df[airline_columns].idxmax(axis=1).str.replace('Airline_Name_', '').str.replace('_True', '')

# Extract departure and arrival cities from boolean flags
departure_columns = [col for col in df.columns if col.startswith('Departure_City_')]
df['Departure_City'] = df[departure_columns].idxmax(axis=1).str.replace('Departure_City_', '').str.replace('_True', '')

arrival_columns = [col for col in df.columns if col.startswith('Arrival_City_')]
df['Arrival_City'] = df[arrival_columns].idxmax(axis=1).str.replace('Arrival_City_', '').str.replace('_True', '')

# Extract travel purpose from boolean flags
travel_purpose_columns = [col for col in df.columns if col.startswith('Travel_Purpose_')]
df['Travel_Purpose'] = df[travel_purpose_columns].idxmax(axis=1).str.replace('Travel_Purpose_', '').str.replace('_True', '')

# Extract seat type from boolean flags
seat_type_columns = [col for col in df.columns if col.startswith('Seat_Type_')]
df['Seat_Type'] = df[seat_type_columns].idxmax(axis=1).str.replace('Seat_Type_', '').str.replace('_True', '')

# Extract booking channel from boolean flags
booking_channel_columns = [col for col in df.columns if col.startswith('Booking_Channel_')]
df['Booking_Channel'] = df[booking_channel_columns].idxmax(axis=1).str.replace('Booking_Channel_', '').str.replace('_True', '')

# Step 4: Create a Data Cube
# Define dimensions and measures for OLAP operations
dimensions = ['Class', 'Gender', 'Departure_City', 'Arrival_City', 'Airline_Name', 'Travel_Purpose', 'Seat_Type', 'Booking_Channel']
measures = ['Overall_Satisfaction']

# Step 5: OLAP Operations

# Function to perform Roll-up (Aggregation)
def roll_up(data, group_by_column, measure_column):
    """
    Roll-up: Aggregate data by a specific dimension.
    """
    return data.groupby(group_by_column)[measure_column].mean().reset_index()

# Function to perform Drill-down
def drill_down(data, group_by_columns, measure_column):
    """
    Drill-down: Analyze data at a more granular level.
    """
    return data.groupby(group_by_columns)[measure_column].mean().reset_index()

# Function to perform Slice and Dice
def slice_and_dice(data, filter_condition):
    """
    Slice and Dice: Filter data based on a condition.
    """
    return data[filter_condition]

# Step 6: Perform OLAP Operations and Visualize Results

# Example 1: Roll-up - Average Satisfaction by Travel Class
rollup_result = roll_up(df, 'Class', 'Overall_Satisfaction')
print("Roll-up: Average Satisfaction by Travel Class")
print(rollup_result)

# Export Roll-up Result to CSV
rollup_result.to_csv('rollup_travel_class.csv', index=False)

# Visualize Roll-up Result
fig1 = px.bar(rollup_result, x='Class', y='Overall_Satisfaction', title='Average Satisfaction by Travel Class')
fig1.show()

# Example 2: Drill-down - Average Satisfaction by Gender within Each Travel Class
drilldown_result = drill_down(df, ['Class', 'Gender'], 'Overall_Satisfaction')
print("Drill-down: Average Satisfaction by Gender within Each Travel Class")
print(drilldown_result)

# Export Drill-down Result to CSV
drilldown_result.to_csv('drilldown_gender_travel_class.csv', index=False)

# Visualize Drill-down Result
fig2 = px.bar(drilldown_result, x='Class', y='Overall_Satisfaction', color='Gender', 
              title='Average Satisfaction by Gender within Each Travel Class', barmode='group')
fig2.show()

# Example 3: Slice and Dice - Satisfaction for Flights with Departure Delay > 15 minutes
sliced_data = slice_and_dice(df, df['Departure_Delay'] > 15)
print("Slice and Dice: Satisfaction for Flights with Departure Delay > 15 minutes")
print(sliced_data[['Passenger_ID', 'Departure_Delay', 'Overall_Satisfaction']].head())

# Export Slice and Dice Result to CSV
if not sliced_data.empty:
    sliced_data.to_csv('slice_departure_delay.csv', index=False)
    fig3 = px.histogram(sliced_data, x='Overall_Satisfaction', title='Satisfaction for Flights with Departure Delay > 15 minutes')
    fig3.show()
else:
    print("No data found for flights with Departure Delay > 15 minutes.")

# Example 4: Roll-up - Average Satisfaction by Airline
rollup_airline = roll_up(df, 'Airline_Name', 'Overall_Satisfaction')
print("Roll-up: Average Satisfaction by Airline")
print(rollup_airline)

# Export Roll-up Result to CSV
rollup_airline.to_csv('rollup_airline.csv', index=False)

# Visualize Roll-up Result
fig4 = px.bar(rollup_airline, x='Airline_Name', y='Overall_Satisfaction', title='Average Satisfaction by Airline')
fig4.show()

# Example 5: Drill-down - Average Satisfaction by Departure and Arrival City
drilldown_city = drill_down(df, ['Departure_City', 'Arrival_City'], 'Overall_Satisfaction')
print("Drill-down: Average Satisfaction by Departure and Arrival City")
print(drilldown_city)

# Export Drill-down Result to CSV
drilldown_city.to_csv('drilldown_city.csv', index=False)

# Visualize Drill-down Result
fig5 = px.scatter(drilldown_city, x='Departure_City', y='Arrival_City', size='Overall_Satisfaction', 
                  title='Average Satisfaction by Departure and Arrival City')
fig5.show()

# Example 6: Roll-up - Average Satisfaction by Travel Purpose
rollup_travel_purpose = roll_up(df, 'Travel_Purpose', 'Overall_Satisfaction')
print("Roll-up: Average Satisfaction by Travel Purpose")
print(rollup_travel_purpose)

# Export Roll-up Result to CSV
rollup_travel_purpose.to_csv('rollup_travel_purpose.csv', index=False)

# Visualize Roll-up Result
fig6 = px.pie(rollup_travel_purpose, names='Travel_Purpose', values='Overall_Satisfaction', 
              title='Average Satisfaction by Travel Purpose')
fig6.show()

# Example 7: Roll-up - Average Satisfaction by Seat Type
rollup_seat_type = roll_up(df, 'Seat_Type', 'Overall_Satisfaction')
print("Roll-up: Average Satisfaction by Seat Type")
print(rollup_seat_type)

# Export Roll-up Result to CSV
rollup_seat_type.to_csv('rollup_seat_type.csv', index=False)

# Visualize Roll-up Result
fig7 = px.bar(rollup_seat_type, x='Seat_Type', y='Overall_Satisfaction', title='Average Satisfaction by Seat Type')
fig7.show()

# Example 8: Roll-up - Average Satisfaction by Booking Channel
rollup_booking_channel = roll_up(df, 'Booking_Channel', 'Overall_Satisfaction')
print("Roll-up: Average Satisfaction by Booking Channel")
print(rollup_booking_channel)

# Export Roll-up Result to CSV
rollup_booking_channel.to_csv('rollup_booking_channel.csv', index=False)

# Visualize Roll-up Result
fig8 = px.bar(rollup_booking_channel, x='Booking_Channel', y='Overall_Satisfaction', title='Average Satisfaction by Booking Channel')
fig8.show()