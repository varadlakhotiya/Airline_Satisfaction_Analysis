import pandas as pd
import numpy as np

# Load the cleaned dataset
# Replace 'cleaned_dataset.csv' with the actual file path
df = pd.read_csv('cleaned_dataset.csv')

# Step 1: Decode One-Hot Encoded Columns
# Function to decode one-hot encoded columns into a single categorical column
def decode_one_hot(df, prefix):
    # Filter columns with the given prefix
    columns = [col for col in df.columns if col.startswith(prefix)]
    # Decode the one-hot encoded columns
    decoded_col = df[columns].idxmax(axis=1).str.replace(prefix, '').str.replace('_True', '')
    return decoded_col

# Decode Departure_City, Arrival_City, and Travel_Purpose
df['Departure_City'] = decode_one_hot(df, 'Departure_City_')
df['Arrival_City'] = decode_one_hot(df, 'Arrival_City_')
df['Travel_Purpose'] = decode_one_hot(df, 'Travel_Purpose_')

# Drop the original one-hot encoded columns
df.drop(columns=[col for col in df.columns if col.startswith(('Departure_City_', 'Arrival_City_', 'Travel_Purpose_'))], inplace=True)

# Step 2: Create a Data Cube
# Define dimensions and measures
dimensions = ['Class', 'Gender', 'Travel_Purpose', 'Departure_City', 'Arrival_City']
measures = ['Overall_Satisfaction', 'Flight_Distance', 'Total_Delay']

# Create a data cube by grouping dimensions and aggregating measures
data_cube = df.groupby(dimensions)[measures].agg({
    'Overall_Satisfaction': 'mean',  # Average satisfaction
    'Flight_Distance': 'sum',       # Total flight distance
    'Total_Delay': 'mean'           # Average total delay
}).reset_index()

# Display the data cube
print("Data Cube:")
print(data_cube.head())

# Step 3: Roll-Up (Aggregate Data)
# Roll-up by aggregating data at a higher level (e.g., by Class and Gender)
rollup_cube = data_cube.groupby(['Class', 'Gender'])[measures].agg({
    'Overall_Satisfaction': 'mean',
    'Flight_Distance': 'sum',
    'Total_Delay': 'mean'
}).reset_index()

print("\nRoll-Up Cube (by Class and Gender):")
print(rollup_cube)

# Step 4: Drill-Down (Analyze Details)
# Drill-down to analyze details within a specific dimension (e.g., Class = 1 and Gender = 0)
drilldown_cube = data_cube[(data_cube['Class'] == 1) & (data_cube['Gender'] == 0)]

print("\nDrill-Down Cube (Class = 1, Gender = 0):")
print(drilldown_cube)

# Step 5: Slice (Filter Data)
# Slice the data cube to focus on a specific dimension value (e.g., Departure_City = 'Delhi')
slice_cube = data_cube[data_cube['Departure_City'] == 'Delhi']

print("\nSlice Cube (Departure_City = Delhi):")
print(slice_cube)

# Step 6: Dice (Filter Multiple Dimensions)
# Dice the data cube to focus on specific dimension values (e.g., Class = 1, Gender = 0, Departure_City = 'Delhi')
dice_cube = data_cube[
    (data_cube['Class'] == 1) &
    (data_cube['Gender'] == 0) &
    (data_cube['Departure_City'] == 'Delhi')
]

print("\nDice Cube (Class = 1, Gender = 0, Departure_City = Delhi):")
print(dice_cube)

# Step 7: Save OLAP Results
# Save the data cube and OLAP results to CSV files
data_cube.to_csv('OLAP Data/data_cube.csv', index=False)
rollup_cube.to_csv('OLAP Data/rollup_cube.csv', index=False)
drilldown_cube.to_csv('OLAP Data/drilldown_cube.csv', index=False)
slice_cube.to_csv('OLAP Data/slice_cube.csv', index=False)
dice_cube.to_csv('OLAP Data/dice_cube.csv', index=False)

print("\nOLAP results saved to CSV files.")