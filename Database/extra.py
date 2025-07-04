import pandas as pd
import numpy as np

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path
df = pd.read_csv('Encoded_Flight_Data_Processed_Scaled_Int_Clamped.csv')

# Step 1: Re-encode Overall_Satisfaction
# Mapping: Dissatisfied = 1, Neutral = 3, Happy = 4, Satisfied = 5
satisfaction_mapping = {0: 3, 1: 4, 2: 5, 3: 1}  # Assuming 0=Neutral, 1=Happy, 2=Satisfied, 3=Dissatisfied
df['Overall_Satisfaction'] = df['Overall_Satisfaction'].map(satisfaction_mapping)

# Step 2: Preprocess Feedback_Comments
# Combine all feedback columns into a single column
feedback_columns = [
    'Feedback_Comments_Food was cold.', 'Feedback_Comments_Good', 'Feedback_Comments_Great service!',
    'Feedback_Comments_Loved the flight.', 'Feedback_Comments_No Feedback', 'Feedback_Comments_Terrible experience.',
    'Feedback_Comments_WiFi was not working.'
]

# Create a new column 'Feedback' that combines all feedback comments
df['Feedback'] = df[feedback_columns].apply(lambda row: ', '.join(row.index[row == True]), axis=1)

# Drop the original feedback columns
df.drop(columns=feedback_columns, inplace=True)

# Step 3: Feature Engineering
# Create a new column 'Total_Delay' as the sum of Departure_Delay and Arrival_Delay
df['Total_Delay'] = df['Departure_Delay'] + df['Arrival_Delay']

# Step 4: Binarize Ratings for Association Rule Mining
# Convert service ratings (1-5) into binary values (Low = 1 if rating <= 2, else 0)
service_columns = [
    'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 'Cabin_Staff_Service',
    'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 'WiFi_Service'
]

for col in service_columns:
    df[f'Low_{col}'] = np.where(df[col] <= 2, 1, 0)  # 1 = Low, 0 = Not Low

# Step 5: Prepare Dataset for Classification
# Binarize Overall_Satisfaction for classification (Satisfied = 1 if >= 4, else 0)
df['Satisfied'] = np.where(df['Overall_Satisfaction'] >= 4, 1, 0)

# Step 6: Save the Cleaned Dataset
df.to_csv('cleaned_dataset.csv', index=False)

# Display the cleaned dataset
print(df.head())