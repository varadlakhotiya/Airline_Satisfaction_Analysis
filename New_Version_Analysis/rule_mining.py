# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
# Replace 'your_dataset.csv' with the actual file name
df = pd.read_csv('Cleaned_Data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())

# Step 1: Preprocess the data for association rule mining
# We will focus on feedback-related columns and satisfaction
feedback_columns = [
    'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 'Cleanliness',
    'Cabin_Staff_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service',
    'Boarding_Process', 'WiFi_Service', 'Overall_Satisfaction'
]

# Create a binary matrix for feedback columns
# Convert ratings into binary values (1 for low/medium, 0 for high)
df_binary = df[feedback_columns].applymap(lambda x: 1 if x <= 5 else 0)

# Add 'Overall_Satisfaction' as a binary column (1 for dissatisfied, 0 for satisfied)
df_binary['Overall_Satisfaction'] = df['Overall_Satisfaction'].apply(lambda x: 1 if x <= 5 else 0)

# Display the binary matrix
print("\nBinary Matrix for Feedback Columns:")
print(df_binary.head())

# Step 2: Apply the Apriori Algorithm to find frequent item-sets
# Set a minimum support threshold (e.g., 0.2)
frequent_itemsets = apriori(df_binary, min_support=0.2, use_colnames=True)

# Display frequent item-sets
print("\nFrequent Item-sets:")
print(frequent_itemsets)

# Step 3: Generate association rules
# Set a minimum confidence threshold (e.g., 0.7)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display association rules
print("\nAssociation Rules:")
print(rules)

# Step 4: Filter and interpret meaningful rules
# Filter rules where the consequent is 'Overall_Satisfaction' (dissatisfaction)
dissatisfaction_rules = rules[rules['consequents'].apply(lambda x: 'Overall_Satisfaction' in x)]

# Display rules leading to dissatisfaction
print("\nRules Leading to Dissatisfaction:")
print(dissatisfaction_rules)

# Filter rules where the consequent is NOT 'Overall_Satisfaction' (other patterns)
other_rules = rules[rules['consequents'].apply(lambda x: 'Overall_Satisfaction' not in x)]

# Display other interesting patterns
print("\nOther Interesting Patterns:")
print(other_rules)

# Save the results to a CSV file (optional)
dissatisfaction_rules.to_csv('dissatisfaction_rules.csv', index=False)
other_rules.to_csv('other_rules.csv', index=False)

print("\nAssociation rule mining completed. Results saved to CSV files.")