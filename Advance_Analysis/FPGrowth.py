import os
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

CONFIG = {
    "transaction_columns": [
        'Travel_Purpose', 'Airline_Name', 'Class',
        'Loyalty_Membership', 'Recommendation',
        'Service_Class', 'Flight_Time_Cat',
        'Departure_City', 'Arrival_City'
    ],
    "min_support": 0.06,
    "max_len": 3,
    "lift_threshold": 1.7,
    "min_confidence": 0.65,
    "min_city_freq": 15  # Minimum occurrences to keep a city
}

def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Clean and standardize city names
    city_cols = ['Departure_City', 'Arrival_City']
    df[city_cols] = df[city_cols].apply(lambda x: x.str.strip().str.title())
    
    # Filter cities based on minimum frequency
    city_freq = pd.concat([df['Departure_City'], df['Arrival_City']]).value_counts()
    valid_cities = city_freq[city_freq >= CONFIG["min_city_freq"]].index
    
    for col in city_cols:
        df[col] = df[col].where(df[col].isin(valid_cities), 'Rare_City')
        print(f"\n{col} distribution:")
        print(df[col].value_counts().head(10))
    
    # Convert flight duration
    if 'Flight_Duration' in df.columns:
        df['Flight_Duration_Mins'] = df['Flight_Duration'].apply(
            lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
        
    # Service class calculation
    service_columns = ['Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality',
                      'Cleanliness', 'Cabin_Staff_Service', 'Legroom',
                      'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process']
    
    service_rank = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
    for col in service_columns:
        df[col+'_Score'] = df[col].map(service_rank).fillna(2)
    
    df['Service_Composite'] = df[[c+'_Score' for c in service_columns]].mean(axis=1)
    df['Service_Class'] = pd.qcut(df['Service_Composite'], q=3, 
                                labels=['Basic', 'Standard', 'Premium'])
    
    # Flight time categorization
    if 'Flight_Duration_Mins' in df.columns:
        bins = [0, 90, 180, 300, np.inf]
        labels = ['Short', 'Medium', 'Long', 'Extended']
        df['Flight_Time_Cat'] = pd.cut(df['Flight_Duration_Mins'], bins=bins, labels=labels)
    
    return df

def create_transactions(df: pd.DataFrame) -> list:
    transactions = []
    city_cols = ['Departure_City', 'Arrival_City']
    
    for _, row in df.iterrows():
        transaction = []
        for col in CONFIG["transaction_columns"]:
            val = row[col]
            if pd.notna(val):
                # Handle cities differently
                if col in city_cols:
                    if val != 'Rare_City':
                        transaction.append(f"{col.split('_')[0]}={val}")
                else:
                    transaction.append(f"{col}={str(val).replace(' ', '_')}")
        transactions.append(transaction)
    
    print("\nValid transaction example:", transactions[0])
    return transactions

def analyze_associations(transactions: list):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df_encoded, 
                                min_support=CONFIG["min_support"],
                                use_colnames=True,
                                max_len=CONFIG["max_len"])
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets,
                                 metric="lift",
                                 min_threshold=CONFIG["lift_threshold"])
        rules = rules[
            (rules['confidence'] >= CONFIG["min_confidence"]) &
            (rules['consequents'].apply(len) == 1)
        ]
        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
        return frequent_itemsets, rules
    return frequent_itemsets, pd.DataFrame()

def generate_business_insights(rules: pd.DataFrame):
    print("\n=== Actionable Insights ===")
    
    if rules.empty:
        print("No strong patterns found. Suggestions:")
        print(f"- Lower min_support from {CONFIG['min_support']}")
        print(f"- Reduce lift_threshold from {CONFIG['lift_threshold']}")
        return
    
    # Categorize rules
    categories = {
        'City Patterns': [],
        'Service Relationships': [],
        'Loyalty Program Insights': [],
        'Flight Operations': []
    }
    
    for idx, (_, row) in enumerate(rules.sort_values('lift', ascending=False).iterrows(), 1):
        ant = " + ".join(row['antecedents'])
        cons = row['consequents'][0]
        
        insight = (f"Rule #{idx}: WHEN {ant} THEN {cons}\n"
                   f"Confidence: {row['confidence']:.0%} | "
                   f"Support: {row['support']:.0%} | "
                   f"Lift: {row['lift']:.1f}x\n")
        
        if any('Departure=' in s or 'Arrival=' in s for s in row['antecedents'] + row['consequents']):
            categories['City Patterns'].append(insight)
        elif 'Service' in cons or 'Service' in ant:
            categories['Service Relationships'].append(insight)
        elif 'Loyalty' in cons or 'Loyalty' in ant:
            categories['Loyalty Program Insights'].append(insight)
        else:
            categories['Flight Operations'].append(insight)
    
    for category, insights in categories.items():
        if insights:
            print(f"\n=== {category} ===")
            print("\n".join(insights[:3]))  # Show top 3 per category

def main():
    file_path = 'final_updated_dataset.csv'
    try:
        df = load_dataset(file_path)
        df = preprocess_data(df)
        transactions = create_transactions(df)
        frequent_itemsets, rules = analyze_associations(transactions)
        
        frequent_itemsets.to_csv('itemsets.csv', index=False)
        rules.to_csv('association_rules.csv', index=False)
        
        generate_business_insights(rules)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()