import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_categorical_variables(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Identify categorical columns (object or boolean type)
    categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    # Initialize LabelEncoder for ordinal categorical variables (if applicable)
    label_encoders = {}
    ordinal_columns = []  # Define any ordinal columns here if needed
    
    for col in ordinal_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoder for reference
    
    # Apply One-Hot Encoding for non-ordinal categorical variables
    df = pd.get_dummies(df, columns=[col for col in categorical_columns if col not in ordinal_columns], drop_first=True)
    
    # Save the encoded dataset
    output_file = "Encoded_Flight_Data_Processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Encoded dataset saved as {output_file}")
    
    return df

# Run the encoding function
encoded_df = encode_categorical_variables('Encoded_Flight_Data.csv')

# Display the first few rows of the encoded dataset
print(encoded_df.head())
