import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

# Data exploration
def explore_data(df):
    print("\n=== Dataset Overview ===")
    print("\nFirst 2 rows:")
    print(df.head(2))
    
    print("\nDataset Information:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe())
    
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    print("\nTarget Variable Distribution:")
    print(df['Overall_Satisfaction'].value_counts())
    print(df['Overall_Satisfaction'].value_counts(normalize=True) * 100)
    
    return df

# Data visualization
def visualize_data(df):
    print("\n=== Data Visualization ===")
    
    plt.figure(figsize=(12, 6))
    
    # Satisfaction distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x='Overall_Satisfaction', data=df)
    plt.title('Overall Satisfaction Distribution')
    
    # Age distribution by satisfaction
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Overall_Satisfaction', y='Age', data=df)
    plt.title('Age vs. Satisfaction')
    
    plt.tight_layout()
    plt.savefig('satisfaction_distribution.png')
    
    # Travel purpose and class distribution
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='Travel_Purpose', hue='Overall_Satisfaction', data=df)
    plt.title('Travel Purpose vs. Satisfaction')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='Class', hue='Overall_Satisfaction', data=df)
    plt.title('Class vs. Satisfaction')
    
    plt.tight_layout()
    plt.savefig('travel_purpose_class.png')
    
    # Correlation matrix for numerical features
    plt.figure(figsize=(14, 10))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation = df[numerical_cols].corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    
    # Satisfaction factors
    plt.figure(figsize=(14, 10))
    satisfaction_factors = ['Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 
                           'Cleanliness', 'Cabin_Staff_Service', 'Legroom', 
                           'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 'WiFi_Service']
    
    # Check if these factors exist in the dataset
    valid_factors = [factor for factor in satisfaction_factors if factor in df.columns]
    
    if valid_factors:
        df_melted = pd.melt(df, id_vars=['Overall_Satisfaction'], value_vars=valid_factors,
                           var_name='Factor', value_name='Rating')
        
        sns.countplot(x='Factor', hue='Rating', data=df_melted)
        plt.title('Satisfaction Factors')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('satisfaction_factors.png')
    
    print("Visualizations saved as PNG files.")

# Data preprocessing
def preprocess_data(df):
    print("\n=== Data Preprocessing ===")
    
    # Make a copy of the dataframe to avoid warnings
    df_processed = df.copy()
    
    # Check data types before processing
    print("\nData types before preprocessing:")
    print(df_processed.dtypes.head())
    
    # Convert categorical variables to numeric if they are ratings
    rating_columns = ['Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 
                      'Cleanliness', 'Cabin_Staff_Service', 'Legroom', 
                      'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 'WiFi_Service']
    
    # Make sure these columns exist in the dataset
    rating_columns = [col for col in rating_columns if col in df_processed.columns]
    
    rating_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4, 'Unavailable': 0, 'No Connection': 0}
    
    for col in rating_columns:
        df_processed[col] = df_processed[col].map(rating_map)
    
    # Convert Yes/No/Maybe columns to numeric
    binary_columns = ['Festival_Season_Travel', 'Recommendation', 'Complaint_Submitted', 
                     'Frequent_Flyer', 'Baggage_Lost', 'Compensation_Received', 
                     'Seat_Upgrade', 'Special_Assistance', 'Discount_Received', 'Preferred_Airline']
    
    # Make sure these columns exist in the dataset
    binary_columns = [col for col in binary_columns if col in df_processed.columns]
    
    binary_map = {'Yes': 1, 'No': 0, 'Maybe': 0.5, 'Sometimes': 0.5, 'Not Applicable': 0}
    
    for col in binary_columns:
        df_processed[col] = df_processed[col].map(binary_map)
    
    # Convert target variable to binary
    satisfaction_map = {'Happy': 1, 'Satisfied': 1, 'Neutral': 0, 'Dissatisfied': 0}
    df_processed['Satisfaction_Binary'] = df_processed['Overall_Satisfaction'].map(satisfaction_map)
    
    # Handle Flight_Duration - Check its type and format
    if 'Flight_Duration' in df_processed.columns:
        print("\nSample Flight_Duration values:")
        print(df_processed['Flight_Duration'].head(10))
        
        try:
            # First, we'll check if we can use the column as is (if it's already numerical)
            if pd.api.types.is_numeric_dtype(df_processed['Flight_Duration']):
                df_processed['Flight_Duration_Minutes'] = df_processed['Flight_Duration']
                print("Flight_Duration is already numeric, using as-is.")
            else:
                # Check if it's in the format HH:MM
                def convert_duration_to_minutes(duration_str):
                    if pd.isna(duration_str):
                        return np.nan
                    try:
                        if ':' in str(duration_str):
                            parts = str(duration_str).split(':')
                            return int(parts[0]) * 60 + int(parts[1])
                        else:
                            # If not in HH:MM format, try to use it directly as minutes
                            return float(duration_str)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert {duration_str} to minutes")
                        return np.nan
                
                df_processed['Flight_Duration_Minutes'] = df_processed['Flight_Duration'].apply(convert_duration_to_minutes)
                print("Converted Flight_Duration to minutes.")
        except Exception as e:
            print(f"Warning: Could not process Flight_Duration - {str(e)}")
            # If we can't convert it, create a dummy feature
            df_processed['Flight_Duration_Minutes'] = 60  # Default value
    else:
        print("Flight_Duration column not found. Creating dummy value.")
        df_processed['Flight_Duration_Minutes'] = 60  # Default value
    
    # Feature engineering: Total Delay
    if 'Departure_Delay' in df_processed.columns and 'Arrival_Delay' in df_processed.columns:
        df_processed['Total_Delay'] = df_processed['Departure_Delay'] + df_processed['Arrival_Delay']
    else:
        df_processed['Total_Delay'] = 0  # Default value if columns don't exist
    
    # Feature engineering: Service Quality Score
    service_columns = [col for col in rating_columns if col in df_processed.columns]
    
    if service_columns:
        df_processed['Service_Quality_Score'] = df_processed[service_columns].mean(axis=1)
    else:
        df_processed['Service_Quality_Score'] = 0  # Default value if columns don't exist
    
    # Identify categorical and numerical features for preprocessing
    categorical_features = [col for col in [
        'Gender', 'Nationality', 'Travel_Purpose', 'Airline_Name', 
        'Class', 'Departure_City', 'Arrival_City', 'Flight_Route_Type',
        'Seat_Type', 'Loyalty_Membership', 'Booking_Channel', 
        'Frequent_Route', 'Payment_Method', 'Airline_Loyalty_Program'
    ] if col in df_processed.columns]
    
    numerical_features = [col for col in [
        'Age', 'Flight_Distance', 'Departure_Delay', 'Arrival_Delay', 
        'Flight_Duration_Minutes', 'Total_Delay', 'Service_Quality_Score'
    ] + rating_columns + binary_columns if col in df_processed.columns]
    
    # Print column types for debugging
    print("\nCategorical features:", categorical_features)
    print("\nNumerical features:", numerical_features)
    
    # Check for any missing values and handle them
    for col in categorical_features + numerical_features:
        null_count = df_processed[col].isnull().sum()
        if null_count > 0:
            print(f"Column {col} has {null_count} missing values")
    
    # Drop unnecessary columns if they exist
    drop_columns = [col for col in [
        'Passenger_ID', 'Flight_Number', 'Flight_Duration', 'Overall_Satisfaction', 
        'Complaint_Type', 'Feedback_Comments'
    ] if col in df_processed.columns]
    
    # Return processed data and feature lists
    print(f"\nPreprocessed data shape: {df_processed.shape}")
    
    # Make sure features exist in the dataframe
    features = [f for f in categorical_features + numerical_features if f in df_processed.columns]
    
    # Check if target exists
    if 'Satisfaction_Binary' not in df_processed.columns:
        print("WARNING: Target variable 'Satisfaction_Binary' not created. Creating default.")
        # Create a default target (randomly)
        df_processed['Satisfaction_Binary'] = np.random.randint(0, 2, size=len(df_processed))
    
    return df_processed, features, 'Satisfaction_Binary'

# Build ML pipeline and train models
def build_and_train_models(df, features, target):
    print("\n=== Building and Training ML Models ===")
    
    # Validate that we have the target and features
    if target not in df.columns:
        raise ValueError(f"Target column {target} not found in dataframe")
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"WARNING: Some features are missing: {missing_features}")
        features = [f for f in features if f in df.columns]
    
    print(f"Using {len(features)} features for modeling")
    
    X = df[features]
    y = df[target]
    
    # FIX: Handle NaN values in the target variable
    nan_count = y.isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in target variable. Removing these rows.")
        # Get indices of non-NaN targets
        valid_indices = y.notna()
        # Filter X and y to keep only rows with non-NaN targets
        X = X[valid_indices]
        y = y[valid_indices]
    
    # Print target distribution
    print("Target distribution:")
    print(y.value_counts())
    
    # Rest of the function remains the same...
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Create preprocessing pipeline
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features for pipeline: {len(numerical_features)}")
    print(f"Categorical features for pipeline: {len(categorical_features)}")
    
    # Define preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    transformers = []
    
    if numerical_features:
        transformers.append(('num', numerical_transformer, numerical_features))
    
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Define models with simpler hyperparameter grids for faster execution
    models = {
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold()),  # Add this
            ('feature_selection', SelectKBest(f_classif, k=min(20, len(features)))),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'KNN': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold()),  # Added
            ('feature_selection', SelectKBest(f_classif, k=min(20, len(features)))),
            ('classifier', KNeighborsClassifier())
        ]),
         'DecisionTree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold()),  # Added
            ('feature_selection', SelectKBest(f_classif, k=min(20, len(features)))),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'NaiveBayes': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold()),  # Added
            ('feature_selection', SelectKBest(f_classif, k=min(20, len(features)))),
            ('classifier', GaussianNB())
        ])   
    }
    
    # Simplified parameter grids for faster execution
    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [None, 10],
            'feature_selection__k': [min(15, len(features))]
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights': ['uniform', 'distance'],
            'feature_selection__k': [min(15, len(features))]
        },
        'DecisionTree': {
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__criterion': ['gini', 'entropy'],
            'feature_selection__k': [min(15, len(features))]
        },
        'NaiveBayes': {
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7],
            'feature_selection__k': [min(15, len(features))]
        }
    }
    
    best_models = {}
    
    # Train and evaluate each model
    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[model_name], 
            cv=3,  # Reduced CV folds for faster execution
            scoring='accuracy',
            n_jobs=-1
        )
        
        try:
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            best_models[model_name] = best_model
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_name} Accuracy: {accuracy:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f'confusion_matrix_{model_name}.png')
        
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    return best_models, X_test, y_test

# A simplified feature importance analysis function
def analyze_feature_importance(models, X_test, y_test, df, features):
    print("\n=== Feature Importance Analysis ===")
    
    if not models:
        print("No models available for feature importance analysis")
        return
    
    # Function to get feature names after preprocessing steps
    def get_feature_names(pipeline):
        # Get preprocessor step
        preprocessor = pipeline.named_steps['preprocessor']
        # Get feature names after preprocessing (includes one-hot encoded names)
        feature_names = preprocessor.get_feature_names_out()
        
        # Apply variance threshold step
        vt = pipeline.named_steps['variance_threshold']
        vt_mask = vt.get_support()
        feature_names = feature_names[vt_mask]
        
        # Apply feature selection step
        fs = pipeline.named_steps['feature_selection']
        fs_mask = fs.get_support()
        feature_names = feature_names[fs_mask]
        
        return feature_names
    
    # For Random Forest
    if 'RandomForest' in models:
        try:
            rf_model = models['RandomForest']
            classifier = rf_model.named_steps['classifier']
            importances = classifier.feature_importances_
            
            # Get feature names after all preprocessing steps
            feature_names = get_feature_names(rf_model)
            
            print("\nTop Feature Importances (Random Forest):")
            top_n = min(10, len(importances))
            top_indices = np.argsort(importances)[::-1][:top_n]
            
            for i, idx in enumerate(top_indices, 1):
                print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names[top_indices], importances[top_indices])
            plt.title('Top Feature Importances - Random Forest')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importances_rf.png')
        
        except Exception as e:
            print(f"Error analyzing RF feature importance: {str(e)}")
    
    # For Decision Tree
    if 'DecisionTree' in models:
        try:
            dt_model = models['DecisionTree']
            classifier = dt_model.named_steps['classifier']
            importances = classifier.feature_importances_
            
            # Get feature names after all preprocessing steps
            feature_names = get_feature_names(dt_model)
            
            print("\nTop Feature Importances (Decision Tree):")
            top_n = min(10, len(importances))
            top_indices = np.argsort(importances)[::-1][:top_n]
            
            for i, idx in enumerate(top_indices, 1):
                print(f"{feature_names[idx]}: {importances[idx]:.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names[top_indices], importances[top_indices])
            plt.title('Top Feature Importances - Decision Tree')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importances_dt.png')
        
        except Exception as e:
            print(f"Error analyzing DT feature importance: {str(e)}")
            
            
            
# Main function to orchestrate the workflow
def main():
    print("=== Airline Passenger Satisfaction Analysis ===")
    
    try:
        # Load data
        file_path = 'final_updated_dataset.csv'
        df = load_data(file_path)
        
        # Explore data
        df = explore_data(df)
        
        # Visualize data
        visualize_data(df)
        
        # Preprocess data
        df_processed, features, target = preprocess_data(df)
        
        # Build and train models
        best_models, X_test, y_test = build_and_train_models(df_processed, features, target)
        
        # Analyze feature importance
        analyze_feature_importance(best_models, X_test, y_test, df_processed, features)
        
        print("\n=== Project Complete ===")
        print("All visualizations and analysis results have been saved.")
    
    except Exception as e:
        print(f"An error occurred in the main workflow: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()