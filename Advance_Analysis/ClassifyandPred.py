import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Load the dataset
# Uncomment one of these lines based on your situation
file_path = 'final_updated_dataset.csv'  # Use if you have the actual file
# file_path = prepare_sample_data()      # Use if working with the sample data

# Read the dataset
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = pd.read_csv(file_path)
    print(f"Using sample data with {df.shape[0]} rows and {df.shape[1]} columns")

# Data Exploration and Preprocessing
def explore_data(df):
    """
    Perform exploratory data analysis
    """
    print("Dataset Overview:")
    print("-" * 50)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({'Missing Values': missing_values, 
                                 'Percent Missing': missing_percent})
    print("\nMissing Values:")
    print(missing_data[missing_data['Missing Values'] > 0])
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Descriptive statistics for numerical columns
    print("\nDescriptive Statistics:")
    print(df.describe(include='all').T)
    
    # Distribution of target variables
    target_variables = ['Overall_Satisfaction', 'Recommendation', 'Complaint_Submitted']
    for target in target_variables:
        if target in df.columns:
            plt.figure(figsize=(8, 5))
            df[target].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {target}')
            plt.tight_layout()
            plt.savefig(f'{target}_distribution.png')
            print(f"\nDistribution of {target}:")
            print(df[target].value_counts(normalize=True) * 100)
    
    return df

# Preprocess the data
def preprocess_data(df, target_variable):
    """
    Preprocess the dataset for model training
    """
    # Convert flight duration to minutes
    if 'Flight_Duration' in df.columns:
        df['Flight_Duration_Minutes'] = df['Flight_Duration'].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else x
        )
    
    # Encode the target variable if it's not binary
    if target_variable in df.columns:
        if df[target_variable].nunique() > 2:
            print(f"Encoding target variable {target_variable} with values: {df[target_variable].unique()}")
            label_encoder = LabelEncoder()
            df[target_variable] = label_encoder.fit_transform(df[target_variable])
            # Store mapping for interpretation
            target_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
            print(f"Target mapping: {target_mapping}")
        elif not pd.api.types.is_numeric_dtype(df[target_variable]):
            # Binary but not numeric
            df[target_variable] = df[target_variable].map({'Yes': 1, 'No': 0, 'Maybe': 0.5})
    
    # Split into features and target
    X = df.drop(['Passenger_ID', target_variable], axis=1, errors='ignore')
    if 'Passenger_ID' in df.columns and target_variable not in df.columns:
        X = df.drop(['Passenger_ID'], axis=1, errors='ignore')
    
    if target_variable in df.columns:
        y = df[target_variable]
    else:
        print(f"Target variable '{target_variable}' not found in the dataset")
        return None, None, None, None
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

# Create preprocessing pipeline
def create_preprocessor(numeric_features, categorical_features):
    """
    Create a scikit-learn preprocessing pipeline for numerical and categorical features
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Create model pipelines
def create_models(preprocessor):
    """
    Create different classification models with preprocessing pipeline
    """
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'SVM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(probability=True, random_state=42))
        ]),
        'KNN': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())
        ]),
        'Naive Bayes': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ]),
        'Neural Network': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(max_iter=1000, random_state=42))
        ])
    }
    
    return models

# Evaluate models
def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluate different classification models
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(report)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
        
        # If binary classification, plot ROC curve
        if len(np.unique(y_test)) == 2:
            try:
                # Get probability predictions
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {name}')
                plt.legend(loc='lower right')
                plt.savefig(f'roc_curve_{name.replace(" ", "_")}.png')
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'{name}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {name}')
                plt.legend()
                plt.savefig(f'pr_curve_{name.replace(" ", "_")}.png')
            except:
                print(f"Could not generate ROC curve for {name}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    
    return results, best_model[1]['model']

# Hyperparameter tuning for the best model
def tune_best_model(best_model_name, model, X_train, y_train):
    """
    Tune hyperparameters for the best performing model
    """
    param_grid = {}
    
    if 'Logistic Regression' in best_model_name:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__solver': ['liblinear', 'lbfgs'],
            'classifier__penalty': ['l1', 'l2']
        }
    elif 'Decision Tree' in best_model_name:
        param_grid = {
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif 'Random Forest' in best_model_name:
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        }
    elif 'SVM' in best_model_name:
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf', 'poly'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1]
        }
    elif 'KNN' in best_model_name:
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
    elif 'Neural Network' in best_model_name:
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        }
    else:
        print(f"No hyperparameter grid defined for {best_model_name}")
        return model
    
    print(f"\nTuning hyperparameters for {best_model_name}...")
    
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# Feature importance analysis
def feature_importance(model, preprocessor, X):
    """
    Extract and visualize feature importance from the model
    """
    # Get feature names after preprocessing
    try:
        feature_names = []
        
        # Get numerical feature names
        if hasattr(preprocessor, 'transformers_'):
            num_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'num'][0]
            num_features = preprocessor.transformers_[num_idx][2]
            feature_names.extend(num_features)
            
            # Get one-hot encoded categorical feature names
            cat_idx = [i for i, (name, _, _) in enumerate(preprocessor.transformers_) if name == 'cat'][0]
            cat_transformer = preprocessor.transformers_[cat_idx][1].named_steps['onehot']
            cat_features = preprocessor.transformers_[cat_idx][2]
            
            if hasattr(cat_transformer, 'get_feature_names_out'):
                cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
                feature_names.extend(cat_feature_names)
            else:
                # Fallback for older scikit-learn versions
                feature_names.extend([f"cat_{i}" for i in range(cat_transformer.transform(X[cat_features].fillna('missing').iloc[:1]).shape[1])])
        
        # Extract feature importance
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            
            if len(importances) == len(feature_names):
                # Create DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
                
                # Plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df)
                plt.title('Top 20 Feature Importance')
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                
                return importance_df
            else:
                print(f"Feature importance shape mismatch: {len(importances)} vs {len(feature_names)}")
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'coef_'):
            # For linear models like logistic regression
            coefs = model.named_steps['classifier'].coef_[0]
            
            if len(coefs) == len(feature_names):
                # Create DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': np.abs(coefs)
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Coefficient', ascending=False).head(20)
                
                # Plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Coefficient', y='Feature', data=importance_df)
                plt.title('Top 20 Feature Coefficients (Absolute)')
                plt.tight_layout()
                plt.savefig('feature_coefficients.png')
                
                return importance_df
            else:
                print(f"Coefficient shape mismatch: {len(coefs)} vs {len(feature_names)}")
        else:
            print("Model does not have feature_importances_ or coef_ attribute")
            
    except Exception as e:
        print(f"Error in feature importance extraction: {e}")
    
    return None

# Main execution function
# Main execution function
def main():
    """
    Main function to run the classification pipeline
    """
    global df  # Declare df as global or pass it as a parameter
    
    print("=" * 50)
    print("AIRLINE PASSENGER CLASSIFICATION")
    print("=" * 50)
    
    # 1. Explore the data
    print("\n[1] EXPLORATORY DATA ANALYSIS")
    df = explore_data(df)  # This line was causing the error
    
    # Try different target variables for classification
    target_variables = ['Overall_Satisfaction', 'Recommendation', 'Complaint_Submitted']
    
    # Rest of the function remains the same
    for target_variable in target_variables:
        if target_variable in df.columns:
            print("\n" + "=" * 50)
            print(f"CLASSIFICATION FOR TARGET: {target_variable}")
            print("=" * 50)
            
            # 2. Preprocess the data
            print(f"\n[2] PREPROCESSING DATA FOR {target_variable}")
            X_train, X_test, y_train, y_test, numeric_features, categorical_features = preprocess_data(df, target_variable)
            
            if X_train is None:
                continue
            
            # 3. Create preprocessor
            preprocessor = create_preprocessor(numeric_features, categorical_features)
            
            # 4. Create models
            models = create_models(preprocessor)
            
            # 5. Evaluate models
            print(f"\n[3] EVALUATING CLASSIFICATION MODELS FOR {target_variable}")
            results, best_model = evaluate_models(models, X_train, X_test, y_train, y_test)
            
            # 6. Tune the best model
            print(f"\n[4] HYPERPARAMETER TUNING FOR {target_variable}")
            best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            tuned_model = tune_best_model(best_model_name, best_model, X_train, y_train)
            
            # 7. Final evaluation
            print(f"\n[5] FINAL MODEL EVALUATION FOR {target_variable}")
            y_pred = tuned_model.predict(X_test)
            final_accuracy = accuracy_score(y_test, y_pred)
            final_report = classification_report(y_test, y_pred)
            
            print(f"Final Tuned Model Accuracy: {final_accuracy:.4f}")
            print("Classification Report:")
            print(final_report)
            
            # 8. Feature importance analysis
            print(f"\n[6] FEATURE IMPORTANCE ANALYSIS FOR {target_variable}")
            importance_df = feature_importance(tuned_model, preprocessor, X_train)
            if importance_df is not None:
                print("Top 10 important features:")
                print(importance_df.head(10))
            
            print("\n" + "-" * 50)

# Execute the main function
if __name__ == "__main__":
    main()

# 1. Multi-class Classification with SMOTE for Imbalanced Data
def run_multiclass_with_smote():
    """
    Run multi-class classification with SMOTE for handling imbalanced classes
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    print("\n" + "=" * 50)
    print("MULTI-CLASS CLASSIFICATION WITH SMOTE")
    print("=" * 50)
    
    target_variable = 'Overall_Satisfaction'
    
    if target_variable not in df.columns:
        print(f"Target variable '{target_variable}' not found")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, numeric_features, categorical_features = preprocess_data(df, target_variable)
    
    if X_train is None:
        return
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Create SMOTE pipeline
    smote_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train model
    print("Training model with SMOTE...")
    smote_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = smote_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"SMOTE Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - SMOTE Random Forest')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_smote.png')
    
# 2. Multi-Label Classification (predicting multiple targets at once)
def run_multi_label_classification():
    """
    Run multi-label classification to predict multiple targets simultaneously
    """
    from sklearn.multioutput import MultiOutputClassifier
    
    print("\n" + "=" * 50)
    print("MULTI-LABEL CLASSIFICATION")
    print("=" * 50)
    
    target_variables = ['Overall_Satisfaction', 'Recommendation']
    
    # Check if target variables exist
    for target in target_variables:
        if target not in df.columns:
            print(f"Target variable '{target}' not found")
            return
    
    # Preprocess targets
    for target in target_variables:
        if df[target].nunique() > 2:
            label_encoder = LabelEncoder()
            df[target] = label_encoder.fit_transform(df[target])
    
    # Split data
    X = df.drop(['Passenger_ID'] + target_variables, axis=1, errors='ignore')
    y = df[target_variables]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Create multi-output classifier
    multi_output_model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    # Create pipeline
    multi_label_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', multi_output_model)
    ])
    
    # Train model
    print("Training Multi-Label Classifier...")
    multi_label_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = multi_label_pipeline.predict(X_test)
    
    # Evaluate each target separately
    for i, target in enumerate(target_variables):
        accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        report = classification_report(y_test.iloc[:, i], y_pred[:, i])
        
        print(f"\nResults for {target}:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {target}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_multilabel_{target}.png')

# 3. Time-based Validation (if dataset has time component)
def run_time_based_validation():
    """
    Run classification with time-based validation
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    print("\n" + "=" * 50)
    print("TIME-BASED VALIDATION CLASSIFICATION")
    print("=" * 50)
    
    # This is a placeholder - would need actual date/time columns in the dataset
    # We'll simulate it using Passenger_ID as a proxy for time ordering
    
    target_variable = 'Overall_Satisfaction'
    
    if 'Passenger_ID' not in df.columns or target_variable not in df.columns:
        print("Passenger_ID or target variable not found")
        return
    
    # Sort by Passenger_ID as a proxy for time
    time_sorted_df = df.sort_values('Passenger_ID')
    
    # Preprocess target
    if time_sorted_df[target_variable].nunique() > 2:
        label_encoder = LabelEncoder()
        time_sorted_df[target_variable] = label_encoder.fit_transform(time_sorted_df[target_variable])
    
    # Split features and target
    X = time_sorted_df.drop(['Passenger_ID', target_variable], axis=1, errors='ignore')
    y = time_sorted_df[target_variable]
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Create model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Running Time-based Cross-validation...")
    
    fold_accuracies = []
    
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Fold {i+1} Accuracy: {accuracy:.4f}")
        fold_accuracies.append(accuracy)
    
    print(f"\nAverage accuracy across folds: {np.mean(fold_accuracies):.4f}")
    
# Run all advanced techniques
def run_advanced_techniques():
    """
    Run all advanced classification techniques
    """
    # Run multi-class with SMOTE
    run_multiclass_with_smote()
    
    # Run multi-label classification
    run_multi_label_classification()
    
    # Run time-based validation
    run_time_based_validation()

# Enhanced main function to run both basic and advanced techniques
def enhanced_main():
    """
    Main function to run both basic and advanced classification techniques
    """
    # Run basic techniques
    main()
    
    # Run advanced techniques
    run_advanced_techniques()

# Execute the enhanced main function
if __name__ == "__main__":
    enhanced_main()        