import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

output_folder = "ClusteringResults"

# Function to load and explore data
def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration
    """
    print("Loading and exploring data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

# Function to preprocess data
def preprocess_data(df):
    """
    Preprocess the data for clustering
    """
    print("\nPreprocessing data...")
    
    # Make a copy of the dataframe
    processed_df = df.copy()
    
    # Handle missing values for numeric columns
    processed_df = processed_df.apply(lambda x: x.fillna(x.mean()) 
                                      if x.dtype.kind in 'if' else x.fillna(x.mode()[0]))
    
    # Convert satisfaction levels to numeric
    satisfaction_map = {
        'Poor': 1, 
        'Fair': 2, 
        'Good': 3, 
        'Excellent': 4,
        'Unavailable': np.nan,
        'No Connection': np.nan
    }
    
    service_columns = [
        'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 
        'Cleanliness', 'Cabin_Staff_Service', 'Legroom', 
        'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 
        'WiFi_Service'
    ]
    
    for col in service_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map(satisfaction_map).fillna(0)
    
    # Convert satisfaction ratings to numeric
    satisfaction_rating = {
        'Happy': 5,
        'Satisfied': 4,
        'Neutral': 3,
        'Dissatisfied': 2,
        'Unhappy': 1
    }
    
    if 'Overall_Satisfaction' in processed_df.columns:
        processed_df['Overall_Satisfaction'] = processed_df['Overall_Satisfaction'].map(satisfaction_rating).fillna(3)
    
    # Convert binary features
    binary_columns = [
        'Festival_Season_Travel', 'Recommendation', 'Complaint_Submitted', 
        'Baggage_Lost', 'Compensation_Received', 'Seat_Upgrade', 
        'Special_Assistance', 'Discount_Received', 'Preferred_Airline', 
        'Frequent_Flyer'
    ]
    
    for col in binary_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].map({'Yes': 1, 'No': 0, 'Maybe': 0.5, 'Sometimes': 0.5}).fillna(0)
    
    # Convert Flight_Duration to minutes if it's in HH:MM format
    if 'Flight_Duration' in processed_df.columns:
        if processed_df['Flight_Duration'].dtype == 'object':
            processed_df['Flight_Duration'] = processed_df['Flight_Duration'].apply(
                lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if ':' in str(x) else x
            )
    
    return processed_df

# Function to select features for clustering
def select_features(df, feature_set='satisfaction'):
    """
    Select features for different clustering scenarios
    """
    print(f"\nSelecting features for {feature_set} clustering...")
    
    if feature_set == 'satisfaction':
        # Features related to passenger satisfaction
        features = [
            'Seat_Comfort', 'InFlight_Entertainment', 'Food_Quality', 
            'Cleanliness', 'Cabin_Staff_Service', 'Legroom', 
            'Baggage_Handling', 'CheckIn_Service', 'Boarding_Process', 
            'WiFi_Service', 'Overall_Satisfaction'
        ]
    
    elif feature_set == 'demographics':
        # Features related to passenger demographics
        features = [
            'Age', 'Gender', 'Nationality', 'Travel_Purpose', 
            'Class', 'Flight_Distance', 'Festival_Season_Travel',
            'Loyalty_Membership', 'Frequent_Flyer', 'Booking_Channel'
        ]
    
    elif feature_set == 'travel_behavior':
        # Features related to travel behavior
        features = [
            'Travel_Purpose', 'Flight_Distance', 'Flight_Duration', 
            'Flight_Route_Type', 'Loyalty_Membership', 'Frequent_Flyer',
            'Preferred_Airline', 'Frequent_Route', 'Payment_Method', 
            'Class', 'Booking_Channel'
        ]
    
    elif feature_set == 'complaint':
        # Features related to complaints and issues
        features = [
            'Departure_Delay', 'Arrival_Delay', 'Overall_Satisfaction',
            'Recommendation', 'Complaint_Submitted', 'Complaint_Type',
            'Baggage_Lost', 'Compensation_Received'
        ]
    
    else:  # all
        # Use most important features from all categories
        features = [
            'Age', 'Class', 'Flight_Distance', 'Departure_Delay', 
            'Arrival_Delay', 'Flight_Duration', 'Seat_Comfort', 
            'InFlight_Entertainment', 'Food_Quality', 'Cleanliness', 
            'Cabin_Staff_Service', 'Legroom', 'Baggage_Handling', 
            'CheckIn_Service', 'Boarding_Process', 'Overall_Satisfaction',
            'Recommendation', 'Complaint_Submitted', 'Loyalty_Membership', 
            'Frequent_Flyer'
        ]
    
    # Return only the features that exist in the dataframe
    return [f for f in features if f in df.columns]

# Function to encode categorical features
def encode_categorical_features(df, features):
    """
    Encode categorical features for clustering
    """
    print("\nEncoding categorical features...")
    
    # Identify categorical and numerical features
    categorical_features = [
        feature for feature in features 
        if df[feature].dtype == 'object' and feature in df.columns
    ]
    
    numerical_features = [
        feature for feature in features 
        if feature in df.columns and feature not in categorical_features
    ]
    
    # Create transformers for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # Apply preprocessing
    X = preprocessor.fit_transform(df[features])
    
    # Get feature names
    encoded_feature_names = []
    
    # Add numerical feature names
    encoded_feature_names.extend(numerical_features)
    
    # Add one-hot encoded feature names
    if categorical_features:
        cat_feature_indices = [features.index(feature) for feature in categorical_features]
        cat_transformer = preprocessor.transformers_[1][1]
        onehot_encoder = cat_transformer.named_steps['onehot']
        
        for i, feature in enumerate(categorical_features):
            if hasattr(onehot_encoder, 'categories_'):
                feature_cats = onehot_encoder.categories_[i]
                for cat in feature_cats:
                    encoded_feature_names.append(f"{feature}_{cat}")
    
    print(f"Data shape after encoding: {X.shape}")
    
    return X, encoded_feature_names, preprocessor

# Function to reduce dimensionality
def reduce_dimensions(X, n_components=2):
    """
    Reduce dimensions using PCA for visualization
    """
    print(f"\nReducing dimensions to {n_components} components...")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    
    # Print explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance: {explained_variance}")
    print(f"Total explained variance: {np.sum(explained_variance):.2f}")
    
    return X_reduced, pca

# Function to perform K-means clustering
def kmeans_clustering(X, max_clusters=10):
    """
    Perform K-means clustering and determine optimal number of clusters
    """
    print("\nPerforming K-means clustering...")
    
    # Calculate silhouette scores for different number of clusters
    silhouette_scores = []
    inertia = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertia.append(kmeans.inertia_)
        
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    
    # Plot elbow method
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), inertia, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'kmeans_optimal_clusters.png'))
    plt.close()
    
    # Determine optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2
    print(f"Optimal number of clusters based on silhouette score: {optimal_clusters}")
    
    # Perform K-means with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    return cluster_labels, kmeans, optimal_clusters

# Function to visualize clusters in 2D
def visualize_clusters(X_reduced, cluster_labels, title="Cluster Visualization", filename="clusters.png"):
    """
    Visualize clusters in 2D using PCA
    """
    plt.figure(figsize=(10, 8))
    
    # Check if there are noise points (DBSCAN specific)
    unique_labels = np.unique(cluster_labels)
    has_noise = -1 in unique_labels
    
    # Separate noise points and valid clusters
    if has_noise:
        noise_mask = (cluster_labels == -1)
        valid_mask = ~noise_mask
        valid_labels = cluster_labels[valid_mask]
        unique_valid = np.unique(valid_labels)
    else:
        valid_mask = np.ones_like(cluster_labels, dtype=bool)
        valid_labels = cluster_labels
        unique_valid = unique_labels
    
    n_clusters = len(unique_valid)
    
    # If there are valid clusters, process them
    if n_clusters > 0:
        # Remap valid labels to 0-based indices
        label_map = {old: new for new, old in enumerate(unique_valid)}
        remapped_valid = np.array([label_map[l] for l in valid_labels])
        
        # Create colormap for valid clusters
        cmap = plt.cm.get_cmap('viridis', n_clusters)
        
        # Plot valid clusters
        scatter = plt.scatter(X_reduced[valid_mask, 0], X_reduced[valid_mask, 1], 
                              c=remapped_valid, cmap=cmap, alpha=0.8, s=50)
        
        # Create colorbar with original cluster labels as ticks
        cbar = plt.colorbar(scatter, label='Cluster')
        cbar.set_ticks(np.arange(n_clusters))
        cbar.set_ticklabels(unique_valid)
    
    # Plot noise points if any
    if has_noise:
        plt.scatter(X_reduced[noise_mask, 0], X_reduced[noise_mask, 1], 
                    c='grey', alpha=0.8, s=50, label='Noise')
        if n_clusters > 0:
            plt.legend()
    
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to analyze clusters
def analyze_clusters(df, cluster_labels, features, encoded_feature_names, preprocessor, feature_set, cluster_method):
    """
    Analyze the characteristics of each cluster
    """
    print(f"\nAnalyzing {cluster_method} clusters for {feature_set} feature set...")
    
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    # Calculate feature importance for each cluster
    cluster_profiles = pd.DataFrame()
    
    # For each feature, calculate mean/mode per cluster
    for feature in features:
        if df[feature].dtype in ['int64', 'float64']:
            # For numerical features, calculate mean per cluster
            feature_means = df_with_clusters.groupby('Cluster')[feature].mean()
            cluster_profiles[feature] = feature_means
        else:
            # For categorical features, calculate mode per cluster
            feature_modes = df_with_clusters.groupby('Cluster')[feature].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            )
            cluster_profiles[feature] = feature_modes
    
    # Add cluster size
    cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
    cluster_profiles['Cluster_Size'] = cluster_sizes
    cluster_profiles['Cluster_Percentage'] = (cluster_sizes / len(df_with_clusters) * 100).round(2)
    
    # Save cluster profiles
    cluster_profiles.to_csv(os.path.join(output_folder, f'{cluster_method}_{feature_set}_cluster_profiles.csv'))
    
    # Prepare visualization data
    if len(features) > 10:
        top_features = features[:10]
    else:
        top_features = features
    
    # Visualize cluster profiles for numerical features
    numerical_features = [f for f in top_features if df[f].dtype in ['int64', 'float64']]
    
    if numerical_features:
        plt.figure(figsize=(14, 8))
        cluster_profiles[numerical_features].plot(kind='bar', ax=plt.gca())
        plt.title(f'{cluster_method} Cluster Profiles - {feature_set} (Numerical Features)')
        plt.xlabel('Cluster')
        plt.ylabel('Average Value')
        plt.xticks(rotation=0)
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{cluster_method}_{feature_set}_numerical_profiles.png'))
        plt.close()
    
    # Report distinctive characteristics of each cluster
    print("\nCluster Profiles:")
    for cluster in unique_clusters:
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        cluster_size = len(cluster_data)
        cluster_pct = round(cluster_size / len(df_with_clusters) * 100, 2)
        
        print(f"\nCluster {cluster} ({cluster_size} passengers, {cluster_pct}% of total):")
        
        # Find distinctive features
        for feature in features:
            if df[feature].dtype in ['int64', 'float64']:
                # For numerical features
                overall_mean = df[feature].mean()
                cluster_mean = cluster_data[feature].mean()
                
                # If the difference is significant
                if abs(cluster_mean - overall_mean) > 0.5 * df[feature].std():
                    comparison = "higher than" if cluster_mean > overall_mean else "lower than"
                    print(f"  - Average {feature}: {cluster_mean:.2f} ({comparison} overall average of {overall_mean:.2f})")
            else:
                # For categorical features
                overall_mode = df[feature].mode()[0] if not df[feature].mode().empty else 'Unknown'
                cluster_mode = cluster_data[feature].mode()[0] if not cluster_data[feature].mode().empty else 'Unknown'
                
                # If the mode is different
                if cluster_mode != overall_mode:
                    print(f"  - Most common {feature}: {cluster_mode} (overall most common: {overall_mode})")
    
    return df_with_clusters, cluster_profiles

# Main function
def main():
    """
    Main function to run the clustering analysis
    """
    # Load and explore data
    file_path = 'final_updated_dataset.csv'  # Change this to your file path
    df = load_and_explore_data(file_path)
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Define feature sets to analyze
    feature_sets = ['satisfaction', 'demographics', 'travel_behavior', 'complaint', 'all']
    
    # Store results for comparison
    results = {}
    
    # For each feature set, perform different clustering
    for feature_set in feature_sets:
        print(f"\n{'-'*50}")
        print(f"Starting clustering analysis for {feature_set} features")
        print(f"{'-'*50}")
        
        # Select features
        features = select_features(processed_df, feature_set)
        
        # Encode categorical features
        X, encoded_feature_names, preprocessor = encode_categorical_features(processed_df, features)
        
        # Reduce dimensions for visualization
        X_reduced, pca = reduce_dimensions(X)
        
        # Perform K-means clustering
        kmeans_labels, kmeans_model, kmeans_clusters = kmeans_clustering(X)
        save_path = os.path.join(output_folder, f"kmeans_{feature_set}_clusters.png")
        visualize_clusters(X_reduced, kmeans_labels, f"K-means Clusters - {feature_set}", save_path)
        kmeans_df, kmeans_profiles = analyze_clusters(processed_df, kmeans_labels, features, encoded_feature_names, preprocessor, feature_set, "KMeans")
    
        # Store results
        results[feature_set] = {
            'KMeans': {
                'labels': kmeans_labels,
                'n_clusters': kmeans_clusters,
                'silhouette': silhouette_score(X, kmeans_labels)
            }
        }
    
    # Compare results across feature sets and clustering methods
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*80)
    
    summary_table = []
    
    for feature_set in feature_sets:
        for method in ['KMeans']:
            n_clusters = results[feature_set][method]['n_clusters']
            silhouette = results[feature_set][method]['silhouette']
            
            summary_table.append([feature_set, method, n_clusters, silhouette])
    
    # Print summary table
    summary_df = pd.DataFrame(summary_table, columns=['Feature Set', 'Method', 'Clusters', 'Silhouette Score'])
    summary_df = summary_df.sort_values(by='Silhouette Score', ascending=False)
    print("\nClustering methods ranked by silhouette score:")
    print(summary_df)
    
    # Identify best clustering model
    best_row = summary_df.iloc[0]
    best_feature_set = best_row['Feature Set']
    best_method = best_row['Method']
    best_score = best_row['Silhouette Score']
    best_clusters = best_row['Clusters']
    
    print(f"\nBest clustering model: {best_method} on {best_feature_set} features")
    print(f"Number of clusters: {best_clusters}")
    print(f"Silhouette score: {best_score:.3f}")
    
    # Save the best model's labels to the original dataframe
    df['Cluster'] = results[best_feature_set][best_method]['labels']
    df.to_csv(os.path.join(output_folder, 'airline_passengers_with_clusters.csv'), index=False)
    
    print("\nFinal dataset with cluster labels saved as 'airline_passengers_with_clusters.csv'")
    print("\nAnalysis complete! Check the output files for detailed results.")

# Execute main function
if __name__ == "__main__":
    main()