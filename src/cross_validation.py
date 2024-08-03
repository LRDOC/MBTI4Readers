from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
import numpy as np
from reader_profiling import cluster_readers
from book_rec import create_user_profile, recommend_books

def cluster_cross_validation(features, n_splits=5):
    """
    Perform clustering on the features and calculate silhouette score.

    Parameters:
    features (pd.DataFrame): DataFrame containing the features.
    n_splits (int): Number of splits for cross-validation.

    Returns:
    float: Mean silhouette score.
    float: Standard deviation of silhouette scores.
     """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    silhouette_scores = []

    for train_index, test_index in kf.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]

        train_model = cluster_readers(x_train, return_model=True)  # Get the KMeans model

        test_clusters = train_model.predict(x_test)  # Use the trained model to predict clusters for test data

        # Calculate silhouette score
        score = silhouette_score(x_test, test_clusters)
        silhouette_scores.append(score)

    return np.mean(silhouette_scores), np.std(silhouette_scores)

def recommendation_cross_validation(df, features, n_splits=5):
    """
    Perform recommendation and calculate precision.

    Parameters:
    df (pd.DataFrame): DataFrame containing the book metadata.
    features (pd.DataFrame): DataFrame containing the features.
    n_splits (int): Number of splits for cross-validation.

    Returns:
    float: Mean precision score.
    float: Standard deviation of precision scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    precision_scores = []

    for train_index, test_index in kf.split(features):
        x_train, x_test = features.iloc[train_index], features.iloc[test_index]
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        # Create a simple user profile from the test set
        test_user = df_test.iloc[0]
        user_preferences = {
            'genre': test_user['genre'],
            'isFiction': test_user['isFiction']
        }
        user_profile = create_user_profile(user_preferences, x_train)

        recommendations = recommend_books(user_profile, x_train) # Get recommendations

        # Calculate precision (assuming top 5 recommendations)
        recommended_genres = set(df_train.loc[recommendations, 'genre'].tolist())
        actual_genre = test_user['genre']
        precision = 1 if actual_genre in recommended_genres else 0
        precision_scores.append(precision)

    return np.mean(precision_scores), np.std(precision_scores)