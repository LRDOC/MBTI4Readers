from typing import Dict, Any

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def create_user_profile(user_preferences: Dict[str, Any], book_features: pd.DataFrame) -> pd.Series:
    """
    Create a user profile based on the user's preferences.

    Parameters:
    user_preferences (Dict[str, Any]): Dictionary containing the user's preferences.
    book_features (pd.DataFrame): DataFrame containing the book features.

    Returns:
    pd.Series: Series containing the user profile.
    """
    user_profile = pd.Series(0, index=book_features.columns)
    for feature, value in user_preferences.items():
        if feature in book_features.columns:
            user_profile[feature] = value
    return user_profile


def recommend_books(user_profile: pd.Series, book_features: pd.DataFrame, n_recommendations: int = 5) -> pd.Series:

    """
    Recommend books based on the user profile.

    Parameters:
    user_profile (pd.Series): Series containing the user profile.
    book_features (pd.DataFrame): DataFrame containing the book features.
    n_recommendations (int): Number of recommendations to return. (default: 5)

    Returns:
    pd.Series: Series containing the recommended book IDs.
    """
    similarities = cosine_similarity(user_profile.values.reshape(1, -1), book_features.values)
    similar_indices = similarities.argsort()[0][::-1]
    return book_features.index[similar_indices[:n_recommendations]]
