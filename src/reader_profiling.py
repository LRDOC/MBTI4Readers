from typing import Dict, Any
import pandas as pd
from pandas import Series
from sklearn.cluster import KMeans


def cluster_readers(df: pd.DataFrame, n_clusters: int = 5, return_model: bool = False) -> KMeans | Series:
    """
    Cluster readers based on their reading preferences.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reader profiles.
    n_clusters (int): Number of clusters to create.
    return_model (bool): If True, return the KMeans model instead of cluster labels.

    Returns:
    pd.Series or KMeans: Series containing the cluster labels or KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters) # Initialize the KMeans model
    kmeans.fit(df) # Fit the KMeans model

    if return_model: # Return the KMeans model if specified
        return kmeans
    else:
        return pd.Series(kmeans.labels_, name='cluster')


def analyze_clusters(df: pd.DataFrame, clusters: pd.Series) -> Dict[int, Dict[str, Any]]:
    """
    Analyze the clusters and generate cluster profiles.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reader profiles.
    clusters (pd.Series): Series containing the cluster labels.

    Returns:
    Dict[int, Dict[str, Any]]: Dictionary containing the cluster profiles.
    """
    df['cluster'] = clusters
    cluster_profiles = {}

    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        profile = {
            'size': len(cluster_data),
            'top_genres': cluster_data['genre'].value_counts().head(3).to_dict(),
            'most_common_narrative_form': cluster_data['narrativeForm'].mode().iloc[0] if not cluster_data['narrativeForm'].empty else 'N/A',
        }

        # Handle lexileLevel from error Column lexileLevel contains 172 NaN values. Filling with mean.
        lexile_levels = pd.to_numeric(cluster_data['lexileLevel'], errors='coerce')
        profile['avg_lexile_level'] = lexile_levels.mean() if not lexile_levels.empty else 'N/A'

        cluster_profiles[cluster] = profile

    return cluster_profiles
