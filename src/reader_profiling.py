from typing import Dict, Any
import pandas as pd
from sklearn.cluster import KMeans


def cluster_readers(df: pd.DataFrame, n_clusters: int = 5) -> pd.Series:
    """
    Cluster readers based on their reading preferences.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reader profiles.
    n_clusters (int): Number of clusters to create.

    Returns:
    pd.Series: Series containing the cluster labels.

    """
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(df)
    return pd.Series(clusters, name='cluster')


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
            'avg_lexile_level': cluster_data['lexileLevel'].mean(),
            'most_common_narrative_form': cluster_data['NarrativeForm'].mode().iloc[0],
        }
        cluster_profiles[cluster] = profile

    return cluster_profiles
