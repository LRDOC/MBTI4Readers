from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cluster_sizes(cluster_profiles: Dict[int, Dict[str, Any]]):
    """
    Plot the sizes of the clusters.

    Parameters:
    cluster_profiles (Dict[int, Dict[str, Any]]): Dictionary containing the cluster
    profiles

    Returns:
    None
    """
    sizes = [profile['size'] for profile in cluster_profiles.values()]
    plt.bar(range(len(sizes)), sizes)
    plt.xlabel('Cluster')
    plt.ylabel('Number of readers')
    plt.title('Cluster Sizes')
    plt.show()


def plot_genre_distribution(df: pd.DataFrame):
    """
    Plot the distribution of genres.

    Parameters:
    df: DataFrame containing the reader profiles

    Returns:
    None
    """
    genre_counts = df['genre'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title('Top 10 Genres')
    plt.xticks(rotation=45)
    plt.show()


def analyze_reading_patterns(df: pd.DataFrame, clusters: pd.Series):
    """
    Analyze the reading patterns of the clusters.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reader profiles.
    clusters (pd.Series): Series containing the cluster labels.

    Returns:
    None
    """
    df['cluster'] = clusters
    # Unused atm TODO: make a main file
    avg_lexile_by_cluster = df.groupby('cluster')['lexileLevel'].mean()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='lexileLevel', data=df)
    plt.title('Lexile Level Distribution by Cluster')
    plt.show()
