import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cluster_sizes(cluster_profiles):
    """
    Plot the sizes of the clusters.

    Parameters:
    cluster_profiles (Dict[int, Dict[str, Any]]): Dictionary containing the cluster
    profiles

    Returns:
    None
    """

    sizes = [profile['size'] for profile in cluster_profiles.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sizes)), sizes)
    plt.xlabel('Cluster')
    plt.ylabel('Number of readers')
    plt.title('Cluster Sizes')
    plt.show()


def plot_genre_distribution(df):
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
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def analyze_reading_patterns(df, clusters):
    """
    Analyze the reading patterns of the clusters.

    Parameters:
    df (pd.DataFrame): DataFrame containing the reader profiles.
    clusters (pd.Series): Series containing the cluster labels.

    Returns:
    None
    """

    df['cluster'] = clusters

    # Analyze lexile levels
    df['lexileLevel'] = pd.to_numeric(df['lexileLevel'], errors='coerce')
    valid_lexile = df[df['lexileLevel'].notna()]

    if not valid_lexile.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y='lexileLevel', data=valid_lexile)
        plt.title('Lexile Level Distribution by Cluster')
        plt.show()
    else:
        print("No valid Lexile levels found for analysis.")

    # Analyze fiction vs non-fiction
    fiction_ratio = df.groupby('cluster')[['isFiction', 'isNonFiction']].mean()
    fiction_ratio.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Fiction vs Non-Fiction Ratio by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Ratio')
    plt.legend(['Fiction', 'Non-Fiction'])
    plt.tight_layout()
    plt.show()

    # Analyze top genres per cluster
    top_genres = df.groupby('cluster')['genre'].apply(lambda x: x.value_counts().nlargest(3).index.tolist())
    for cluster, genres in top_genres.items():
        print(f"Cluster {cluster} top genres: {', '.join(genres)}")
