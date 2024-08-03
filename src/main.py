import os
import traceback
from data_processer import load_and_preprocess_data
from feature_engineering import engineer_features
from reader_profiling import cluster_readers, analyze_clusters
from book_rec import create_user_profile, recommend_books
from analyze import plot_cluster_sizes, plot_genre_distribution, analyze_reading_patterns
from cross_validation import cluster_cross_validation, recommendation_cross_validation


def main():
    file_path = '../data/copy/bookMeta.json'
    print(f"Current working directory: {os.getcwd()}")

    try:
        df = load_and_preprocess_data(file_path)
        if df is None:
            print("Failed to load data. Exiting.")
            return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        print(traceback.format_exc())
        return

    print(f"Successfully loaded data. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types: {df.dtypes}")

    try:
        book_features = engineer_features(df)
        print(f"Features engineered. Shape: {book_features.shape}")
        print(f"Engineered features: {book_features.columns.tolist()}")
    except Exception as e:
        print(f"An error occurred while engineering features: {e}")
        print(traceback.format_exc())
        return

    try:
        # Perform cross-validation for clustering
        mean_silhouette, std_silhouette = cluster_cross_validation(book_features)
        print(f"Clustering Cross-Validation Results:")
        print(f"Mean Silhouette Score: {mean_silhouette:.4f} (+/- {std_silhouette:.4f})")

        # Perform cross-validation for recommendations
        mean_precision, std_precision = recommendation_cross_validation(df, book_features)
        print(f"Recommendation Cross-Validation Results:")
        print(f"Mean Precision: {mean_precision:.4f} (+/- {std_precision:.4f})")

        clusters = cluster_readers(book_features)
        print(f"Clustering completed. Number of clusters: {len(set(clusters))}")
    except Exception as e:
        print(f"An error occurred during cross-validation or clustering: {e}")
        print(traceback.format_exc())
        return

    try:
        cluster_profiles = analyze_clusters(df, clusters)
        print(f"Cluster profiles created. Number of profiles: {len(cluster_profiles)}")
        for cluster, profile in cluster_profiles.items():
            print(f"Cluster {cluster}:")
            print(f"  Size: {profile['size']}")
            print(f"  Top genres: {profile['top_genres']}")
            print(f"  Most common narrative form: {profile['most_common_narrative_form']}")
            print()
    except Exception as e:
        print(f"An error occurred while analyzing clusters: {e}")
        print(traceback.format_exc())
        return

    try:
        plot_cluster_sizes(cluster_profiles)
        plot_genre_distribution(df)
        analyze_reading_patterns(df, clusters)
    except Exception as e:
        print(f"An error occurred while plotting: {e}")
        print(traceback.format_exc())

    # Mock user preferences
    try:
        user_preferences = {
            'genre': 'fantasy',
            'isFiction': 1,
        }
        user_profile = create_user_profile(user_preferences, book_features)
        recommendations = recommend_books(user_profile, book_features)

        print("Recommended books:")
        print(df.loc[recommendations, 'title'])
    except Exception as e:
        print(f"An error occurred while recommending books: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
