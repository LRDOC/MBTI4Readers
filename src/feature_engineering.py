import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def engineer_text_features(df: pd.DataFrame, text_columns: list) -> pd.DataFrame:
    '''
    Engineer text features using TF-IDF.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text columns.
    text_columns (list): List of text columns to engineer.

    Returns:
    pd.DataFrame: DataFrame with the engineered text features.
    '''
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')

    for column in text_columns:
        tfidf_matrix = tfidf.fit_transform(df[column].fillna(''))
        feature_names = [f"{column}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        df = pd.concat([df, tfidf_df], axis=1)

    return df


def normalize_numerical_features(df: pd.DataFrame, numerical_columns: list) -> pd.DataFrame:
    '''
    Normalize numerical features using StandardScaler.

    Parameters:
    df (pd.DataFrame): DataFrame containing the numerical columns.
    numerical_columns (list): List of numerical columns to normalize.

    Returns:
    pd.DataFrame: DataFrame with the normalized numerical
    '''
    scaler = StandardScaler()
    for col in numerical_columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"Could not convert column {col} to numeric. Skipping.")
                continue
        if df[col].isna().sum() > 0:
            print(f"Column {col} contains {df[col].isna().sum()} NaN values. Filling with mean.")
            df[col] = df[col].fillna(df[col].mean())

    numerical_columns = [col for col in numerical_columns if df[col].dtype in ['int64', 'float64']]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


def reduce_dimensions(df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
    """"
    reduce the number of dimensions using PCA

    Parameters:
    df (pd.DataFrame): DataFrame containing the features to reduce.
    n_components (int): Number of components to reduce to.

    Returns:
    pd.DataFrame: DataFrame with the reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    pca_result = pca.fit_transform(df[numerical_columns])
    pca_df = pd.DataFrame(pca_result, columns=[f'PCA_{i}' for i in range(n_components)])
    return pca_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    this function engineers the features of the input DataFrame so that they can be used in ml model

    Parameters:
    df (pd.DataFrame): DataFrame containing the features to engineer

    Returns:
    pd.DataFrame: DataFrame with the engineered features
    '''
    text_columns = ['synopsis', 'subject', 'topic']
    df = engineer_text_features(df, text_columns)

    numerical_columns = ['lexileLevel', 'seriesBookNumber'] + [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    df = normalize_numerical_features(df, numerical_columns)

    df = reduce_dimensions(df)

    return df
