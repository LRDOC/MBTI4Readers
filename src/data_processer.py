import json
import pandas as pd

file_path = 'data/copy/bookMeta.json'

def load_book_meta(file_path: str) -> pd.DataFrame:
    """'
    Load book metadata from a JSON file and return a DataFrame.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    pd.DataFrame: DataFrame containing the book metadata.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    return df


def preprocess_book_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the book metadata DataFrame.
    Parameters:
    df (pd.DataFrame): DataFrame containing the book metadata.
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = df.fillna('') # Replace NaN values with empty strings

    # Convert list fields to strings
    list_fields = ['creators', 'characterEthnicity', 'characterGenderIdentity', 'characterRaceCulture',
                   'characterReligion', 'characterSexualOrientation', 'Awards', 'contentWarning',
                   'genre', 'historicalEvents', 'InternationalAwards', 'literaryDevices', 'modesOfWriting',
                   'subject', 'textFeatures', 'textStructure', 'topic', 'tags']

    for field in list_fields:
        df[field] = df[field].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Convert boolean fields to int so they can be used in models
    bool_fields = ['isFiction', 'isNonFiction', 'isBlended', 'hasMultiplePov', 'hasUnreliableNarrative']
    for field in bool_fields:
        df[field] = df[field].astype(int) # Convert boolean to int

    return df


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    '''
    Load and preprocess book metadata from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file.

    '''
    df = load_book_meta(file_path)
    df = preprocess_book_meta(df)
    return df
