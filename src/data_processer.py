import json
import pandas as pd
import os


def load_book_meta(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error when reading file: {e}")
        return None

    df = pd.DataFrame(data)
    return df


def preprocess_book_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna('')

    list_fields = ['creators', 'characterEthnicity', 'characterGenderIdentity', 'characterRaceCulture',
                   'characterReligion', 'characterSexualOrientation', 'Awards', 'contentWarning',
                   'genre', 'historicalEvents', 'InternationalAwards', 'literaryDevices', 'modesOfWriting',
                   'subject', 'textFeatures', 'textStructure', 'topic', 'tags']

    for field in list_fields:
        if field in df.columns:
            df[field] = df[field].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    bool_fields = ['isFiction', 'isNonFiction', 'isBlended', 'hasMultiplePov', 'hasUnreliableNarrative']
    for field in bool_fields:
        if field in df.columns:
            df[field] = df[field].astype(int)

    return df


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    print(f"Attempting to load data from: {os.path.abspath(file_path)}")
    df = load_book_meta(file_path)
    if df is not None:
        print(f"Successfully loaded data. Shape before preprocessing: {df.shape}")
        df = preprocess_book_meta(df)
        print(f"Shape after preprocessing: {df.shape}")
        return df
    else:
        print("Failed to load data.")
        return None


if __name__ == "__main__":
    path = '../data/copy/bookMeta.json'
    print(f"Current working directory: {os.getcwd()}")
    df = load_and_preprocess_data(path)
    if df is not None:
        print(f"Successfully loaded and preprocessed data. Final shape: {df.shape}")
    else:
        print("Failed to load or preprocess data.")
