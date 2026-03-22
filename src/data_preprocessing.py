# src/data_preprocessing.py
import pandas as pd

def load_data(path):
    """Load CSV dataset and clean column names"""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove any extra spaces
    return df

def preprocess_data(df):
    """Preprocess dataset: drop non-numeric columns and separate target"""
    # Drop URL column if present
    if "url" in df.columns:
        df = df.drop("url", axis=1)

    # Separate features and target
    X = df.drop("status", axis=1)
    y = df["status"]

    return X, y