"""
Data preprocessing utilities: load, clean, encode, scale.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(file_path, target_column=None):
    """
    Load CSV dataset and return features (X) and target (y).
    If target_column not given, assume last column is target.
    """
    df = pd.read_csv(file_path)
    if target_column is None:
        target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y, df.columns.tolist(), target_column

def handle_missing_values(df, strategy='drop'):
    """Handle missing values: drop rows or fill with mean/mode."""
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill_mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == 'fill_mode':
        df = df.fillna(df.mode().iloc[0])
    return df

def encode_categorical(X, y=None):
    """
    Encode categorical features using LabelEncoder.
    Returns encoded X and y, along with fitted encoders (if needed).
    """
    X_encoded = X.copy()
    le_dict = {}
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            le_dict[col] = le
    
    y_encoded = None
    le_y = None
    if y is not None:
        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(y)
    
    return X_encoded, y_encoded, le_dict, le_y

def scale_features(X_train, X_test):
    """Apply StandardScaler to numeric features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def get_dataset_summary(df):
    """Generate summary statistics for dataset preview."""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'statistics': df.describe().to_dict() if df.select_dtypes(include=[np.number]).shape[1] > 0 else {}
    }
    return summary