"""
Prediction module: predict on new data using trained model.
"""

import pandas as pd
import numpy as np

def predict_from_dict(model, input_dict, feature_names, scaler=None, encoders=None):
    """
    Predict class for a single sample given as dictionary.
    input_dict: dict with feature names as keys.
    feature_names: ordered list of feature names.
    scaler: StandardScaler object (optional).
    encoders: dict of LabelEncoders for categorical features (optional).
    """
    # Create DataFrame with correct order
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    # Encode categorical features if encoders provided
    if encoders:
        for col, le in encoders.items():
            if col in input_df.columns:
                # Handle unseen labels
                val = input_df[col].iloc[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    # Fallback: set to most frequent class or 0
                    input_df[col] = 0
    
    # Scale numeric features
    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values
    
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
    
    return {
        'prediction': int(prediction) if isinstance(prediction, (np.integer, np.int64)) else prediction,
        'probabilities': proba.tolist() if proba is not None else None,
        'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
    }