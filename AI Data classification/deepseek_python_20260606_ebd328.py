"""
AI Data Classification System - Main Flask Application
"""

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import uuid

from utils.preprocessing import (load_dataset, handle_missing_values, encode_categorical,
                                 scale_features, split_data, get_dataset_summary)
from utils.training import get_model_by_name
from utils.evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance
from utils.prediction import predict_from_dict

app = Flask(__name__)
app.secret_key = 'ai_classification_system_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'csv'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dataset upload and preview dashboard."""
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_dataset():
    """Handle dataset upload, preprocess, and store in session."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load dataset
        try:
            df = pd.read_csv(filepath)
            # Store file path and dataset info in session
            session['dataset_path'] = filepath
            session['dataset_shape'] = df.shape
            # Preprocess basic: drop missing (optional, we'll let user decide later)
            df_clean = handle_missing_values(df, strategy='drop')
            # Store preview data (first 100 rows as dict)
            preview = df_clean.head(100).to_dict(orient='records')
            summary = get_dataset_summary(df_clean)
            session['dataset_columns'] = list(df_clean.columns)
            
            return jsonify({
                'success': True,
                'preview': preview,
                'summary': summary,
                'columns': list(df_clean.columns),
                'shape': df_clean.shape
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/train', methods=['POST'])
def train_model():
    """
    Train selected model on uploaded dataset.
    Expects JSON: { 'algorithm': 'Logistic Regression', 'target_column': 'species' }
    """
    data = request.get_json()
    algorithm = data.get('algorithm', 'Random Forest')
    target_column = data.get('target_column')
    
    dataset_path = session.get('dataset_path')
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({'error': 'No dataset uploaded. Please upload first.'}), 400
    
    try:
        # Load dataset
        df = pd.read_csv(dataset_path)
        # Handle missing values - simple drop for demo
        df = handle_missing_values(df, strategy='drop')
        
        # Identify features and target
        if target_column is None:
            target_column = df.columns[-1]  # default last column
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        feature_names = X.columns.tolist()
        
        # Encode categorical features and target
        X_encoded, y_encoded, encoders, le_target = encode_categorical(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Train model
        train_func = get_model_by_name(algorithm)
        model = train_func(X_train_scaled, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test_scaled, y_test)
        
        # Generate confusion matrix image
        class_names = le_target.classes_.astype(str) if le_target else None
        cm_img = plot_confusion_matrix(y_test, metrics['y_pred'], class_names)
        
        # Generate feature importance (if available)
        importance_img = plot_feature_importance(model, feature_names)
        
        # Save model and scaler, encoders, target encoder
        model_id = str(uuid.uuid4())
        model_path = os.path.join('models', f'model_{model_id}.pkl')
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'le_target': le_target,
            'feature_names': feature_names,
            'algorithm': algorithm
        }, model_path)
        session['model_path'] = model_path
        session['feature_names'] = feature_names
        session['target_encoder'] = le_target.classes_.tolist() if le_target else None
        
        # Prepare response
        return jsonify({
            'success': True,
            'algorithm': algorithm,
            'metrics': metrics,
            'confusion_matrix_image': cm_img,
            'feature_importance_image': importance_img,
            'model_id': model_id,
            'target_classes': le_target.classes_.tolist() if le_target else None,
            'feature_names': feature_names
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Prediction form and API."""
    if request.method == 'GET':
        # Get feature names from session to generate dynamic form
        feature_names = session.get('feature_names', [])
        return render_template('prediction.html', features=feature_names)
    
    elif request.method == 'POST':
        # Handle prediction request (JSON)
        data = request.get_json()
        user_input = data.get('features', {})
        
        model_path = session.get('model_path')
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found. Please train first.'}), 400
        
        try:
            saved = joblib.load(model_path)
            model = saved['model']
            scaler = saved['scaler']
            encoders = saved['encoders']
            le_target = saved['le_target']
            feature_names = saved['feature_names']
            
            # Predict
            result = predict_from_dict(model, user_input, feature_names, scaler, encoders)
            # Decode target if needed
            if le_target and result['prediction'] is not None:
                pred_class = le_target.inverse_transform([result['prediction']])[0]
                result['prediction_label'] = str(pred_class)
            else:
                result['prediction_label'] = str(result['prediction'])
            
            return jsonify({'success': True, 'prediction': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/results')
def results():
    """Show evaluation results page (renders metrics from session?)."""
    # For simplicity, we'll let the frontend fetch results via /train endpoint
    return render_template('results.html')

@app.route('/get_results')
def get_results():
    """Retrieve last saved evaluation results (optional)."""
    # This would typically fetch from session, but we'll keep lightweight
    return jsonify({'message': 'Use /train with algorithm to get fresh results'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)