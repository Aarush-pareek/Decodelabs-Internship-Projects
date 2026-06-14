"""
Model evaluation: accuracy, precision, recall, f1, confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import io
import base64

def evaluate_model(model, X_test, y_test, average='weighted'):
    """
    Compute all classification metrics.
    Returns dictionary of metrics and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
    
    # Get class names if available
    target_names = None
    if hasattr(model, 'classes_'):
        target_names = model.classes_.astype(str)
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'classification_report': report,
        'y_pred': y_pred.tolist(),
        'y_true': y_test.tolist()
    }

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Generate confusion matrix heatmap and return base64 encoded image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save to bytes and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance for tree-based models.
    Returns base64 image.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        names = [feature_names[i] for i in indices]
        values = [importances[i] for i in indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(names)), values[::-1], color='#00f2fe')
        plt.yticks(range(len(names)), names[::-1])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64
    else:
        return None