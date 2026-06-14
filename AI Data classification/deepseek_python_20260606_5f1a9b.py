"""
Model training module: train multiple classifiers and return trained models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train, **kwargs):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, **kwargs):
    """Train Decision Tree classifier."""
    model = DecisionTreeClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X_train, y_train)
    return model

def get_model_by_name(name):
    """Return model training function by name."""
    models = {
        'Logistic Regression': train_logistic_regression,
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest
    }
    return models.get(name, train_logistic_regression)