# 🤖 AI Data Classification System

## 🚀 Overview

An industry-grade web application that performs supervised machine learning classification. Upload any CSV dataset, train multiple models (Logistic Regression, Decision Tree, Random Forest), evaluate with precision, recall, F1, confusion matrix, and get interactive visualizations. Built with Flask, scikit-learn, Plotly, and a modern glassmorphic UI.

## ✨ Features

- **Dataset Upload & Preview** – Supports CSV, handles missing values, categorical encoding.
- **Model Training** – Choose from 3 algorithms, automatic train-test split (80/20), feature scaling.
- **Evaluation Metrics** – Accuracy, Precision, Recall, F1 Score, Confusion Matrix (heatmap), Feature Importance.
- **Prediction Module** – Dynamic form based on dataset features, real-time prediction.
- **Interactive Charts** – Plotly.js for distribution, accuracy comparison, and more.
- **Export Options** – Download results as CSV/PDF (extendable).
- **AI Insights** – Automated textual analysis of model performance.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript

## 📦 Installation

```bash
git clone https://github.com/yourusername/ai-classification-system.git
cd ai-classification-system
pip install -r requirements.txt
python app.py