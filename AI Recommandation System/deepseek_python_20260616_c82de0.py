from flask import Flask, request, jsonify
from flask_cors import CORS
from recommendation_engine import RecommendationEngine

app = Flask(__name__)
CORS(app)  # Enable cross-origin for React frontend

# Initialize engine with dataset
engine = RecommendationEngine('dataset/items.json')

@app.route('/api/items', methods=['GET'])
def get_items():
    """Return all items (for search/filter)"""
    return jsonify(engine.items)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    Expects JSON: { "interests": ["AI", "Programming", ...] }
    Returns top 10 recommendations with scores, reasons, confidence
    """
    data = request.get_json()
    user_interests = data.get('interests', [])
    if not user_interests:
        return jsonify([])
    
    recommendations = engine.get_recommendations(user_interests, top_n=10)
    return jsonify(recommendations)

@app.route('/api/analytics', methods=['POST'])
def analytics():
    """
    Expects JSON: { "interests": [...] }
    Returns aggregated analytics for the given user interests
    """
    data = request.get_json()
    user_interests = data.get('interests', [])
    analytics_data = engine.get_analytics(user_interests)
    return jsonify(analytics_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)