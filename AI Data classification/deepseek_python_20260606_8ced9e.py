@app.route('/get_features')
def get_features():
    feature_names = session.get('feature_names', [])
    return jsonify(feature_names)