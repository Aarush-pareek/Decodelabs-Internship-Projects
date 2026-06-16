import json
from typing import List, Dict

class RecommendationEngine:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.items = json.load(f)

    def _calculate_score(self, item: Dict, user_interests: List[str]) -> Dict:
        """
        Weighted scoring:
        interest_match = (# of user interests matching item tags) / len(user_interests)
        category_match = 1 if item['category'] in user_interests else 0
        rating_factor = item['rating'] / 5.0
        score = (interest_match * 0.6 + category_match * 0.3 + rating_factor * 0.1) * 100
        """
        user_set = set(user_interests)
        item_tags = set(item.get('tags', []))
        
        matching_tags = user_set.intersection(item_tags)
        interest_match = len(matching_tags) / len(user_set) if user_set else 0
        
        category_match = 1 if item.get('category', '') in user_set else 0
        rating_factor = item.get('rating', 0) / 5.0
        
        raw_score = (interest_match * 0.6) + (category_match * 0.3) + (rating_factor * 0.1)
        score_percent = round(raw_score * 100, 2)
        
        # Confidence levels
        if score_percent >= 70:
            confidence = "High Confidence"
        elif score_percent >= 40:
            confidence = "Medium Confidence"
        else:
            confidence = "Low Confidence"
        
        # Reason generation
        matched = list(matching_tags)
        if item['category'] in user_set:
            matched.append(item['category'])
        reason = f"Recommended because you selected {', '.join(matched[:3])}" if matched else "Based on popularity and rating"
        
        return {
            "score": score_percent,
            "matching_interests": matched,
            "confidence": confidence,
            "reason": reason
        }

    def get_recommendations(self, user_interests: List[str], top_n: int = 10) -> List[Dict]:
        """Return top N recommendations sorted by score desc, then rating desc"""
        scored = []
        for item in self.items:
            result = self._calculate_score(item, user_interests)
            scored.append({**item, **result})
        
        scored.sort(key=lambda x: (x['score'], x['rating']), reverse=True)
        return scored[:top_n]

    def get_analytics(self, user_interests: List[str]) -> Dict:
        """Generate analytics based on all items for dashboard"""
        # Category distribution across all items
        category_counts = {}
        for item in self.items:
            cat = item['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Interest distribution: how many items match each user interest
        interest_distribution = {interest: 0 for interest in user_interests}
        for item in self.items:
            tags = set(item.get('tags', []))
            for interest in user_interests:
                if interest in tags or item['category'] == interest:
                    interest_distribution[interest] += 1
        
        # Average rating per category
        category_ratings = {}
        category_rating_count = {}
        for item in self.items:
            cat = item['category']
            category_ratings[cat] = category_ratings.get(cat, 0) + item['rating']
            category_rating_count[cat] = category_rating_count.get(cat, 0) + 1
        avg_category_rating = {cat: round(category_ratings[cat] / category_rating_count[cat], 2)
                               for cat in category_ratings}
        
        # Top scoring items for this user (for histogram data)
        recommendations = self.get_recommendations(user_interests, top_n=10)
        scores = [r['score'] for r in recommendations]
        
        return {
            "category_distribution": category_counts,
            "interest_distribution": interest_distribution,
            "avg_category_rating": avg_category_rating,
            "recommendation_scores": scores,
            "total_items": len(self.items),
            "total_interests": len(user_interests),
            "recommendation_count": len(recommendations),
            "average_match_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "highest_match": recommendations[0] if recommendations else None
        }