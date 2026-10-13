"""End-to-end behaviour of the recommendation engine."""
import pytest

from src import config
from src.exceptions import EmptyPreferencesError, InvalidPreferencesError
from src.models import UserPreferences
from src.recommendation_engine import build_user_profile


def titles(result):
    return [r.title for r in result.recommendations]


def test_sci_fi_preference_ranks_sci_fi_first(small_engine):
    result = small_engine.generate_recommendations(UserPreferences(genres={"Sci-Fi": 3}, top_n=3))
    assert set(titles(result)) == {"Star Quest", "Dark Orbit"}


def test_results_are_sorted_by_descending_score(real_engine):
    prefs = UserPreferences(genres={"Action": 3, "Sci-Fi": 3, "Adventure": 3}, top_n=10)
    recs = real_engine.generate_recommendations(prefs).recommendations
    scores = [r.details.final_score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommendation_count_respects_top_n(real_engine):
    prefs = UserPreferences(genres={"Drama": 3}, top_n=4)
    assert len(real_engine.generate_recommendations(prefs).recommendations) == 4


def test_no_duplicate_recommendations(real_engine):
    prefs = UserPreferences(genres={"Drama": 3, "Crime": 3}, top_n=20)
    ids = [r.movie_id for r in real_engine.generate_recommendations(prefs).recommendations]
    assert len(ids) == len(set(ids))


def test_different_preferences_give_different_recommendations(real_engine):
    space = real_engine.generate_recommendations(
        UserPreferences(genres={"Sci-Fi": 3, "Adventure": 3}, top_n=5))
    romance = real_engine.generate_recommendations(
        UserPreferences(genres={"Romance": 3, "Comedy": 3}, top_n=5))
    assert set(titles(space)).isdisjoint(titles(romance))


def test_empty_preferences_raise_friendly_error(small_engine):
    with pytest.raises(EmptyPreferencesError):
        small_engine.generate_recommendations(UserPreferences())


def test_invalid_inputs_are_rejected(small_engine):
    with pytest.raises(InvalidPreferencesError):
        small_engine.generate_recommendations(UserPreferences(genres={"Western": 3}))
    with pytest.raises(InvalidPreferencesError):
        small_engine.generate_recommendations(UserPreferences(genres={"Sci-Fi": 3}, top_n=0))
    with pytest.raises(InvalidPreferencesError):
        small_engine.generate_recommendations(UserPreferences(genres={"Sci-Fi": 99}))
    with pytest.raises(InvalidPreferencesError):
        small_engine.generate_recommendations(UserPreferences(interests=["unknown tag"]))


def test_zero_similarity_items_are_not_padded_in(small_engine):
    result = small_engine.generate_recommendations(UserPreferences(genres={"Comedy": 3}, top_n=5))
    assert titles(result) == ["Laugh Riot"]
    assert result.warnings  # tells the user why fewer than 5 were returned


def test_min_rating_and_language_filters(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3, "Romance": 3}, min_rating=7.8, top_n=5)
    assert titles(small_engine.generate_recommendations(prefs)) == ["Star Quest"]
    prefs = UserPreferences(genres={"Sci-Fi": 3, "Romance": 3}, languages=["French"])
    assert titles(small_engine.generate_recommendations(prefs)) == ["Love Letters"]


def test_explanations_only_mention_real_matches(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3}, interests=["space"], top_n=2)
    for rec in small_engine.generate_recommendations(prefs).recommendations:
        assert set(rec.details.matched_genres) <= set(rec.genres) & {"Sci-Fi"}
        assert set(rec.details.matched_interests) <= set(rec.tags) & {"space"}
        for feature in rec.explanation.matched_features:
            assert feature in rec.explanation.reason


def test_explanation_does_not_claim_unmatched_genres(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3, "Romance": 3}, top_n=5)
    by_title = {r.title: r for r in small_engine.generate_recommendations(prefs).recommendations}
    assert "Romance" not in by_title["Star Quest"].explanation.reason
    assert "Sci-Fi" not in by_title["Love Letters"].explanation.reason


def test_free_text_matches_overview_words(small_engine):
    prefs = UserPreferences(free_text="wedding chaos", top_n=3)
    result = small_engine.generate_recommendations(prefs)
    assert titles(result)[0] == "Laugh Riot"
    assert "wedding" in result.recommendations[0].details.matched_keywords


def test_unknown_free_text_is_flagged(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3}, free_text="zzzzz")
    result = small_engine.generate_recommendations(prefs)
    assert any("ignored" in w for w in result.warnings)


def test_cold_start_profile_uses_only_selected_preferences(small_movies):
    profile = build_user_profile(UserPreferences(genres={"Sci-Fi": 3}), small_movies)
    assert profile.is_cold_start
    assert profile.genre_weights == {"Sci-Fi": 3.0}


def test_liked_movie_boosts_profile_and_dislike_removes_item(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3}, top_n=5)
    base = small_engine.generate_recommendations(prefs)
    star_quest = next(r for r in base.recommendations if r.title == "Star Quest")
    dark_orbit = next(r for r in base.recommendations if r.title == "Dark Orbit")

    liked = small_engine.generate_recommendations(prefs, {star_quest.movie_id: config.LIKE_SCORE})
    assert not liked.profile.is_cold_start
    assert "Adventure" in liked.profile.genre_weights  # inferred from the liked movie

    disliked = small_engine.generate_recommendations(
        prefs, {dark_orbit.movie_id: config.DISLIKE_SCORE})
    assert "Dark Orbit" not in titles(disliked)


def test_balanced_mode_can_change_order(small_engine):
    prefs = UserPreferences(genres={"Sci-Fi": 3}, top_n=2, mode=config.MODE_BALANCED)
    result = small_engine.generate_recommendations(prefs)
    for rec in result.recommendations:
        expected = 0.8 * rec.details.preference_similarity + 0.2 * rec.details.rating_score
        assert abs(rec.details.final_score - expected) < 1e-9


def test_popular_fallback_is_sorted_by_rating(real_engine):
    ratings = real_engine.popular_fallback(5)["rating"].tolist()
    assert ratings == sorted(ratings, reverse=True)
