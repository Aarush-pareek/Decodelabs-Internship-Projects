"""SmartRec - AI-Powered Personalized Recommendation System (Streamlit UI)."""
from collections import Counter

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_movies
from src.exceptions import DatasetError, SmartRecError
from src.models import RecommendationResult, UserPreferences
from src.recommendation_engine import RecommendationEngine
from utils.components import card_html, hero_html, steps_html
from utils.helpers import chips_html, format_percent, truncate
from utils.styles import CSS

st.set_page_config(page_title="SmartRec", page_icon=None, layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

PRESETS = {
    "Custom": None,
    "Space explorer": {"genres": ["Sci-Fi", "Adventure"], "interests": ["space"]},
    "Date night": {"genres": ["Romance", "Comedy"], "interests": ["love story"]},
    "Edge of your seat": {"genres": ["Thriller", "Mystery", "Crime"], "interests": ["twist"]},
    "Family movie night": {"genres": ["Animation", "Family"], "interests": ["friendship"]},
    "True-story drama": {"genres": ["Drama", "War"], "interests": ["true story"]},
}


# ---------------------------------------------------------------- data/engine
@st.cache_data(show_spinner=False)
def load_data():
    """Load and clean the dataset once per session."""
    return load_movies()


@st.cache_resource(show_spinner=False)
def get_engine() -> RecommendationEngine:
    """Build item vectors once and reuse them for every interaction."""
    movies, _ = load_data()
    return RecommendationEngine(movies)


try:
    _, cleaning_report = load_data()
    engine = get_engine()
except DatasetError as exc:
    st.error(f"SmartRec could not load its dataset. {exc}")
    st.info("Make sure data/movies.csv exists and has the required columns.")
    st.stop()


# --------------------------------------------------------------- session state
DEFAULT_STATE = {
    "preset": "Custom", "sel_genres": [], "sel_interests": [], "free_text": "",
    "min_rating": 6.0, "sel_languages": [], "top_n": config.DEFAULT_TOP_N,
    "mode": config.MODE_BEST_MATCH, "prefs": None, "feedback": {},
}
for key, value in DEFAULT_STATE.items():
    st.session_state.setdefault(key, value)


def apply_preset() -> None:
    """Fill the sidebar controls from a demo preset."""
    preset = PRESETS.get(st.session_state.preset)
    if not preset:
        return
    st.session_state.sel_genres = list(preset["genres"])
    st.session_state.sel_interests = [i for i in preset["interests"]
                                      if i in engine.available_interests]
    st.session_state.free_text = ""
    for genre in engine.available_genres:
        st.session_state.pop(f"strength_{genre}", None)


def set_feedback(movie_id: int, score: int) -> None:
    st.session_state.feedback[movie_id] = score


def preferences_from_sidebar() -> UserPreferences:
    genres = {g: float(st.session_state.get(f"strength_{g}", config.DEFAULT_STRENGTH))
              for g in st.session_state.sel_genres}
    return UserPreferences(
        genres=genres, interests=list(st.session_state.sel_interests),
        free_text=st.session_state.free_text, min_rating=float(st.session_state.min_rating),
        languages=list(st.session_state.sel_languages), top_n=int(st.session_state.top_n),
        mode=st.session_state.mode,
    )


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand">Smart<span>Rec</span></div>'
                '<div class="tagline">Personalized Recommendation Engine</div>',
                unsafe_allow_html=True)
    st.selectbox("Demo preset", list(PRESETS), key="preset", on_change=apply_preset,
                 help="Quickly load a sample profile, then tweak it.")
    st.multiselect("Genres", engine.available_genres, key="sel_genres")
    if st.session_state.sel_genres:
        with st.expander("Genre importance (1-5)"):
            for genre in st.session_state.sel_genres:
                st.session_state.setdefault(f"strength_{genre}", int(config.DEFAULT_STRENGTH))
                st.slider(genre, 1, 5, key=f"strength_{genre}")
    st.multiselect("Interests", engine.available_interests, key="sel_interests")
    st.text_input("Describe what you want (optional)", key="free_text",
                  placeholder="e.g. spaceship survival")
    st.slider("Minimum rating", 0.0, 10.0, step=0.5, key="min_rating")
    st.multiselect("Language (empty = any)", engine.available_languages, key="sel_languages")
    st.slider("Number of recommendations", 3, 10, key="top_n")
    st.radio("Recommendation mode", config.MODES, key="mode",
             help="Best match = similarity only. Balanced = 80% similarity + 20% rating.")
    generate = st.button("Generate Recommendations", type="primary")

if generate:
    candidate = preferences_from_sidebar()
    if candidate.has_signal():
        st.session_state.prefs = candidate
        st.session_state.feedback = {}
    else:
        st.sidebar.warning("Choose at least one genre or interest, or describe what you want.")


# ------------------------------------------------------------------ renderers
def render_landing() -> None:
    st.markdown(hero_html(), unsafe_allow_html=True)
    st.markdown(steps_html(), unsafe_allow_html=True)
    st.info("Choose your preferences in the sidebar (or load a demo preset) and press "
            "**Generate Recommendations**.")
    st.markdown('<div class="section-label">Popular starting points (not personalised)</div>',
                unsafe_allow_html=True)
    for _, row in engine.popular_fallback(5).iterrows():
        st.markdown(f"- **{row['title']}** &middot; {', '.join(row['genres'])} &middot; "
                    f"rating {row['rating']:.1f}", unsafe_allow_html=True)


def render_summary(prefs: UserPreferences, result: RecommendationResult) -> None:
    st.markdown('<div class="section-label">Your preferences</div>', unsafe_allow_html=True)
    genre_chips = [f"{g} ({int(s)}/5)" for g, s in prefs.genres.items()]
    html = chips_html(genre_chips, "chip chip-pref") + chips_html(prefs.interests, "chip chip-pref")
    if prefs.free_text.strip():
        html += chips_html([f'"{prefs.free_text.strip()}"'], "chip chip-pref")
    st.markdown(html, unsafe_allow_html=True)
    if result.profile.is_cold_start:
        st.caption("Cold start: this profile is built only from the preferences you selected.")
    else:
        st.caption(f"Profile also uses {result.profile.liked_count} title(s) you liked.")


def render_metrics(prefs: UserPreferences, result: RecommendationResult) -> None:
    recs = result.recommendations
    selected = set(prefs.genres) | set(prefs.interests)
    matched = {f for r in recs for f in r.details.matched_genres + r.details.matched_interests}
    genre_counts = Counter(g for r in recs for g in r.genres)
    top_category = genre_counts.most_common(1)[0][0] if genre_counts else "-"
    avg = sum(r.details.final_score for r in recs) / len(recs) if recs else 0.0
    cols = st.columns(4)
    cols[0].metric("Average match", format_percent(avg))
    cols[1].metric("Preferences matched", f"{len(matched)} of {len(selected)}")
    cols[2].metric("Top category", top_category)
    cols[3].metric("Recommendations", len(recs))


def render_cards(result: RecommendationResult) -> None:
    for message in result.warnings:
        st.warning(message)
    if not result.recommendations:
        st.error("No titles matched. Try more genres, a lower minimum rating, or any language.")
        return
    feedback = st.session_state.feedback
    for rec in result.recommendations:
        st.markdown(card_html(rec, feedback.get(rec.movie_id) == config.LIKE_SCORE),
                    unsafe_allow_html=True)
        left, right, _ = st.columns([1, 1.4, 6])
        left.button("Like", key=f"like_{rec.movie_id}", on_click=set_feedback,
                    args=(rec.movie_id, config.LIKE_SCORE))
        right.button("Not for me", key=f"dislike_{rec.movie_id}", on_click=set_feedback,
                     args=(rec.movie_id, config.DISLIKE_SCORE))
    st.caption("Feedback is kept for this session only: likes boost similar genres/tags, "
               "and 'Not for me' removes a title. It is a simple profile update, "
               "not a trained model.")


def render_insights(result: RecommendationResult) -> None:
    import plotly.express as px

    recs = result.recommendations
    if not recs:
        st.info("Generate recommendations to see insights.")
        return
    table = pd.DataFrame({
        "Rank": [r.rank for r in recs], "Title": [r.title for r in recs],
        "Match %": [r.match_percent for r in recs],
        "Preference similarity %": [round(r.details.preference_similarity * 100) for r in recs],
        "Rating": [r.rating for r in recs],
    })
    st.markdown("#### Ranking")
    st.dataframe(table, hide_index=True)

    rows = []
    for r in recs:
        for block, value in r.details.contributions.items():
            rows.append({"Title": r.title, "Block": block.capitalize(),
                         "Contribution": round(value * 100, 1)})
    if rows:
        st.markdown("#### What contributed to each similarity score")
        fig = px.bar(pd.DataFrame(rows), x="Contribution", y="Title", color="Block",
                     orientation="h", template="simple_white",
                     color_discrete_sequence=["#4F46E5", "#94A3B8", "#C7D2FE"])
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title="Points of preference similarity")
        fig.update_layout(height=60 + 42 * len(recs), margin=dict(l=0, r=0, t=10, b=0),
                          legend_title_text="")
        st.plotly_chart(fig)

    top = recs[0]
    st.markdown(f"#### Score breakdown for #1: {top.title}")
    breakdown = pd.DataFrame([
        {"Block": b.capitalize(), "Cosine similarity": round(s, 3),
         "Weight": round(top.details.block_weights[b], 2),
         "Contribution": round(top.details.block_weights[b] * s, 3)}
        for b, s in top.details.block_scores.items()
    ])
    st.dataframe(breakdown, hide_index=True)
    st.caption("Preference similarity = sum of contributions = "
               f"{top.details.preference_similarity:.3f}")


def render_how_it_works(result: RecommendationResult | None) -> None:
    st.markdown("""
**Pipeline:** preferences → user profile → vectors → cosine similarity → relevance score → ranking → Top-N

1. **Preferences become a profile.** Selected genres (with 1-5 importance) and interests
   become weighted terms. Cold start: nothing else is assumed about you.
2. **Items and users share one vocabulary.** Every movie is a TF-IDF vector over genres,
   interest tags, and overview text. Rare terms (e.g. *heist*) weigh more than common ones.
3. **Cosine similarity** compares the angle between your vector and each movie's vector:
   `cos(θ) = (A · B) / (‖A‖ ‖B‖)`, from 0 (nothing in common) to 1 (same direction).
4. **Blend.** Only blocks you filled in count: genres 50%, interests 30%, description 20%
   (re-normalised). *Balanced* mode adds 20% weight for the movie's rating.
5. **Rank and cut.** Sort by score, drop zero-similarity titles, keep the Top-N.
6. **Explain.** The reason text lists only features you selected that the movie really has.
""")
    if result and result.recommendations:
        with st.expander("Your profile as numbers (genre block)"):
            vector = engine.features.vectorize_genres(result.profile.genre_weights)
            vocab = engine.features.genre_vocabulary
            rows = [{"Genre": vocab[i], "Strength x IDF": round(float(v), 3)}
                    for i, v in enumerate(vector) if v > 0]
            st.dataframe(pd.DataFrame(rows), hide_index=True)


def render_compare() -> None:
    st.markdown("Same engine, different preferences, different recommendations.")
    names = [n for n in PRESETS if PRESETS[n]]
    left_col, right_col = st.columns(2)
    picks = []
    for column, default, key in ((left_col, 0, "cmp_a"), (right_col, 1, "cmp_b")):
        picks.append(column.selectbox("Profile", names, index=default, key=key))
    lists = []
    for column, name in zip((left_col, right_col), picks):
        preset = PRESETS[name]
        prefs = UserPreferences(
            genres={g: config.DEFAULT_STRENGTH for g in preset["genres"]},
            interests=[i for i in preset["interests"] if i in engine.available_interests],
            top_n=5)
        recs = engine.generate_recommendations(prefs).recommendations
        lists.append({r.title for r in recs})
        for r in recs:
            column.markdown(f"**{r.rank}. {r.title}** &middot; {r.match_percent}%")
    st.metric("Titles in both lists", len(lists[0] & lists[1]))


def render_dataset() -> None:
    movies = engine.movies
    cols = st.columns(4)
    cols[0].metric("Movies", len(movies))
    cols[1].metric("Genres", len(engine.available_genres))
    cols[2].metric("Interest tags", len(engine.features.interest_vocabulary))
    cols[3].metric("Languages", len(engine.available_languages))
    if cleaning_report.messages:
        for message in cleaning_report.messages:
            st.warning(message)
    else:
        st.success(f"Data quality check passed: {cleaning_report.rows_kept} of "
                   f"{cleaning_report.rows_loaded} rows kept, nothing dropped or repaired.")
    view = pd.DataFrame({
        "Title": movies["title"], "Year": movies["year"],
        "Genres": movies["genres"].map(", ".join), "Rating": movies["rating"],
        "Language": movies["language"], "Overview": movies["overview"].map(truncate),
    })
    st.dataframe(view, hide_index=True)
    st.caption("Ratings are rounded, approximate values included for demonstration.")


# ------------------------------------------------------------------------ main
prefs = st.session_state.prefs
result = None
if prefs is not None:
    try:
        result = engine.generate_recommendations(prefs, st.session_state.feedback)
    except SmartRecError as exc:
        st.error(str(exc))
    except Exception:  # last-resort guard: never show a raw traceback to users
        st.error("Something went wrong while calculating recommendations. "
                 "Please change your preferences and try again.")

if prefs is None:
    render_landing()
    tab_names = ["How it works", "Compare profiles", "Dataset"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_how_it_works(None)
    with tabs[1]:
        render_compare()
    with tabs[2]:
        render_dataset()
else:
    st.markdown(hero_html(), unsafe_allow_html=True)
    tabs = st.tabs(["Recommendations", "Insights", "How it works", "Compare profiles", "Dataset"])
    with tabs[0]:
        if result is not None:
            render_summary(prefs, result)
            render_metrics(prefs, result)
            render_cards(result)
    with tabs[1]:
        if result is not None:
            render_insights(result)
    with tabs[2]:
        render_how_it_works(result)
    with tabs[3]:
        render_compare()
    with tabs[4]:
        render_dataset()
