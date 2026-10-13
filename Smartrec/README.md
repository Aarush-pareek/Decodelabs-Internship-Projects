# SmartRec — AI Recommendation System

**A transparent, content-based movie recommender built with Python, scikit-learn and Streamlit.**
Choose your interests, and SmartRec ranks movies by cosine similarity and tells you *why* each one was recommended.

> Industrial Training **Project 3: AI Recommendation Logic** — a *simple* recommendation system based on user preferences.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

SmartRec turns a user's preferences (genres, interests, an optional free-text description) into a numeric profile, compares it with every movie in a small catalogue, and returns a ranked Top-N list with a match percentage and a generated explanation.

## Problem Statement

Catalogues are too large to browse. A recommender reduces choice overload by surfacing the few items that best fit what the user already said they like.

## Project Objective

Create a simple recommendation system based on user preferences that:

1. takes user input (choices / interests),
2. matches preferences using logic and similarity,
3. displays recommended items.

## Features

- Multi-select genres with optional 1–5 **importance** per genre
- Interest tags and an optional free-text description (TF-IDF text matching)
- Filters: minimum rating, language, number of results
- Two modes: **Best match** (similarity only) and **Balanced** (80 % similarity + 20 % rating)
- **Explainable results** generated from the real matched features
- Insights tab: ranking table, score-contribution chart, per-block score breakdown
- **Compare profiles** tab and one-click demo presets: *different preferences → different recommendations*
- Optional **Like / Not for me** feedback (session only)
- Cold-start handling, data validation messages, friendly error handling
- Unit tests for the recommendation engine

## How It Works

```
User Preferences → Feature Representation → Similarity Calculation → Ranking → Recommendations
```

1. **Preferences → profile.** Selected genres (with strengths) and interests become weighted terms.
2. **Feature representation.** Each movie is a TF-IDF vector in three blocks: *genres*, *interest tags*, *overview text*. The user is mapped into the same vocabulary.
3. **Similarity.** Cosine similarity is computed per block (own NumPy implementation in `src/similarity.py`).
4. **Blend.** Only blocks the user filled in count. Base weights: genres 0.5, interests 0.3, text 0.2, re-normalised.
5. **Rank.** Sort by score (ties → higher rating), drop zero-similarity items, keep Top-N.
6. **Explain.** The reason lists only selected features the movie actually has.

## Recommendation Algorithm

**TF-IDF** gives rare features more weight than common ones (*heist* says more about a movie than *drama*).
**Cosine similarity** measures the angle between two vectors:

```
cos(θ) = (A · B) / (‖A‖ ‖B‖)        0 = nothing in common … 1 = same direction
```

```
preference_similarity = Σ  weightᵦ × cosine(user_vectorᵦ, item_vectorᵦ)     (b = active blocks)
final_score (Best match) = preference_similarity
final_score (Balanced)   = 0.8 × preference_similarity + 0.2 × (rating / 10)
```

> **Reading the percentage:** *Match %* is the final score × 100. A genre-only profile gives 100 % to a movie whose genres are exactly the ones selected. It describes how well a movie fits your stated preferences — not a prediction that you will enjoy it.

**Cold start:** a new user has no history, so the profile is built only from the selected preferences. Nothing is "learned". Likes during a session add a small boost to the liked movie's genres/tags — a simple profile update, not a trained model.

## Technology Stack

Python · pandas · NumPy · scikit-learn (`TfidfVectorizer` only; similarity and ranking are hand-written) · Streamlit · Plotly · pytest

## Dataset

`data/movies.csv` — 103 movies with `title, year, genres, overview, rating, language, tags` (`|`-separated lists). Overviews are short original summaries; **ratings are rounded, approximate values included for demonstration**. The file is bundled, so nothing needs downloading. To use another dataset or domain, provide a CSV with the same columns (see `src/config.py`).

## Project Architecture

```
smartrec/
├── app.py                      # Streamlit UI
├── data/movies.csv             # bundled sample dataset
├── src/
│   ├── config.py               # paths, weights, thresholds
│   ├── exceptions.py           # friendly custom errors
│   ├── models.py               # dataclasses (preferences, profile, results)
│   ├── data_loader.py          # load + validate CSV
│   ├── preprocessing.py        # cleaning + CleaningReport
│   ├── feature_engineering.py  # TF-IDF feature space
│   ├── similarity.py           # cosine similarity + score blending
│   ├── recommendation_engine.py# profile → score → rank → Top-N
│   └── explanation.py          # explanation text from matched features
├── utils/                      # HTML components, CSS, helpers
├── tests/                      # pytest suite
├── .streamlit/config.toml      # light theme
├── requirements.txt · README.md · PROJECT_REPORT.md · DEMO_GUIDE.md · LICENSE
```

## Installation

```bash
git clone <repository-url>
cd smartrec

python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest
```

## Example Results

Real output of the engine on the bundled dataset (all genres at importance 3, Best-match mode):

| Preferences | #1 | #2 | #3 |
|---|---|---|---|
| Action, Sci-Fi, Adventure | Star Wars: A New Hope (100 %) | Avengers: Endgame (100 %) | Mad Max: Fury Road (100 %) |
| Romance, Comedy | Amelie (100 %) | Crazy Rich Asians (100 %) | Bridesmaids (100 %) |
| Thriller, Mystery, Crime + *twist* | Shutter Island (64 %) | Memento (64 %) | Se7en (62 %) |

Example explanation: *"It matches your genres Mystery and Thriller and interests twist."*

## Screenshots

| Recommendations | Insights | Compare profiles |
|---|---|---|
| _add `docs/screenshots/recommendations.png`_ | _add `docs/screenshots/insights.png`_ | _add `docs/screenshots/compare.png`_ |

## Internship Requirement Mapping

| Official Requirement | Project Implementation |
|---|---|
| Take user input | User selects interests/preferences through the UI |
| Match preferences using logic/similarity | Similarity-based recommendation engine (TF-IDF + cosine similarity) |
| Display recommended items | Ranked recommendation cards |
| Logic building | Preference matching and scoring |
| Pattern matching | Feature/similarity matching |
| Recommendation concepts | Personalized Top-N recommendation |

This is a **simple content-based** recommender. It is not a neural or collaborative-filtering system.

## Future Improvements

- Larger catalogue (e.g. a public movie dataset) with sparse matrices for scale
- Persistent user profiles and richer feedback (ratings 1–5)
- Evaluation with held-out user data (precision@k) once real interactions exist
- Hybrid approach combining content-based and collaborative signals
- Deployment on Streamlit Community Cloud

## Author

- **Name:** `<your name>`
- **GitHub:** `<github profile url>`
- **LinkedIn:** `<linkedin url>`
- **Email:** `<email>`

## GitHub Repository Suggestions

- **Name:** `smartrec-recommendation-system`
- **Description:** Explainable content-based movie recommender using TF-IDF and cosine similarity, built with Python and Streamlit.
- **Topics:** `recommendation-system` `content-based-filtering` `cosine-similarity` `tf-idf` `scikit-learn` `streamlit` `machine-learning` `python` `explainable-ai` `internship-project`
