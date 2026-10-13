# SmartRec — Project Report

**Industrial Training Project 3: AI Recommendation Logic**
Author: `<your name>` · Date: `<date>`

---

## 1. Introduction

SmartRec is a content-based movie recommendation system. It converts a user's stated preferences into a numeric profile, compares that profile with every movie in a catalogue using cosine similarity, ranks the results and explains each recommendation. The project is intentionally small and transparent: its purpose is to demonstrate the fundamentals of preference matching, similarity scoring and ranking.

## 2. Problem Statement

Digital catalogues contain more items than a person can browse. Without help, users face choice overload. A recommender reduces this by selecting the items that best match what the user has said they like.

## 3. Objective

As specified in the project brief, build a simple recommendation system based on user preferences that (a) takes user input, (b) matches preferences using logic or similarity, and (c) displays recommended items. The evaluated skills are logic building, pattern matching and recommendation concepts.

## 4. Recommendation System Concept

Recommender systems are commonly grouped into **collaborative filtering** (patterns across many users' behaviour) and **content-based filtering** (matching user preferences to item attributes). SmartRec uses content-based filtering only. This works immediately with no interaction history, which suits a new user with only a few stated preferences.

## 5. Dataset

- File: `data/movies.csv`, 103 movies, 16 genres, 148 distinct tags, 8 languages.
- Columns: `title, year, genres, overview, rating, language, tags`. Genres and tags are `|`-separated.
- Overviews are short original summaries. Ratings are rounded, approximate values included for demonstration and are not an official data export.
- The dataset is bundled so the project runs without downloads. The code reads the required columns from `src/config.py`, so another dataset with the same columns can be substituted.

## 6. Data Preprocessing

`src/data_loader.py` and `src/preprocessing.py`:

1. Check the file exists and parses; check all required columns are present (otherwise a clear `DatasetValidationError` lists what is missing).
2. Strip text; split genre and tag lists; normalise genre spelling (e.g. "science fiction" → "Sci-Fi"); lowercase and de-duplicate tags.
3. Drop rows with a missing title, no genres, or a rating that is missing or outside 0–10.
4. Remove duplicates (same title and year).
5. Fill missing overviews with an empty string and mark missing years as empty.
6. Return a `CleaningReport`. Every drop or repair is counted and shown in the *Dataset* tab, so nothing is hidden. The bundled file needs no repairs.

## 7. Feature Engineering

`src/feature_engineering.py` builds three TF-IDF blocks over the same catalogue:

| Block | Source | Purpose |
|---|---|---|
| genres | genre list | broad taste |
| interests | tag list | specific themes (e.g. *heist*, *space*) |
| text | overview + tags + genres, English stop words removed | free-text descriptions |

**TF-IDF** weights each term by how often it appears in an item (TF) and how rare it is across the catalogue (IDF), so rare terms contribute more than common ones. The user is mapped into the *same* vocabularies, which is essential for the similarity maths to be meaningful.

## 8. User Preference Processing

`build_user_profile()` (cold start): selected genres become weights equal to the chosen importance (1–5); selected interests get weight 1; free text is kept as text. Each weight is multiplied by the term's IDF when vectorised. If the user supplied nothing, an `EmptyPreferencesError` is raised and the UI shows a friendly message.

If the user has liked titles during the session, their genres and tags receive a fixed boost (+1). This is explicit feedback applied to the profile; the system does not claim to have learned behaviour beyond that.

## 9. Similarity Calculation

`src/similarity.py` implements cosine similarity directly with NumPy:

`cos(θ) = (A · B) / (‖A‖ ‖B‖)`

Zero vectors return 0 instead of causing a division error, and mismatched sizes raise a `SimilarityError`. Because all TF-IDF values are non-negative, scores lie between 0 and 1.

Cosine similarity compares the direction of vectors rather than their length, so an item with a long description is not penalised for size alone.

## 10. Recommendation Algorithm

```
preferences → user profile → block vectors → cosine per block
→ weighted blend → (optional rating term) → rank → Top-N → explanation
```

- **Preference similarity** = Σ weightᵦ × cosineᵦ over the blocks the user filled in. Base weights are genres 0.5, interests 0.3, text 0.2, re-normalised over active blocks.
- **Best match mode:** final score = preference similarity.
- **Balanced mode:** final score = 0.8 × preference similarity + 0.2 × (rating / 10). The rating term is optional and was added only to let users trade pure similarity for quality.

The weights are design choices fixed in `src/config.py`; they were not tuned on user data.

## 11. Ranking

`rank_recommendations()` sorts eligible items by score (descending), breaking ties by rating. Items below a similarity of 0.01, below the minimum rating, outside the chosen languages, or marked "Not for me" are excluded. If fewer than N items qualify, fewer are returned and a warning explains why; irrelevant items are not added to fill the list.

## 12. Explainable Recommendations

`explain_recommendation()` builds the explanation from `MatchDetails`, which the engine fills from the actual comparison: genres and interests that are both selected by the user *and* present on the movie, and text keywords that have non-zero weight in both vectors. The label (Strong / Good / Moderate / Partial match) is derived from the final score. Genres the movie does not have are never mentioned (covered by tests). In Balanced mode the text states that the rating raised the rank.

## 13. User Interface

Streamlit dashboard with a sidebar for controls and demo presets, and tabs for *Recommendations*, *Insights*, *How it works*, *Compare profiles* and *Dataset*. Recommendation cards show title, year, genres, rating, match %, matching features, overview and "Why recommended". The Insights tab shows the ranking table, a stacked bar chart of block contributions and a score breakdown for the top result. Item vectors are computed once with `st.cache_resource`; the cleaned dataset is cached with `st.cache_data`.

## 14. Results

Output of the engine on the bundled dataset (all genres at importance 3, Best match):

| Preferences | Top results |
|---|---|
| Action, Sci-Fi, Adventure | Star Wars: A New Hope, Avengers: Endgame, Mad Max: Fury Road, Black Panther (all 100 %), The Matrix (85 %) |
| Romance, Comedy | Amelie, Crazy Rich Asians, Bridesmaids (100 %), Forrest Gump (92 %), Groundhog Day (77 %) |
| Thriller, Mystery, Crime + *twist* | Shutter Island (64 %), Memento (64 %), Se7en (62 %), Gone Girl (61 %), Oldboy (60 %) |

For the five demo presets (Space explorer, Date night, Edge of your seat, Family movie night, True-story drama) the Top-5 lists share no titles with each other, which demonstrates that different preferences produce different recommendations.

A 100 % match means the movie's genres are exactly the selected genres in the same proportions; it is a similarity score, not a probability of enjoyment. No user study or accuracy benchmark has been performed.

## 15. Testing

The pytest suite (`tests/`) contains 41 tests focused on the engine rather than the UI:

- **Data:** loading, missing file, missing columns, empty file, duplicates, invalid rows, normalisation
- **Features:** matrix shapes, shared vocabulary, rare-term weighting, unknown terms
- **Similarity:** identical/orthogonal vectors, a hand-calculated value, scale invariance, zero vectors, size mismatch, block blending, balanced mode
- **Recommendations:** correct ranking, descending order, Top-N count, no duplicates, different preferences → different results, empty and invalid input, no padding with irrelevant items, filters, explanation accuracy, free text, cold start, like/dislike feedback

Run with `pytest`. Record the result of your own run here: `<N passed in X s>`.

## 16. Limitations

- Small catalogue (103 titles) and short descriptions limit recommendation variety; several movies tie at 100 % for genre-only profiles.
- Text matching is literal: no stemming or synonyms ("spaceship" does not match "space").
- Block weights are fixed by design, not learned or validated on users.
- Feedback exists only for the current session and is a simple profile boost.
- Ratings in the dataset are approximate; there is no popularity or recency signal.
- Content-based filtering tends to recommend more of the same and cannot discover cross-user patterns.

## 17. Future Improvements

Larger public dataset with sparse matrices; persistent profiles; 1–5 star feedback; offline evaluation (precision@k) when real interaction data exists; stemming/synonym handling; a hybrid with collaborative filtering; deployment on Streamlit Community Cloud.

## 18. Conclusion

SmartRec meets the brief: it takes user preferences, matches them with a transparent TF-IDF and cosine-similarity pipeline, and displays ranked, explained recommendations. The design favours clarity — every score can be traced to specific features — over complexity, which makes it easy to demonstrate, test and extend.
