# SmartRec Demo Guide

## 5-minute review flow

1. **Open the app** (`streamlit run app.py`). Point out the pipeline strip and "Popular starting points (not personalised)" — the cold-start fallback.
2. **Select interests.** Choose the *Space explorer* preset (Sci-Fi, Adventure, *space*), or pick genres manually.
3. **Generate Recommendations.** Show the preference chips and the four metrics.
4. **Show similarity scores.** Each card has a match %, bar and label. Open **Insights** for the ranking table and the score-contribution chart.
5. **Show why.** Read one "Why recommended" box: it names only features the movie really shares with the selection.
6. **Change preferences.** Switch the preset to *Date night* → Generate. The list changes completely.
7. **Prove it.** Open **Compare profiles** (e.g. Space explorer vs Date night): "Titles in both lists" is 0.
8. **Optional feedback.** Click *Like* on a card: its genres/tags get a boost and the ranking updates; *Not for me* removes a title.
9. **Explain the algorithm.** Open **How it works** and the "profile as numbers" expander.

## "How does your recommendation system work?"

1. The user selects preferences (genres with importance, interests, optional description).
2. Those preferences are converted into a numeric vector using TF-IDF weights — rarer features count more.
3. Every movie is represented in the same vocabulary using its genres, tags and overview.
4. Cosine similarity measures the angle between the user vector and each movie vector, block by block, and the blocks are combined with fixed weights.
5. Movies are ranked by that score (optionally 20 % rating in Balanced mode).
6. The Top-N are displayed, skipping anything with no meaningful similarity.
7. The explanation lists the selected genres/interests/keywords that the movie actually has — it is built from the same data that produced the score.

**Likely follow-ups**

- *Is this machine learning?* The TF-IDF vocabulary is fitted from data, but there is no trained predictive model; it is content-based similarity matching, as the project brief asks.
- *Why not collaborative filtering?* It needs a large history of user interactions, which we don't have; content-based works from day one (cold start).
- *Why cosine instead of Euclidean distance?* Cosine compares direction, so a long description doesn't beat a short one by size alone.
- *What does 100 % mean?* The movie's genres are exactly the selected ones in the same proportions — not a guarantee you'll like it.
