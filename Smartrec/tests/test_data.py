"""Dataset loading, validation and preprocessing."""
import pandas as pd
import pytest

from src.data_loader import load_movies, load_raw_movies
from src.exceptions import DatasetNotFoundError, DatasetValidationError
from src.preprocessing import clean_movies, normalize_genre


def test_bundled_dataset_loads_cleanly():
    movies, report = load_movies()
    assert len(movies) >= 50
    assert report.rows_kept == len(movies)
    assert {"movie_id", "title", "genres", "tags", "rating"} <= set(movies.columns)


def test_missing_file_raises_friendly_error(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_raw_movies(tmp_path / "nope.csv")


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["A"], "genres": ["Drama"]}).to_csv(path, index=False)
    with pytest.raises(DatasetValidationError) as error:
        load_raw_movies(path)
    assert "rating" in str(error.value)


def test_empty_file_raises_validation_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetValidationError):
        load_raw_movies(path)


def test_duplicates_are_removed_and_reported(raw_movies):
    doubled = pd.concat([raw_movies, raw_movies.iloc[[0]]], ignore_index=True)
    movies, report = clean_movies(doubled)
    assert len(movies) == len(raw_movies)
    assert report.dropped_duplicates == 1
    assert any("duplicate" in m for m in report.messages)


def test_invalid_rows_are_dropped_not_hidden(raw_movies):
    raw_movies.loc[0, "rating"] = 42
    raw_movies.loc[1, "genres"] = ""
    raw_movies.loc[2, "title"] = " "
    movies, report = clean_movies(raw_movies)
    assert len(movies) == 2
    assert report.dropped_invalid_rating == 1
    assert report.dropped_missing_genres == 1
    assert report.dropped_missing_title == 1


def test_missing_overview_is_filled_and_reported(raw_movies):
    raw_movies.loc[0, "overview"] = None
    movies, report = clean_movies(raw_movies)
    assert movies.loc[0, "overview"] == ""
    assert report.missing_overviews == 1


def test_lists_and_genre_names_are_normalised(raw_movies):
    raw_movies.loc[0, "genres"] = "science fiction|Sci-Fi| adventure "
    raw_movies.loc[0, "tags"] = "Space| space |Crew"
    movies, _ = clean_movies(raw_movies)
    assert movies.loc[0, "genres"] == ["Sci-Fi", "Adventure"]
    assert movies.loc[0, "tags"] == ["space", "crew"]


def test_all_rows_invalid_raises(raw_movies):
    raw_movies["rating"] = "bad"
    with pytest.raises(DatasetValidationError):
        clean_movies(raw_movies)


def test_normalize_genre_aliases():
    assert normalize_genre("sci fi") == "Sci-Fi"
    assert normalize_genre("  comedy ") == "Comedy"
