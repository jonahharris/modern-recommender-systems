"""Shared fixtures for the book-listing ("figure") test harness.

Provides a small, deterministic MovieLens-shaped dataset and a seeded execution
namespace so that structural listings (build matrix, similarity, TF-IDF, function
definitions) can run without downloading the real 25M dataset.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# --- synthetic MovieLens-shaped data -------------------------------------------------

_MOVIES = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
    (3, "Grumpier Old Men (1995)", "Comedy|Romance"),
    (4, "Waiting to Exhale (1995)", "Comedy|Drama|Romance"),
    (5, "Heat (1995)", "Action|Crime|Thriller"),
    (6, "The Matrix (1999)", "Action|Sci-Fi|Thriller"),
    (7, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
    (8, "Shrek (2001)", "Adventure|Animation|Children|Comedy|Fantasy|Romance"),
    (9, "Gattaca (1997)", "Drama|Sci-Fi|Thriller"),
    (10, "Back to the Future (1985)", "Adventure|Comedy|Sci-Fi"),
    (11, "Toy Story 2 (1999)", "Adventure|Animation|Children|Comedy|Fantasy"),
    (12, "Ex Machina (2015)", "Drama|Sci-Fi|Thriller"),
]


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(_MOVIES, columns=["movieId", "title", "genres"])


@pytest.fixture
def ratings() -> pd.DataFrame:
    # deterministic pseudo-interactions: 20 users, each rates a rotating window of movies
    rng = np.random.RandomState(42)
    rows = []
    ts = 1_000_000_000
    movie_ids = [m[0] for m in _MOVIES]
    for user_id in range(1, 21):
        n = rng.randint(4, 9)
        chosen = rng.choice(movie_ids, size=n, replace=False)
        for mid in chosen:
            ts += 60
            rows.append((user_id, int(mid), float(rng.randint(3, 6)), ts))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def chapter_namespace(ratings, movies):
    """Fresh globals dict seeded with the imports and data the ch02 listings assume
    already exist in the notebook session."""
    return {
        "__name__": "__figure__",
        "np": np,
        "pd": pd,
        "csr_matrix": csr_matrix,
        "cosine_similarity": cosine_similarity,
        "ratings": ratings.copy(),
        "movies": movies.copy(),
    }
