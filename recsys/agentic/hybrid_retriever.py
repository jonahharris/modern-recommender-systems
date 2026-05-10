"""
Hybrid retriever that combines content-based (FAISS) and 
collaborative (ItemKNN) retrieval with reciprocal rank fusion.

Composes the existing retrieval classes from earlier chapters
rather than reimplementing them.
"""

import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class HybridRetriever:

  def __init__(self, 
               faiss_index,
               embeddings,
               movies_df,
               item_knn=None,
               encoder_name="all-MiniLM-L6-v2"):
    """
    Args:
      faiss_index: Trained FAISS index from Chapter 5
      embeddings: numpy array of item embeddings (384-dim)
      movies_df: DataFrame with movieId, title, genres columns
      item_knn: Optional ItemKNNRetrieval from Chapter 3
      encoder_name: Sentence transformer for query encoding
    """
    self.index = faiss_index                           #A
    self.embeddings = embeddings
    self.movies_df = movies_df
    self.item_knn = item_knn
    self.encoder = SentenceTransformer(encoder_name)   #B

    # Build lookup tables
    self._movie_id_to_idx = {
      int(row["movieId"]): i 
      for i, row in movies_df.iterrows()
    }
    self._idx_to_movie = {
      i: int(row["movieId"]) 
      for i, row in movies_df.iterrows()
    }
    self._movie_id_to_title = dict(
      zip(movies_df["movieId"].astype(int), 
          movies_df["title"])
    )

  def search(self, query, user_id=None, k=10, 
             filters=None):
    """Unified search combining content and collaborative.
    
    Args:
      query: Natural language search query
      user_id: Optional user ID for collaborative signal
      k: Number of results to return
      filters: Optional dict with genre, year_min, year_max
      
    Returns:
      List of dicts with title, movie_id, score, genres
    """
    content_results = self._search_by_text(
      query, k=k * 2
    )

    if user_id is not None and self.item_knn is not None:
      # Get a seed item from content results for ItemKNN
      collab_results = self._search_by_collaborative(
        user_id, k=k * 2
      )
      merged = self._reciprocal_rank_fusion(
        content_results, collab_results, k=k
      )
    else:
      merged = content_results[:k]

    if filters:
      merged = self._apply_filters(merged, filters)

    return merged[:k]

  def _search_by_text(self, query, k=20):
    """Content-based retrieval via FAISS."""
    query_embedding = self.encoder.encode(query)       #C
    query_vector = np.array(
      [query_embedding], dtype=np.float32
    )

    distances, indices = self.index.search(
      query_vector, k
    )

    results = []
    for rank, (dist, idx) in enumerate(
      zip(distances[0], indices[0])
    ):
      if idx < 0 or idx >= len(self.movies_df):
        continue
      row = self.movies_df.iloc[idx]
      results.append({
        "title": row["title"],
        "movie_id": int(row["movieId"]),
        "score": float(1.0 / (1.0 + dist)),            #D
        "genres": row.get("genres", ""),
        "year": self._extract_year(row.get("title", "")),
        "overview": row.get("overview", ""),
        "source": "content"
      })
    return results

  def _search_by_collaborative(self, user_id, k=20):
    """Collaborative retrieval via ItemKNN."""
    if self.item_knn is None:
      return []

    # Get user's most recent item for seed
    from recsys.fourstage_recsys.filtering.history_filtering import HistoryFiltering
    
    # Use ItemKNN's stored data to find user items
    if hasattr(self.item_knn, 'ratings'):
      user_rows = self.item_knn.ratings[
        self.item_knn.ratings['userId'] == user_id
      ]
      if user_rows.empty:
        return []
      
      # Use highest-rated item as seed
      seed_movie = user_rows.sort_values(
        'rating', ascending=False
      ).iloc[0]['movieId']
      
      knn_results = self.item_knn.retrieve_similar_items(
        int(seed_movie), k=k
      )
      
      results = []
      for item in knn_results:
        mid = item['movie_id']
        title = self._movie_id_to_title.get(mid, "Unknown")
        row_matches = self.movies_df[
          self.movies_df['movieId'] == mid
        ]
        genres = ""
        if not row_matches.empty:
          genres = row_matches.iloc[0].get("genres", "")
        
        results.append({
          "title": title,
          "movie_id": mid,
          "score": item['similarity'],
          "genres": genres,
          "year": self._extract_year(title),
          "overview": "",
          "source": "collaborative"
        })
      return results
    return []

  def _reciprocal_rank_fusion(self, list_a, list_b, 
                               k=10, rrf_k=60):
    """Merge two ranked lists using RRF.
    
    Args:
      rrf_k: RRF constant (default 60, from Cormack et al.)
    """
    scores = {}
    all_items = {}

    for rank, item in enumerate(list_a):
      mid = item["movie_id"]
      scores[mid] = 1.0 / (rank + rrf_k)              #E
      all_items[mid] = item

    for rank, item in enumerate(list_b):
      mid = item["movie_id"]
      scores[mid] = scores.get(mid, 0) + 1.0 / (rank + rrf_k)
      if mid not in all_items:
        all_items[mid] = item

    ranked = sorted(
      scores.items(), key=lambda x: x[1], reverse=True
    )
    return [all_items[mid] for mid, _ in ranked[:k]]

  def _apply_filters(self, results, filters):
    """Post-retrieval metadata filtering."""
    filtered = results
    if "genre" in filters:
      genre = filters["genre"].lower()
      filtered = [
        r for r in filtered 
        if genre in r.get("genres", "").lower()
      ]
    if "year_min" in filters:
      filtered = [
        r for r in filtered 
        if r.get("year") and r["year"] >= filters["year_min"]
      ]
    if "year_max" in filters:
      filtered = [
        r for r in filtered 
        if r.get("year") and r["year"] <= filters["year_max"]
      ]
    return filtered

  @staticmethod
  def _extract_year(title):
    """Extract year from title like 'Toy Story (1995)'."""
    import re
    match = re.search(r'\((\d{4})\)', str(title))
    return int(match.group(1)) if match else None

  def add_item(self, title, genres, overview, movie_id):
    """Add a new item to the index. No retraining needed."""
    text = f"{title} {genres} {overview}"
    embedding = self.encoder.encode(text)
    embedding = np.array(
      [embedding], dtype=np.float32
    )
    self.index.add(embedding)                          #F
    
    new_row = {
      "movieId": movie_id, 
      "title": title,
      "genres": genres, 
      "overview": overview
    }
    import pandas as pd
    self.movies_df = pd.concat(
      [self.movies_df, pd.DataFrame([new_row])],
      ignore_index=True
    )
    self._movie_id_to_title[movie_id] = title

#A Reuse the FAISS index built in Chapter 5
#B Same encoder as Chapter 6 for embedding consistency
#C Encode the natural-language query into item embedding space
#D Convert distance to similarity score
#E Reciprocal rank fusion with k=60
#F New items are immediately searchable
