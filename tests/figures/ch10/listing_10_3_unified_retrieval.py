# Figure - Listing 10.3: Unified retrieval
# Source: chapters/ch10.md lines 169-199
# Chapter: 10
# Category: needs-package  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import numpy as np  #A
from sentence_transformers import SentenceTransformer  #A

class HybridRetriever:
  def __init__(self, faiss_index, embeddings, movies_df,
               item_knn=None,
               encoder_name="all-MiniLM-L6-v2"):
    self.index = faiss_index
    self.embeddings = embeddings
    self.movies_df = movies_df
    self.item_knn = item_knn
    self.encoder = SentenceTransformer(encoder_name)

  def search(self, query, user_id=None, k=10, filters=None):  #B
    content_results = self._search_by_text(query, k=k * 2)
    if user_id is not None and self.item_knn is not None:
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

  def _reciprocal_rank_fusion(self, list_a, list_b,
                              k=10, rrf_k=60):  #C
    scores = {}
    all_items = {}
    for rank, item in enumerate(list_a):
      mid = item["movie_id"]
      scores[mid] = 1.0 / (rank + rrf_k)
      all_items[mid] = item
    for rank, item in enumerate(list_b):
      mid = item["movie_id"]
      scores[mid] = scores.get(mid, 0) + 1.0 / (rank + rrf_k)
      if mid not in all_items:
        all_items[mid] = item
    ranked = sorted(scores.items(),
                    key=lambda x: x[1], reverse=True)
    return [all_items[mid] for mid, _ in ranked[:k]]

# Callout annotations (from the book):
#   #A A sentence-transformer encodes queries into the same space as the FAISS item index
#   #B Combine content (FAISS) and collaborative (ItemKNN) signals into one ranked list
#   #C Reciprocal rank fusion with k=60 (Cormack et al.)
