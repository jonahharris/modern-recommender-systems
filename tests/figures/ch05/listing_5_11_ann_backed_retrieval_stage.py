# Figure - Listing 5.11: ANN-backed retrieval stage
# Source: chapters/ch05.md lines 728-757
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import faiss
import numpy as np

from recsys.fourstage_recsys.item_context import ScoredItem

class ANNRetrieval:
  def __init__(
    self,
    index: faiss.Index,
    item_embeddings: np.ndarray,
    id_to_item: dict[int, str],
    item_to_id: dict[str, int],
  ):
    self.index = index  #A
    self.item_embeddings = item_embeddings
    self.id_to_item = id_to_item
    self.item_to_id = item_to_id

  def retrieve_similar_items(
    self, seed_items: list[str], k: int = 100
  ) -> list[ScoredItem]:
    faiss_ids = [
      self.item_to_id[sid] for sid in seed_items if sid in self.item_to_id
    ]  #B
    if not faiss_ids:
      return []
    query = self.item_embeddings[faiss_ids].mean(axis=0).reshape(1, -1)  #C
    faiss.normalize_L2(query)

    scores, ids = self.index.search(query, k + len(faiss_ids))  #D

    seed_set = set(seed_items)
    candidates = []
    for faiss_id, score in zip(ids[0], scores[0]):
      if faiss_id == -1:
        break
      item_id = self.id_to_item[faiss_id]
      if item_id not in seed_set:  #E
        candidates.append(
          ScoredItem(item_id=item_id, scores={"similarity": float(score)})
        )
    return candidates[:k]

# Callout annotations (from the book):
#   #A The prebuilt FAISS index from Section 5.2
#   #B Look up the seed items' embedding indices in the FAISS index
#   #C Average the seed embeddings and use the result as the query
#   #D Request extra results because the seed items may appear among the neighbors
#   #E Exclude the seed items from the candidate list
