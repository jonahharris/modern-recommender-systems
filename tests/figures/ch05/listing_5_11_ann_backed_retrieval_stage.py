# Figure — Listing 5.11: ANN-backed retrieval stage
# Source: chapters/ch05.md lines 728-757
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A The prebuilt FAISS index from Section 5.2
#   #B Look up the seed item's embedding and use it as the query
#   #C Request k+1 results because the seed item itself will be in the results
#   #D Exclude the seed item from the candidate list
class ANNRetrieval:
  def __init__(
    self,
    index: faiss.Index,
    item_embeddings: np.ndarray,
    id_to_item: dict[int, str],
    item_to_id: dict[str, int],
  ):
    self.index = index
    self.item_embeddings = item_embeddings
    self.id_to_item = id_to_item
    self.item_to_id = item_to_id

  def retrieve_similar_items(
    self, seed_movie_id: str, k: int = 100
  ) -> list[str]:
    faiss_id = self.item_to_id[seed_movie_id]
    query = self.item_embeddings[faiss_id].reshape(1, -1)
    faiss.normalize_L2(query)

    scores, ids = self.index.search(query, k + 1)

    results = []
    for faiss_id, score in zip(ids[0], scores[0]):
      if faiss_id == -1:
        break
      item_id = self.id_to_item[faiss_id]
      if item_id != seed_movie_id:
        results.append(item_id)
    return results[:k]
