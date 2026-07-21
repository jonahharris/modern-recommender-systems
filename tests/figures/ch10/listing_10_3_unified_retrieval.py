# Figure - Listing 10.3: Unified retrieval
# Source: chapters/ch10.md lines 169-199
# Chapter: 10
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from recsys.retrieval import ContentRetriever  #A
from recsys.retrieval import CollaborativeRetriever  #A

class HybridRetriever:
  def __init__(self, content_retriever,
               collab_retriever):
    self.content = content_retriever
    self.collab = collab_retriever

  def search(self, query, user_id=None, k=10):
    content_results = self.content.search(query, k=k*2)
    if user_id is not None:
      collab_results = self.collab.recommend(
        user_id, k=k*2
      )
      return self._merge(content_results,
                         collab_results, k)
    return content_results[:k]

  def _merge(self, content, collab, k):  #B
    scores = {}
    for rank, item in enumerate(content):
      scores[item["movie_id"]] = 1.0 / (rank + 60)  #C
    for rank, item in enumerate(collab):
      mid = item["movie_id"]
      scores[mid] = scores.get(mid, 0) + 1.0 / (rank + 60)
    merged = sorted(scores.items(),
                    key=lambda x: x[1], reverse=True)
    all_items = {i["movie_id"]: i
                 for i in content + collab}
    return [all_items[mid] for mid, _ in merged[:k]]

# Callout annotations (from the book):
#   #A Reuse the retriever classes from Chapter 3
#   #B merge content for two sources
#   #C Reciprocal rank fusion with k=60 (Cormack et al.)
