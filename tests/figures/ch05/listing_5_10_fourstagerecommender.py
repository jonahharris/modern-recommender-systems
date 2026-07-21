# Figure - Listing 5.10: FourStageRecommender
# Source: chapters/ch05.md lines 694-714
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class FourStageRecommender:
  def __init__(
    self,
    retrieval,
    filter: Filtering,
    scorer: Scoring,
    ranker,
  ):  #A
    self.retrieval = retrieval
    self.filter = filter
    self.scorer = scorer
    self.ranker = ranker

  def recommend(self, context: RecommendationContext):
    candidates = self.retrieval.retrieve_similar_items(
      context.seed_movie_id, k=100
    )  #B
    candidates = self.filter.filter(candidates, context.user_id)
    candidates = self.scorer.score(candidates)
    ranked = self.ranker.rank(candidates)
    return ranked[:context.k]

# Callout annotations (from the book):
#   #A initialize with instances of each stage
#   #B When called, it will call each step in turn
