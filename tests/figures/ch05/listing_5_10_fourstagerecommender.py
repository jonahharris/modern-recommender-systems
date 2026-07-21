# Figure — Listing 5.10: FourStageRecommender
# Source: chapters/ch05.md lines 694-714
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A initialize with instances of each stage
#   #B When called, it will call each step in turn
class FourStageRecommender:
  def __init__(
    self,
    retrieval,
    filter: Filtering,
    scorer: Scoring,
    ranker,
  ):
    self.retrieval = retrieval
    self.filter = filter
    self.scorer = scorer
    self.ranker = ranker

  def recommend(self, context: RecommendationContext):
    candidates = self.retrieval.retrieve_similar_items(
      context.seed_movie_id, k=100
    )
    candidates = self.filter.filter(candidates, context.user_id)
    candidates = self.scorer.score(candidates)
    ranked = self.ranker.rank(candidates)
    return ranked[:context.k]
