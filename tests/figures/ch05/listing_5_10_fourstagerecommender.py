# Figure - Listing 5.10: FourStageRecommender
# Source: chapters/ch05.md lines 694-714
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.stages.filtering import Filtering
from recsys.fourstage_recsys.stages.scoring import Scorer
from recsys.fourstage_recsys.stages.ordering import Ordering
from recsys.fourstage_recsys.recsys_context import RecommendationContext

class FourStageRecommender:
  def __init__(
    self,
    retrieval: Retrieval,
    filter: Filtering,
    scorer: Scorer,
    ordering: Ordering,
  ):  #A
    self.retrieval = retrieval
    self.filter = filter
    self.scorer = scorer
    self.ordering = ordering

  def recommend(self, context: RecommendationContext, debug=False):
    candidates = self.retrieval.retrieve_similar_items(
      context.seed_items, k=100
    )  #B
    candidates = self.filter.filter(candidates, context)
    candidates = self.scorer.score(candidates, context)
    ordered = self.ordering.order(candidates, context, debug=debug)
    return ordered[:context.k]

# Callout annotations (from the book):
#   #A initialize with instances of each stage
#   #B When called, it will call each step in turn
