# Figure - Listing 5.13: Transformer cross-encoder scoring stage
# Source: chapters/ch05.md lines 816-838
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
from recsys.fourstage_recsys.stages.scoring import Scorer
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

class TransformerScoring(Scorer):
  def __init__(
    self,
    cross_encoder: TransformerCrossEncoder,
    user_histories: dict[str, str],
    item_descriptions: dict[str, str],
  ):
    self.cross_encoder = cross_encoder
    self.user_histories = user_histories  #A
    self.item_descriptions = item_descriptions

  def score(
    self,
    candidates: list[ScoredItem],
    context: RecommendationContext,
  ) -> list[ScoredItem]:
    user_text = self.user_histories[context.user_id]
    item_texts = [
      self.item_descriptions[item.item_id]
      for item in candidates
    ]  #B
    scores = self.cross_encoder(
      [user_text] * len(candidates), item_texts
    )  #C
    for item, relevance in zip(candidates, scores.tolist()):
      item.scores["relevance"] = relevance
    return candidates

# Callout annotations (from the book):
#   #A User histories represented as text (Section 5.5.3)
#   #B Collect item descriptions for all candidates
#   #C Score all candidates in a single batched forward pass
