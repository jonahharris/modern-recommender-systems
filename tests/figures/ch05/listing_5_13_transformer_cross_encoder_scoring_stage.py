# Figure - Listing 5.13: Transformer cross-encoder scoring stage
# Source: chapters/ch05.md lines 816-838
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class TransformerScoring(Scoring):
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
    self, candidates: list[str], user_id: str
  ) -> list[tuple[str, float]]:
    user_text = self.user_histories[user_id]
    item_texts = [
      self.item_descriptions[item_id]
      for item_id in candidates
    ]  #B
    scores = self.cross_encoder(
      [user_text] * len(candidates), item_texts
    )  #C
    return list(zip(candidates, scores.tolist()))

# Callout annotations (from the book):
#   #A User histories represented as text (Section 5.5.3)
#   #B Collect item descriptions for all candidates
#   #C Score all candidates in a single batched forward pass
