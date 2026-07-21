# Figure — Listing 5.12: Cross-encoder scoring stage
# Source: chapters/ch05.md lines 777-800
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A The cross-encoder model from Section 5.4 or 5.5
#   #B Look up precomputed user features
#   #C Look up precomputed item features
#   #D Score the user-item pair through the cross-encoder
class CrossEncoderScoring(Scoring):
  def __init__(
    self,
    cross_encoder: nn.Module,
    user_features: dict[str, torch.Tensor],
    item_features: dict[str, torch.Tensor],
  ):
    self.cross_encoder = cross_encoder
    self.user_features = user_features
    self.item_features = item_features

  def score(
    self, candidates: list[str], user_id: str
  ) -> list[tuple[str, float]]:
    user_feat = self.user_features[user_id]
    scored = []
    for item_id in candidates:
      item_feat = self.item_features[item_id]
      score = self.cross_encoder(
        user_feat.unsqueeze(0),
        item_feat.unsqueeze(0),
      ).item()
      scored.append((item_id, score))
    return scored
