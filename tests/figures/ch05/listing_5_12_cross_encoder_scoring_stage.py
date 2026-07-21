# Figure - Listing 5.12: Cross-encoder scoring stage
# Source: chapters/ch05.md lines 777-800
# Chapter: 5
# Category: api-drift  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import torch
import torch.nn as nn

from recsys.fourstage_recsys.stages.scoring import Scorer
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.recsys_context import RecommendationContext

class CrossEncoderScoring(Scorer):
  def __init__(
    self,
    cross_encoder: nn.Module,
    user_features: dict[str, torch.Tensor],
    item_features: dict[str, torch.Tensor],
  ):
    self.cross_encoder = cross_encoder  #A
    self.user_features = user_features
    self.item_features = item_features

  def score(
    self,
    candidates: list[ScoredItem],
    context: RecommendationContext,
  ) -> list[ScoredItem]:
    user_feat = self.user_features[context.user_id]  #B
    for item in candidates:
      item_feat = self.item_features[item.item_id]  #C
      relevance = self.cross_encoder(
        user_feat.unsqueeze(0),
        item_feat.unsqueeze(0),
      ).item()  #D
      item.scores["relevance"] = relevance
    return candidates

# Callout annotations (from the book):
#   #A The cross-encoder model from Section 5.4 or 5.5
#   #B Look up precomputed features for the user in the context
#   #C Look up precomputed item features
#   #D Score the user-item pair through the cross-encoder
