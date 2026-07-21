# Figure — Listing 5.8: A simple cross-encoder reranker
# Source: chapters/ch05.md lines 531-554
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Combine user and item feature dimensions
#   #B A small interaction network that scores the combined input
#   #C Concatenate user and item features into a single vector
#   #D Produce a relevance score
class CrossEncoderReranker(nn.Module):
  def __init__(
    self,
    user_feature_dim: int,
    item_feature_dim: int,
    hidden_dim: int = 256,
  ):
    super().__init__()
    input_dim = user_feature_dim + item_feature_dim
    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1),
    )

  def forward(
    self,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
  ) -> torch.Tensor:
    combined = torch.cat(
      [user_features, item_features], dim=-1
    )
    return self.network(combined).squeeze(-1)
