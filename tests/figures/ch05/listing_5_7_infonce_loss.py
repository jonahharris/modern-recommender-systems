# Figure - Listing 5.7: InfoNCE loss
# Source: chapters/ch05.md lines 411-440
# Chapter: 5
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class InfoNCELoss(torch.nn.Module):
  def __init__(self, temperature: float = 0.07):
    super().__init__()
    self.temperature = temperature

  def forward(
    self,
    user_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
  ) -> torch.Tensor:
    u = F.normalize(user_embeddings, dim=-1)  #A
    v_pos = F.normalize(positive_embeddings, dim=-1)
    v_neg = F.normalize(negative_embeddings, dim=-1)

    pos_scores = (u * v_pos).sum(dim=-1) / self.temperature  #B

    neg_scores = torch.bmm(
      u.unsqueeze(1),
      v_neg.transpose(1, 2),
    ).squeeze(1) / self.temperature  #C

    logits = torch.cat(
      [pos_scores.unsqueeze(1), neg_scores], dim=1
    )  #D
    targets = torch.zeros(
      u.shape[0], dtype=torch.long, device=u.device
    )  #E

    return F.cross_entropy(logits, targets)  #F

# Callout annotations (from the book):
#   #A Normalize embeddings so dot product equals cosine similarity
#   #B Compute similarity score for the positive item
#   #C Compute similarity scores for all negative items
#   #D Combine into a single logits tensor
#   #E The positive item is always at index 0
#   #F Cross-entropy loss identifies the positive among all candidates
