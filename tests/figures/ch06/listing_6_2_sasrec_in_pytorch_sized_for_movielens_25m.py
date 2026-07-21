# Figure - Listing 6.2: SASRec in PyTorch, sized for MovieLens 25M
# Source: chapters/ch06.md lines 196-213
# Chapter: 6
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class SASRec(_SequentialBase):
  def forward(self, sequences: torch.Tensor) -> torch.Tensor:
    seq_len = sequences.shape[1]
    causal_mask = torch.triu(  #A
      torch.ones(seq_len, seq_len,
                 device=sequences.device, dtype=torch.bool),
      diagonal=1,
      )
    return self._encode(sequences, attn_mask=causal_mask)

  def recommend(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
    with torch.no_grad():
      hidden = self.forward(sequences)
      user_state = hidden[:, -1, :]  #B
      scores = user_state @ self.item_emb.weight[
                     :self.num_items + 1].T  #C
      scores[:, 0] = -float("inf")  #D
    return torch.topk(scores, k, dim=1).indices

# Callout annotations (from the book):
#   #A The causal mask blocks position i from attending to positions j \> i. This mask is the only thing SASRec adds to the backbone.
#   #B The user state is the hidden vector at the last (rightmost) position.
#   #C Score every real item by inner product with the same embedding table used at the input — the tied scoring head.
#   #D Never recommend the padding token.
