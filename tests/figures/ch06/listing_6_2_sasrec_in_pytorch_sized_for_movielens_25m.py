# Figure — Listing 6.2: SASRec in PyTorch, sized for MovieLens 25M
# Source: chapters/ch06.md lines 196-213
# Chapter: 6
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A The causal mask blocks position i from attending to positions j \> i. This mask is the only thing SASRec adds to the backbone.
#   #B The user state is the hidden vector at the last (rightmost) position.
#   #C Score every real item by inner product with the same embedding table used at the input — the tied scoring head.
#   #D Never recommend the padding token.
class SASRec(_SequentialBase):
  def forward(self, sequences: torch.Tensor) -> torch.Tensor:
    seq_len = sequences.shape[1]
    causal_mask = torch.triu(
      torch.ones(seq_len, seq_len,
                 device=sequences.device, dtype=torch.bool),
      diagonal=1,
      )
    return self._encode(sequences, attn_mask=causal_mask)

  def recommend(self, sequences: torch.Tensor, k: int = 10) -> torch.Tensor:
    with torch.no_grad():
      hidden = self.forward(sequences)
      user_state = hidden[:, -1, :]
      scores = user_state @ self.item_emb.weight[
                     :self.num_items + 1].T
      scores[:, 0] = -float("inf")
    return torch.topk(scores, k, dim=1).indices
