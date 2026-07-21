# Figure - Listing 6.6: BERT4Rec inference: append \[MASK\]
# Source: chapters/ch06.md lines 436-444
# Chapter: 6
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def recommend(self, sequences, k=10):
    mask_col  = torch.full((sequences.shape[0], 1), self.mask_token,
                            dtype=torch.long, device=sequences.device)
    augmented = torch.cat([sequences, mask_col], dim=1)[:, -self.max_len:]  # A
    hidden     = self.forward(augmented)
    user_state = hidden[:, -1, :] # B
    scores     = user_state @ self.item_emb.weight[:self.num_items + 1].T
    scores[:, 0] = -float("inf")
    return torch.topk(scores, k, dim=1).indices

# Callout annotations (from the book):
#   #A Trim to max\_len
#   #B The \[MASK\] token sits at the rightmost position, so hidden\[:, \-1, :\] is the user state.
