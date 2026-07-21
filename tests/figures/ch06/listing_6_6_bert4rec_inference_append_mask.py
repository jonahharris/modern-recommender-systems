# Figure - Listing 6.6: BERT4Rec inference: append \[MASK\]
# Source: chapters/ch06.md lines 436-444
# Chapter: 6
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def recommend(self, sequences, k=10):
    mask_col  = torch.full((sequences.shape[0], 1), self.mask_token,
                            dtype=torch.long, device=sequences.device)
    augmented = torch.cat([sequences, mask_col], dim=1)[:, -self.max_len:]  # A
    with torch.no_grad():  # C
        hidden     = self.forward(augmented)
        user_state = hidden[:, -1, :]  # B
        scores     = user_state @ self.item_emb.weight[:self.num_items + 1].T
        scores[:, 0] = -float("inf")
    return torch.topk(scores, k, dim=1).indices

# Callout annotations (from the book):
#   #A Trim to max\_len
#   #B The \[MASK\] token sits at the rightmost position, so hidden\[:, \-1, :\] is the user state.
#   #C Inference only \- disable gradient tracking so scoring the full item table allocates no autograd graph.
