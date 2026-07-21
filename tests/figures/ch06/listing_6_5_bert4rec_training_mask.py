# Figure — Listing 6.5: BERT4Rec training mask
# Source: chapters/ch06.md lines 389-402
# Chapter: 6
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Each non-padding position is independently masked with probability mask\_prob.
#   #B Force at least one so the loss never receives an empty mask.
#   #C Replace the selected positions with the \[MASK\] token in the input.
def mask_sequence(self, sequences):
    non_pad    = (sequences != 0)
    rand       = torch.rand_like(sequences, dtype=torch.float)
    cloze_mask = non_pad & (rand < self.mask_prob)  # A

    # Guarantee at least one masked position per sequence.
    empty_rows = cloze_mask.sum(dim=1) == 0  # B
    if empty_rows.any():
        last_real = (non_pad.cumsum(dim=1) * non_pad).argmax(dim=1)
        cloze_mask[empty_rows, last_real[empty_rows]] = True

    masked_seqs = sequences.clone()
    masked_seqs[cloze_mask] = self.mask_token  # C
    return masked_seqs, cloze_mask
