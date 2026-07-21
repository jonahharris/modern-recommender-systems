# Figure — Listing 6.5: BERT4Rec training loop
# Source: chapters/ch06.md lines 420-426
# Chapter: 6
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A mask\_sequence returns the masked input and a boolean mask marking which positions were replaced. This is the only place masking happens — the dataset contains unmasked sequences.
#   #B The scoring target is raw\_seq, not input\_seq. input\_seq has \[MASK\] at the masked positions; scoring against those embeddings would reward the model for predicting \[MASK\] rather than the original item.
#   #C cloze\_mask passes to gbce\_loss as the pos\_mask argument, so the loss is computed only at masked positions. Unmasked positions, where the model wasn't asked to predict, contribute nothing to the gradient.
raw_seq, neg_items = [b.to(device) for b in batch]
input_seq, cloze_mask = model.mask_sequence(raw_seq) # A

hidden     = model(input_seq)
pos_scores = (hidden * model.item_emb(raw_seq)).sum(-1) # B
neg_scores = (hidden.unsqueeze(2) * model.item_emb(neg_items)).sum(-1)
loss = gbce_loss(pos_scores, neg_scores, cloze_mask, num_items)  # C
