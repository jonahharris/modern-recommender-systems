# Figure — Listing 7.15: Combined generative and contrastive loss
# Source: chapters/ch09.md lines 524-553
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Triplet margin loss function
#   #B Anchor: last hidden state of user history
#   #C Positive: hidden state of true next item
#   #D Negative: hidden state of random item
#   #E Contrastive loss
#   #F Standard cross-entropy loss
#   #G Contrastive weight
#   #H Combined loss
import torch
import torch.nn as nn
import torch.nn.functional as F

contrastive_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

def compute_combined_loss(model,
                          user_history_ids,
                          true_next_item_id,
                          negative_item_id):

  outputs_hist = model(user_history_ids, output_hidden_states=True)
  anchor_embed = outputs_hist.hidden_states[-1][:, -1, :]

  outputs_pos = model(true_next_item_id, output_hidden_states=True)
  pos_embed = outputs_pos.hidden_states[-1][:, -1, :]

  outputs_neg = model(negative_item_id, output_hidden_states=True)
  neg_embed = outputs_neg.hidden_states[-1][:, -1, :]

  loss_cl = contrastive_loss_fn(anchor_embed, pos_embed, neg_embed)

  logits = model(user_history_ids).logits
  loss_gen = F.cross_entropy(
    logits[:, -1, :], true_next_item_id.view(-1)
  )

  lambda_cl = 0.1
  total_loss = loss_gen + (lambda_cl * loss_cl)
  return total_loss
