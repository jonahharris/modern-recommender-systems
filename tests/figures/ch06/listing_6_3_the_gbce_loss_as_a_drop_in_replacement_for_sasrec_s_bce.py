# Figure - Listing 6.3: The gBCE loss as a drop-in replacement for SASRec's BCE
# Source: chapters/ch06.md lines 246-254
# Chapter: 6
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import torch.nn.functional as F

def gbce_loss(pos_scores, neg_scores, pos_mask, num_items, t=0.75):
    n_neg = neg_scores.shape[-1]
    alpha = n_neg / max(num_items - 1, 1)  #A
    beta = alpha * (t * (1.0 - 1.0 / alpha) + 1.0 / alpha)  #B
    pos_loss = beta * F.softplus(-pos_scores)  #C
    neg_loss = F.softplus(neg_scores).sum(dim=-1)  #D
    per_position = (pos_loss + neg_loss) / (n_neg + 1)
    per_position = per_position * pos_mask.float()  #E
    return per_position.sum() / pos_mask.float().sum().clamp(min=1.0)  #E

# Callout annotations (from the book):
#   #A Sampling rate: the fraction of the catalogue that appears as a negative for each positive.
#   #B The calibration exponent; t=0 makes beta=1 (vanilla BCE), t=1 maximally correct.
#   #C Numerically stable form of \-beta \* log sigmoid(score).
#   #D Numerically stable form of \-log(1 \- sigmoid(score)).
#   #E Zero out non-target positions after computing the loss, then average over the masked target positions only. softplus(-0)=log(2), so averaging over padding would add an irreducible constant.
