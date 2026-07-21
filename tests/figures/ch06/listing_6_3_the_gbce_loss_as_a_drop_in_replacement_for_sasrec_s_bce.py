# Figure — Listing 6.3: The gBCE loss as a drop-in replacement for SASRec's BCE
# Source: chapters/ch06.md lines 246-254
# Chapter: 6
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Sampling rate: the fraction of the catalogue that appears as a negative for each positive.
#   #B The calibration exponent; t=0 makes beta=1 (vanilla BCE), t=1 maximally correct.
#   #C Numerically stable form of \-beta \* log sigmoid(score).
#   #D Numerically stable form of \-log(1 \- sigmoid(score)).
#   #E Return the aggregated value of the negative and positive loss.
import torch.nn.functional as F

def gbce_loss(pos_scores, neg_scores, num_items, t=0.75):
    n_neg = neg_scores.shape[-1]
    alpha = n_neg / max(num_items - 1, 1)
    beta = alpha * (t * (1.0 - 1.0 / alpha) + 1.0 / alpha)
    pos_loss = beta * F.softplus(-pos_scores)
    neg_loss = F.softplus(neg_scores).sum(dim=-1)
    return (pos_loss + neg_loss).mean() / (n_neg + 1)
