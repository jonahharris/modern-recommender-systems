# Figure - Listing 7.21: DPO loss function
# Source: chapters/ch09.md lines 979-998
# Chapter: 9
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
import torch.nn.functional as F
import torch.nn as nn

class DPOLoss(nn.Module):
  def __init__(self, beta=0.1):
    super().__init__()
    self.beta = beta  #A

  def forward(self,
              policy_chosen_logps,
              policy_rejected_logps,
              ref_chosen_logps,
              ref_rejected_logps):  #B
    pi_logratios = (policy_chosen_logps
                    - policy_rejected_logps)  #C
    ref_logratios = (ref_chosen_logps
                     - ref_rejected_logps)
    logits = pi_logratios - ref_logratios  #D
    losses = -F.logsigmoid(self.beta * logits)
    return losses.mean()

# Callout annotations (from the book):
#   #A Controls how far the model can deviate from the base
#   #B Requires log-probabilities from both models
#   #C Gap between chosen and rejected in the policy model
#   #D Compare against the same gap in the reference model
