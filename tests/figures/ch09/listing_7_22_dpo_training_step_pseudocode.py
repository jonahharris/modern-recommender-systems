# Figure - Listing 7.22: DPO training step (pseudocode)
# Source: chapters/ch09.md lines 1036-1053
# Chapter: 9
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def train_dpo_step(batch):
  with torch.no_grad():  #A
    ref_chosen_logps = get_logprobs(
      ref_model, batch['prompt'], batch['chosen'])
    ref_rejected_logps = get_logprobs(
      ref_model, batch['prompt'], batch['rejected'])

  policy_chosen_logps = get_logprobs(
    policy_model, batch['prompt'], batch['chosen'])  #B
  policy_rejected_logps = get_logprobs(
    policy_model, batch['prompt'], batch['rejected'])

  loss = dpo_loss(
    policy_chosen_logps, policy_rejected_logps,
    ref_chosen_logps, ref_rejected_logps)  #C

  loss.backward()
  optimizer.step()

# Callout annotations (from the book):
#   #A Frozen reference model, no gradients
#   #B Active policy model, gradients enabled
#   #C DPO loss compares preference gaps
