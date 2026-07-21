# Figure - Listing 7.26: Sampled softmax loss (illustrative)
# Source: chapters/ch09.md lines 1244-1265
# Chapter: 9
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class SampledSoftmaxLoss(nn.Module):
  def __init__(self, embedding, num_samples=10000):
    super().__init__()
    self.embedding = embedding  #A
    self.num_samples = num_samples

  def forward(self, hidden_states, target_ids,
              full_vocab_size):
    sampled_ids = target_ids.clone()  #B
    num_negatives = self.num_samples - len(target_ids)
    negatives = torch.randint(
      0, full_vocab_size, (num_negatives,)
    )
    sampled_ids = torch.cat([sampled_ids, negatives])  #C

    sampled_weights = self.embedding.weight[sampled_ids]
    logits = torch.matmul(
      hidden_states, sampled_weights.T
    )

    target_positions = torch.arange(len(target_ids))
    return F.cross_entropy(logits, target_positions)  #D

# Callout annotations (from the book):
#   #A Reference to the model's output embedding layer
#   #B Always include target items
#   #C Add random negative samples
#   #D Cross-entropy over the reduced vocabulary
