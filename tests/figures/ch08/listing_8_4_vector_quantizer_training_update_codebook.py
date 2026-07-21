# Figure - Listing 8.4: Vector Quantizer (training \- Update codebook)
# Source: chapters/ch08.md lines 407-421
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
  def _update_codebooks(self, encodings, flat_input):  #A
    dw = torch.sum(encodings, dim=0)
    self.cluster_size.data.mul_(self.decay).add_(
      dw, alpha=1 - self.decay)  #B

    dw_embed = torch.matmul(encodings.t(), flat_input)
    self.embed_avg.data.mul_(self.decay).add_(
      dw_embed, alpha=1 - self.decay)  #C

    n = self.cluster_size.sum()
    cluster_size = ((self.cluster_size + self.epsilon) /
                   (n + self.num_embeddings * self.epsilon) * n)  #D

    embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
    self.embedding.data.copy_(embed_normalized)  #E

# Callout annotations (from the book):
#   #A Updates the codebook using Exponential Moving Average (EMA).
#   #B Update the "Usage Count".
#   #C Update the "Average Position".
#   #D Normalize the vectors.
#   #E Assign new weights for the Codebook.
