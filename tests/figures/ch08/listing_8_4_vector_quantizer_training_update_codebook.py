# Figure — Listing 8.4: Vector Quantizer (training \- Update codebook)
# Source: chapters/ch08.md lines 407-427
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
  def _update_codebooks(self, encodings, flat_input):
    dw = torch.sum(encodings, dim=0)
    self.cluster_size.data.mul_(self.decay).add_(
      dw, alpha=1 - self.decay)

    dw_embed = torch.matmul(encodings.t(), flat_input)
    self.embed_avg.data.mul_(self.decay).add_(
      dw_embed, alpha=1 - self.decay)

    n = self.cluster_size.sum()
    cluster_size = ((self.cluster_size + self.epsilon) /
                   (n + self.num_embeddings * self.epsilon) * n)

    embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
    self.embedding.data.copy_(embed_normalized)

#A Updates the codebook using Exponential Moving Average (EMA).
#B Update the "Usage Count".
#C Update the "Average Position".
#D Normalize the vectors.
#E Assign new weights for the Codebook.
