# Figure - Listing 8.1: Vector Quantizer (initialization)
# Source: chapters/ch08.md lines 253-273
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
class VectorQuantizerEMA(nn.Module):
  def __init__(self, num_embeddings, embedding_dim,
               commitment_cost=0.5,  #A
               decay=0.95,
               usage_loss_weight=0.1):  #B
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.commitment_cost = commitment_cost
    self.decay = decay
    self.epsilon = 1e-5
    self.usage_loss_weight = usage_loss_weight

    self.register_buffer('embedding',
      torch.randn(num_embeddings, embedding_dim) * 0.01)  #C
    self.register_buffer('cluster_size',
      torch.zeros(num_embeddings))  #D
    self.register_buffer('embed_avg',
      self.embedding.clone())  #D
    self.register_buffer('_usage_smoothed',
      torch.ones(num_embeddings) / num_embeddings)  #E

# Callout annotations (from the book):
#   #A Encoder commitment strength
#   #B Penalty for unbalanced codebook usage
#   #C Codebook vectors (buffer, not updated by optimizer)
#   #D EMA tracking buffers
#   #E Smoothed usage statistics for entropy loss
