# Figure — Listing 8.7: SemanticIDPipeline (initialization)
# Source: chapters/ch08.md lines 596-614
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Internal dimension for quantization space
#   #B Usage loss passed through to quantizers
#   #C Text encoder (384-dim embeddings)
#   #D Get BERT embedding dimension
#   #E Create RQ-VAE with usage balancing
#   #F Adam optimizer with learning rate 5e-4
class SemanticIDPipeline:
  def __init__(self,
               codebook_sizes=[16, 32, 128],
               internal_dim=512,
               usage_loss_weight=2.0):
    self.device = torch.device(
      "cuda" if torch.cuda.is_available()
      else "cpu")
    self.text_encoder = SentenceTransformer(
      'all-MiniLM-L6-v2')
    bert_dim = self.text_encoder \
      .get_sentence_embedding_dimension()
    self.codebook_sizes = codebook_sizes
    self.rqvae = RQVAE(
      bert_dim, internal_dim, codebook_sizes,
      usage_loss_weight=usage_loss_weight
    ).to(self.device)
    self.optimizer = optim.Adam(
      self.rqvae.parameters(), lr=5e-4)
