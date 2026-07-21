# Figure - Listing 8.7: SemanticIDPipeline (initialization)
# Source: chapters/ch08.md lines 596-614
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class SemanticIDPipeline:
  def __init__(self,
               codebook_sizes=[8, 32, 128],
               internal_dim=64,  #A
               usage_loss_weight=10.0):  #B
    self.device = torch.device(
      "cuda" if torch.cuda.is_available()
      else "cpu")
    self.text_encoder = SentenceTransformer(
      'all-MiniLM-L6-v2')  #C
    bert_dim = self.text_encoder \
      .get_sentence_embedding_dimension()  #D
    self.codebook_sizes = codebook_sizes
    self.rqvae = RQVAE(
      bert_dim, internal_dim, codebook_sizes,
      usage_loss_weight=usage_loss_weight
    ).to(self.device)  #E
    self.optimizer = optim.Adam(
      self.rqvae.parameters(), lr=5e-4)  #F

# Callout annotations (from the book):
#   #A Internal dimension for quantization space
#   #B Usage loss passed through to quantizers
#   #C Text encoder (384-dim embeddings)
#   #D Get BERT embedding dimension
#   #E Create RQ-VAE with usage balancing
#   #F Adam optimizer with learning rate 5e-4
