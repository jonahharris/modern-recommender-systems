# Figure - Listing 8.9: Codebook initialization
# Source: chapters/ch08.md lines 658-671
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def initialize_data_with_embeddings(self,
                                     embeddings):
  data = torch.tensor(embeddings,
    dtype=torch.float32).to(self.device)  #A
  with torch.no_grad():
    latents = self.rqvae.encoder(data)  #B
    residual = latents  #C
    for i in range(self.rqvae.depth):  #D
      self.rqvae.quantizers[i] \
        .init_codebook(residual)  #E
      z_q, _, _ = \
        self.rqvae.quantizers[i](residual)  #F
      residual = residual - z_q  #G
  return data  #H

# Callout annotations (from the book):
#   #A Convert embeddings to tensor
#   #B Encode through the RQ-VAE encoder
#   #C Start with raw encoder output
#   #D Iterate through codebook levels
#   #E Initialize this level with K-means
#   #F Quantize to compute this level's approximation
#   #G Residual for next level
#   #H Return original data (not encoded) for training
