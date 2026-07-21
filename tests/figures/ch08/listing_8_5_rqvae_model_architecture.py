# Figure - Listing 8.5: RQVAE Model Architecture
# Source: chapters/ch08.md lines 437-504
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
class RQVAE(nn.Module):
  def __init__(self, input_dim, embed_dim,
               codebook_sizes=[8, 64, 512],
               usage_loss_weight=2.0):
    super().__init__()
    self.depth = len(codebook_sizes)  #A

    self.encoder = nn.Sequential(  #B
      nn.Linear(input_dim, input_dim),
      nn.LayerNorm(input_dim),
      nn.ReLU(),
      nn.Linear(input_dim, embed_dim * 2),
      nn.LayerNorm(embed_dim * 2),
      nn.ReLU(),
      nn.Linear(embed_dim * 2, embed_dim),
      nn.LayerNorm(embed_dim)
    )

    self.quantizers = nn.ModuleList([  #C
      VectorQuantizerEMA(size, embed_dim,
        commitment_cost=0.5,
        decay=0.90,
        usage_loss_weight=usage_loss_weight)
      for size in codebook_sizes
    ])

    self.decoder = nn.Sequential(  #D
      nn.Linear(embed_dim, embed_dim * 2),
      nn.LayerNorm(embed_dim * 2),
      nn.ReLU(),
      nn.Linear(embed_dim * 2, input_dim),
      nn.LayerNorm(input_dim),
      nn.ReLU(),
      nn.Linear(input_dim, input_dim)
    )
    self._init_encoder()  #E

  def _init_encoder(self):
    with torch.no_grad():
      for module in self.encoder:  #F
        if isinstance(module, nn.Linear):
          nn.init.kaiming_uniform_(
            module.weight, nonlinearity='relu')
          nn.init.zeros_(module.bias)
      for module in self.decoder:  #G
        if isinstance(module, nn.Linear):
          nn.init.xavier_uniform_(module.weight)
          nn.init.zeros_(module.bias)

  def forward(self, x):
    z = self.encoder(x)  #H

    quantized_sum = 0
    residual = z
    total_loss = 0
    all_indices = []

    for i in range(self.depth):  #I
      z_q, loss, indices = \
        self.quantizers[i](residual)
      residual = residual - z_q  #J
      quantized_sum += z_q
      total_loss += loss
      all_indices.append(indices)

    reconstructed = self.decoder(quantized_sum)  #K
    codes = torch.stack(all_indices, dim=1)  #L
    return reconstructed, total_loss, codes

# Callout annotations (from the book):
#   #A Number of quantization levels
#   #B Three-layer encoder with LayerNorm
#   #C One quantizer per level with usage balancing
#   #D Decoder mirrors encoder, outputs input dimension
#   #E Weight initialization
#   #F Kaiming init for ReLU layers
#   #G Xavier init for decoder
#   #H Encode input to internal representation
#   #I Loop through levels (coarse to fine)
#   #J Residual for next level
#   #K Decode accumulated representation
#   #L Stack codes into semantic ID tensor
