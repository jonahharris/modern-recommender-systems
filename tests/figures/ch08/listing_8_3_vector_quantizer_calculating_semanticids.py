# Figure - Listing 8.3: Vector Quantizer (Calculating semanticIDs)
# Source: chapters/ch08.md lines 333-373
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def forward(self, inputs):
  input_shape = inputs.shape
  flat_input = inputs.view(-1, self.embedding_dim)

  distances = (
    torch.sum(flat_input ** 2, dim=1, keepdim=True) +
    torch.sum(self.embedding ** 2, dim=1) -
    2 * torch.matmul(flat_input, self.embedding.t())
  )  #A

  encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  #B

  encodings = torch.zeros(encoding_indices.shape[0],
    self.num_embeddings, device=inputs.device)
  encodings.scatter_(1, encoding_indices, 1)  #C
  quantized = torch.matmul(
    encodings, self.embedding).view(input_shape)  #D

  if self.training:
    self._update_codebooks(encodings, flat_input)  #E
    batch_usage = encodings.mean(0)
    self._usage_smoothed.mul_(0.99).add_(
      batch_usage, alpha=0.01)  #F

  e_latent_loss = F.mse_loss(quantized.detach(), inputs)  #G
  q_latent_loss = F.mse_loss(quantized, inputs.detach())  #H
  loss = q_latent_loss + self.commitment_cost * e_latent_loss  #I

  if self.training and self.usage_loss_weight > 0:
    usage_probs = self._usage_smoothed + 1e-10
    usage_probs = usage_probs / usage_probs.sum()
    entropy = -(usage_probs *
      torch.log(usage_probs + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(
      self.num_embeddings, dtype=torch.float32,
      device=inputs.device))
    usage_loss = 1.0 - (entropy / max_entropy)
    loss = loss + self.usage_loss_weight * usage_loss  #J

  quantized = inputs + (quantized - inputs).detach()  #K
  return quantized, loss, encoding_indices.squeeze(1)

# Callout annotations (from the book):
#   #A Squared Euclidean distance to all codebook vectors
#   #B Index of closest codebook vector
#   #C One-hot encoding of selected codes
#   #D Look up selected codebook vector
#   #E EMA codebook update (see Listing 8.4)
#   #F Track smoothed usage statistics
#   #G Commitment loss: encoder should stay close to codebook
#   #H Codebook loss: codebook should stay close to encoder
#   #I Combined VQ loss
#   #J Usage loss: penalize unbalanced code usage
#   #K Straight-Through Estimator
