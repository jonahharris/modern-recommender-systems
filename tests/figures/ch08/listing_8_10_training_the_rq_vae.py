# Figure — Listing 8.10: Training the RQ-VAE
# Source: chapters/ch08.md lines 695-723
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Set model to training mode
#   #B Iterate through epochs
#   #C Iterate over mini-batches
#   #D Forward pass through RQ-VAE
#   #E Reconstruction loss against original input
#   #F Cosine similarity loss
#   #G Combined loss
#   #H Gradient clipping for stability
def train(self, data, epochs=500,
          batch_size=64):
  self.rqvae.train()
  num_batches = len(data) // batch_size
  pbar = tqdm(range(epochs))
  for epoch in pbar:
    epoch_loss = 0.0
    indices = torch.randperm(len(data))
    for i in range(num_batches):
      batch = data[indices[
        i*batch_size : (i+1)*batch_size]]
      self.optimizer.zero_grad()
      reconstructed, vq_loss, codes = \
        self.rqvae(batch)
      recon_loss = F.mse_loss(
        reconstructed, batch)
      cos_sim = F.cosine_similarity(
        reconstructed, batch, dim=1).mean()
      cos_loss = 1.0 - cos_sim
      loss = recon_loss + vq_loss \
        + 0.5 * cos_loss
      loss.backward()
      torch.nn.utils.clip_grad_norm_(
        self.rqvae.parameters(),
        max_norm=1.0)
      self.optimizer.step()
      epoch_loss += loss.item()
    pbar.set_postfix(
      {'loss': f'{epoch_loss/num_batches:.4f}'})
