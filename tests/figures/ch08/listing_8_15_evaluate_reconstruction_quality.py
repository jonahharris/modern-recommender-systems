# Figure — Listing 8.15: Evaluate reconstruction quality
# Source: chapters/ch08.md lines 872-896
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
def evaluate_reconstruction(pipeline,
                             data_tensor):
  pipeline.rqvae.eval()
  with torch.no_grad():
    reconstructed, vq_loss, codes = \
      pipeline.rqvae(data_tensor)
    mse = F.mse_loss(
      reconstructed, data_tensor)
    cos_sim = F.cosine_similarity(
      reconstructed, data_tensor).mean()
    correlation = torch.corrcoef(
      torch.stack([
        reconstructed.flatten(),
        data_tensor.flatten()])
    )[0, 1]
  return {
    'mse': mse.item(),
    'cosine_similarity': cos_sim.item(),
    'correlation': correlation.item()
  }

#A Forward pass to get reconstructed embeddings
#B MSE against original BERT embeddings
#C Cosine similarity (directional agreement)
#D Correlation across all dimensions
