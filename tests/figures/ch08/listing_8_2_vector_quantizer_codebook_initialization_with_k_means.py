# Figure — Listing 8.2: Vector Quantizer (Codebook initialization with K-means )
# Source: chapters/ch08.md lines 293-309
# Chapter: 8
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A CPU transfer for scikit-learn
#   #B Skip if data is degenerate or too small
#   #C Cluster data into num\_embeddings groups
#   #D Replace random codebook with centroids
#   #E Avoid division by zero
#   #F Align EMA buffer with new centroids
def init_codebook(self, data):
  data_np = data.detach().cpu().numpy()

  if data_np.std() < 1e-6 or len(data_np) < self.num_embeddings:
    return

  kmeans = KMeans(n_clusters=self.num_embeddings,
                  n_init=10, max_iter=300,
                  random_state=42)
  kmeans.fit(data_np)

  centroids = torch.tensor(kmeans.cluster_centers_,
                            device=data.device,
                            dtype=data.dtype)
  self.embedding.data.copy_(centroids)
  self.cluster_size.data.fill_(1.0)
  self.embed_avg.data.copy_(self.embedding)
