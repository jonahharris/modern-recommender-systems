# Figure - Listing 8.2: Vector Quantizer (Codebook initialization with K-means )
# Source: chapters/ch08.md lines 293-309
# Chapter: 8
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def init_codebook(self, data):
  data_np = data.detach().cpu().numpy()  #A

  if data_np.std() < 1e-6 or len(data_np) < self.num_embeddings:
    return  #B

  kmeans = KMeans(n_clusters=self.num_embeddings,
                  n_init=10, max_iter=300,
                  random_state=42)
  kmeans.fit(data_np)  #C

  centroids = torch.tensor(kmeans.cluster_centers_,
                            device=data.device,
                            dtype=data.dtype)
  self.embedding.data.copy_(centroids)  #D
  self.cluster_size.data.fill_(1.0)  #E
  self.embed_avg.data.copy_(self.embedding)  #F

# Callout annotations (from the book):
#   #A CPU transfer for scikit-learn
#   #B Skip if data is degenerate or too small
#   #C Cluster data into num\_embeddings groups
#   #D Replace random codebook with centroids
#   #E Avoid division by zero
#   #F Align EMA buffer with new centroids
