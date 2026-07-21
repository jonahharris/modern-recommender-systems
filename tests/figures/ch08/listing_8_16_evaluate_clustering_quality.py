# Figure - Listing 8.16: Evaluate clustering quality
# Source: chapters/ch08.md lines 916-934
# Chapter: 8
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def evaluate_clustering(df, embeddings):
  top_level = df['semantic_id'].apply(
    lambda x: x[0]).values  #A
  silhouette = silhouette_score(
    embeddings, top_level)  #B
  calinski = calinski_harabasz_score(
    embeddings, top_level)  #C
  second_level = df['semantic_id'].apply(
    lambda x: f"{x[0]}_{x[1]}").values  #D
  silhouette_l2 = silhouette_score(
    embeddings, second_level)  #E
  return {
    'silhouette_l1': silhouette,
    'silhouette_l2': silhouette_l2,
    'calinski_harabasz': calinski
  }

# Callout annotations (from the book):
#   #A Extract top-level codes
#   #B Silhouette: -1 (bad) to 1 (perfect)
#   #C Between-cluster vs. within-cluster variance
#   #D Combine first and second level
#   #E Silhouette at finer granularity
