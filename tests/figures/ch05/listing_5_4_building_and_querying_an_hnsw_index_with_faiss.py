# Figure — Listing 5.4: Building and querying an HNSW index with FAISS
# Source: chapters/ch05.md lines 209-230
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Normalize embeddings so dot product equals cosine similarity
#   #B Create an HNSW index with 32 connections per node
#   #C Higher efConstruction improves recall during index building
#   #D Higher efSearch improves recall at query time (at the cost of speed)
#   #E Add all item embeddings to the index
#   #F Normalize the query to match the index
#   #G Return the top-k nearest item indices and their scores
import faiss
import numpy as np

def build_index(item_embeddings: np.ndarray) -> faiss.Index:
  embeddings = item_embeddings.astype(np.float32)
  faiss.normalize_L2(embeddings)
  dim = embeddings.shape[1]
  index = faiss.IndexHNSWFlat(dim, 32)
  index.hnsw.efConstruction = 200
  index.hnsw.efSearch = 50
  index.add(embeddings)
  return index

def query_index(
  index: faiss.Index,
  query_embedding: np.ndarray,
  k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
  query = query_embedding.astype(np.float32).reshape(1, -1)
  faiss.normalize_L2(query)
  scores, ids = index.search(query, k)
  return ids[0], scores[0]
