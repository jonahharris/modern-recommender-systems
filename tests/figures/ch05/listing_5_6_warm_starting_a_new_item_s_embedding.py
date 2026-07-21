# Figure - Listing 5.6: Warm-starting a new item's embedding
# Source: chapters/ch05.md lines 287-314
# Chapter: 5
# Category: needs-package  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def warm_start_embedding(
  new_movie_genres: list[str],
  genre_to_items: dict[str, list[int]],
  item_embeddings: np.ndarray,
) -> np.ndarray:
  neighbor_indices = set()
  for genre in new_movie_genres:
    neighbor_indices.update(
      genre_to_items.get(genre, [])
    )  #A

  if not neighbor_indices:
    return np.random.randn(
      item_embeddings.shape[1]
    ).astype(np.float32)  #B

  neighbors = item_embeddings[list(neighbor_indices)]
  embedding = neighbors.mean(axis=0)  #C
  embedding = embedding/np.linalg.norm(embedding)  #D
  return embedding
# Usage: a new animated children's film arrives
new_embedding = warm_start_embedding(
  new_movie_genres=["Animation", "Children"],
  genre_to_items=genre_to_items,
  item_embeddings=item_embeddings,
)  #E
new_embedding_2d = new_embedding.reshape(1, -1)
index.add(new_embedding_2d)  #F

# Callout annotations (from the book):
#   #A Collect existing items that share any genre with the new item
#   #B Fall back to random if no genre matches exist
#   #C Average their embeddings to create an initial representation
#   #D Normalize to match the index
#   #E Compute a warm-start embedding for the new movie
#   #F Add it to the FAISS index — the item is now retrievable
