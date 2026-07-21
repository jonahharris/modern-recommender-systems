# Figure - Listing 2.18: Content-based retrieval function
# Source: chapters/ch02.md lines 685-709
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
movie_id_to_idx = {
    int(mid): idx
    for idx, mid in enumerate(movies['movieId'])
}  #A
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}  #A

def retrieve_similar_by_content(movie_id, k=100):
    if movie_id not in movie_id_to_idx:
        return []

    movie_idx = movie_id_to_idx[movie_id]  #B
    movie_vector = content_vectors[movie_idx]  #B

    similarities = cosine_similarity(movie_vector, content_vectors)[0]  #C

    top_indices = np.argsort(similarities)[-(k+1):-1][::-1]  #D

    candidates = []
    for idx in top_indices:
        candidates.append({
            'movie_id': int(idx_to_movie_id[idx]),
            'content_similarity': float(similarities[idx])
        })

    return candidates  #E

# Callout annotations (from the book):
#   #A Create movie ID to index mapping
#   #B Get vector for seed movie
#   #C Compute similarity to all other movies
#   #D Find top k most similar
#   #E Return candidates with scores
