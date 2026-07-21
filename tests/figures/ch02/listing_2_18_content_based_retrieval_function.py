# Figure — Listing 2.18: Content-based retrieval function
# Source: chapters/ch02.md lines 685-709
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Create movie ID to index mapping
#   #B Get vector for seed movie
#   #C Compute similarity to all other movies
#   #D Find top k most similar
#   #E Return candidates with scores
movie_id_to_idx = {
    int(mid): idx
    for idx, mid in enumerate(movies['movieId'])
}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}

def retrieve_similar_by_content(movie_id, k=100):
    if movie_id not in movie_id_to_idx:
        return []

    movie_idx = movie_id_to_idx[movie_id]
    movie_vector = content_vectors[movie_idx]

    similarities = cosine_similarity(movie_vector, content_vectors)[0]

    top_indices = np.argsort(similarities)[-(k+1):-1][::-1]

    candidates = []
    for idx in top_indices:
        candidates.append({
            'movie_id': int(idx_to_movie_id[idx]),
            'content_similarity': float(similarities[idx])
        })

    return candidates
