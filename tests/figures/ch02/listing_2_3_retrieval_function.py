# Figure - Listing 2.3: Retrieval function
# Source: chapters/ch02.md lines 141-157
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def retrieve_similar_items(movie_id, k=100):
    if movie_id not in movie_to_idx:  #A
        return []

    movie_idx = movie_to_idx[movie_id]
    similarities = item_similarity[movie_idx].toarray()[0]  #B

    top_indices = np.argsort(similarities)[-(k+1):-1][::-1]  #C

    candidates = []
    for idx in top_indices:
        candidates.append({
            'movie_id': idx_to_movie[idx],
            'similarity': float(similarities[idx])
        })  #D

    return candidates

# Callout annotations (from the book):
#   #A Check if movie exists in our data
#   #B Get pre-computed similarity scores
#   #C Find top k most similar (excluding the movie itself)
#   #D Return list of candidates with scores
