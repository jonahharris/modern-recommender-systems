# Figure - Listing 2.20: Hybrid retrieval
# Source: chapters/ch02.md lines 818-843
# Chapter: 2
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def retrieve_hybrid(movie_id, k=100):
    content_candidates = retrieve_similar_by_content(movie_id, k=k//2)
    behavioral_candidates = retrieve_similar_items(movie_id, k=k//2)

    all_candidates = {}

    for item in content_candidates:
        mid = item['movie_id']
        all_candidates[mid] = {
            'movie_id': mid,
            'content_similarity': item['content_similarity'],
            'behavioral_similarity': 0.0
        }

    for item in behavioral_candidates:
        mid = item['movie_id']
        if mid in all_candidates:
            all_candidates[mid]['behavioral_similarity'] = item['similarity']
        else:
            all_candidates[mid] = {
                'movie_id': mid,
                'content_similarity': 0.0,
                'behavioral_similarity': item['similarity']
            }

    return list(all_candidates.values())

# Callout annotations (from the book):
#   #A Get candidates from content similarity
#   #B Get candidates from behavioral similarity
#   #C Merge, keeping both scores for each item
