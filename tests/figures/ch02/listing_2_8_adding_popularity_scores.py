# Figure - Listing 2.8: Adding popularity scores
# Source: chapters/ch02.md lines 277-291
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def score_popularity(candidates):
    for item in candidates:
        item['popularity'] = popularity_scores.get(item['movie_id'], 0.0)
    return candidates

candidates = retrieve_similar_items(seed_movie_id, k=10)
candidates = filter_watched(candidates, test_user)
candidates = score_popularity(candidates)

print(f"Candidates with multiple scores:\n")
for item in candidates[:3]:
    title = movies[movies['movieId'] == item['movie_id']]['title'].values[0]
    print(f"{title}")
    print(f"  Similarity: {item['similarity']:.3f}")
    print(f"  Popularity: {item['popularity']:.3f}\n")

# Callout annotations (from the book):
#   #A Add popularity score to each candidate
#   #B Display candidates with both scores
