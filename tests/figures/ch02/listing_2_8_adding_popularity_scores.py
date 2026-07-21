# Figure - Listing 2.8: Adding popularity scores
# Source: chapters/ch02.md lines 277-291
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def score_popularity(candidates):
    for item in candidates:
        item['popularity'] = popularity_scores.get(item['movie_id'], 0.0)  #A
    return candidates

candidates = retrieve_similar_items(seed_movie_id, k=10)
candidates = filter_watched(candidates, test_user)
candidates = score_popularity(candidates)

print(f"Candidates with multiple scores:\n")  #B
for item in candidates[:3]:
    title = movies[movies['movieId'] == item['movie_id']]['title'].values[0]
    print(f"{title}")
    print(f"  Similarity: {item['similarity']:.3f}")
    print(f"  Popularity: {item['popularity']:.3f}\n")

# Callout annotations (from the book):
#   #A Add popularity score to each candidate
#   #B Display candidates with both scores
