# Figure - Listing 2.12: Most popular items
# Source: chapters/ch02.md lines 454-466
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
movie_counts = ratings['movieId'].value_counts()
max_count = movie_counts.max()

popularity_scores = {
    int(mid): float(count / max_count)
    for mid, count in movie_counts.items()
}

def get_most_popular_movies(k=10):
  return sorted(popularity_scores,  #A
      key=popularity_scores.get,  #B
      reverse=True
    )[:k]  #C

# Callout annotations (from the book):
#   #A Sort the popularity scored items
#   #B Sort based on the popularity score
#   #C Retun the k first ones.
