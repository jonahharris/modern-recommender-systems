# Figure - Listing 2.7: Computing popularity scores
# Source: chapters/ch02.md lines 253-259
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
movie_counts = ratings['movieId'].value_counts()  #A
max_count = movie_counts.max()  #B

popularity_scores = {
    int(mid): float(count / max_count)
    for mid, count in movie_counts.items()
}

# Callout annotations (from the book):
#   #A Count ratings per movie
#   #B Normalize by max count to get 0-1 range
