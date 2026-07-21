# Figure — Listing 2.7: Computing popularity scores
# Source: chapters/ch02.md lines 253-259
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Count ratings per movie
#   #B Normalize by max count to get 0-1 range
movie_counts = ratings['movieId'].value_counts()
max_count = movie_counts.max()

popularity_scores = {
    int(mid): float(count / max_count)
    for mid, count in movie_counts.items()
}
