# Figure — Listing 2.10: Comparing different ranking strategies
# Source: chapters/ch02.md lines 340-347
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=fail)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Compute weighted combination of scores
#   #B Collect the results to make the chart below
candidates = retrieve_similar_items(seed_movie_id, k=10)
candidates = filter_watched(candidates, test_user)
candidates = add_popularity_scores(candidates)

weighted_results = []
for weight in [1.0, 0.5, 0.0]:
  ranked = rank_candidates(candidates, similarity_weight=weight, k=5)
  weighted_results.append((weight, ranked))
