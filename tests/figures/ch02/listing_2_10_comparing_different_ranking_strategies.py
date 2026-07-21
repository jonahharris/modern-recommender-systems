# Figure - Listing 2.10: Comparing different ranking strategies
# Source: chapters/ch02.md lines 340-347
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=fail)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
candidates = retrieve_similar_items(seed_movie_id, k=10)
candidates = filter_watched(candidates, test_user)
candidates = score_popularity(candidates)

weighted_results = []
for weight in [1.0, 0.5, 0.0]:
  ranked = rank_candidates(candidates, similarity_weight=weight, k=5)  #A
  weighted_results.append((weight, ranked))  #B

# Callout annotations (from the book):
#   #A Compute weighted combination of scores
#   #B Collect the results to make the chart below
