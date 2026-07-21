# Figure - Listing 2.14: Four-stage recommender
# Source: chapters/ch02.md lines 510-515
# Chapter: 2
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def recommend_for_user(user_id, k=10):
  candidates = retrieve_candidates_for_user(user_id)  #A
  candidates = apply_filters(candidates, user_id)  #B
  candidates = add_additional_scores(candidates)  #C
  recommendations = rank_and_select_top_k(candidates, k)  #D
  return recommendations

# Callout annotations (from the book):
#   #A Stage 1: Retrieval
#   #B Stage 2: Filtering
#   #C Stage 3: Scoring
#   #D Stage 4: Ranking
