# Figure — Listing 2.14: Four-stage recommender
# Source: chapters/ch02.md lines 510-515
# Chapter: 2
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Stage 1: Retrieval
#   #B Stage 2: Filtering
#   #C Stage 3: Scoring
#   #D Stage 4: Ranking
def recommend_for_user(user_id, k=10):
  candidates = retrieve_candidates_for_user(user_id)
  candidates = apply_filters(candidates, user_id)
  candidates = add_additional_scores(candidates)
  recommendations = rank_and_select_top_k(candidates, k)
  return recommendations
