# Figure - Listing 2.22: Using Three-signal ranker
# Source: chapters/ch02.md lines 880-888
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
candidates = retrieve_hybrid(seed_movie_id, k=100)  #A
Candidates = filter_watched(candidates)  #B
candidates = add_popularity_scores(candidates)  #C

ranked = rank_three_signals(
    candidates,
    content_weight=0.3,
    behavioral_weight=0.5
  )  #D

# Callout annotations (from the book):
#   #A Retrieve items based on both content and behavorial similarity
#   #B filter what the user has watched already
#   #C Scoring the items based on popularity
#   #D order all the items based on a weighted average of the scores.
