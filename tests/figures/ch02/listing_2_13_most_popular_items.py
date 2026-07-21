# Figure — Listing 2.13: Most popular items
# Source: chapters/ch02.md lines 480-483
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Get most popular items
#   #B Get user history (only last 5 items)
#   #C Filter out watched movies from candidates
#   #D Reorder items according to the average similarity of the user history items
candidates = get_most_popular_movies(k=10)
user_movie_ids = list(get_user_history(user_id, k=5))
candidates = [candidate for candidate in candidates if candidate not in user_movie_ids]
recs = reorder_candidates_by_seed_similarity(candidates, user_movie_ids)
