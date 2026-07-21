# Figure — Listing 2.11: Complete recommendation function
# Source: chapters/ch02.md lines 372-405
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Get user's watch history
#   #B Handle cold start: return popular movies
#   #C Aggregate candidates from the user's recent watches
#   #D For each recent movie, get similar items
#   #E Keep the highest similarity score for each candidate
#   #F Convert to list of candidates
#   #G Add popularity scores
#   #H Filter out watched movies
#   #I Rank and return top k
def recommend_for_user(user_id, k=10, similarity_weight=0.7):
    user_history = get_user_history(user_id)

    if len(user_history) == 0:
        popular_movies = movie_counts.head(k).index.tolist()
        return [{'movie_id': int(mid)} for mid in popular_movies]

    all_candidates = {}
    recent_movies = list(user_history)[-20:]

    for movie_id in recent_movies:
        candidates = retrieve_similar_items(movie_id, k=50)

        for item in candidates:
            mid = item['movie_id']
            score = item['similarity']

            if mid in all_candidates:
                all_candidates[mid] = max(all_candidates[mid], score)
            else:
                all_candidates[mid] = score
    filtered = filter_watched(candidates, user_id)

    candidates = [
        {'movie_id': mid, 'similarity': score}
        for mid, score in all_candidates.items()
    ]

    candidates = add_popularity_scores(candidates)


    ranked = rank_candidates(filtered, similarity_weight, k)

    return ranked
