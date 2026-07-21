# Figure - Listing 2.11: Complete recommendation function
# Source: chapters/ch02.md lines 372-405
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def recommend_for_user(user_id, k=10, similarity_weight=0.7):
    user_history = get_user_history(user_id)  #A

    if len(user_history) == 0:  #B
        popular_movies = movie_counts.head(k).index.tolist()
        return [{'movie_id': int(mid)} for mid in popular_movies]

    all_candidates = {}  #C
    recent_movies = list(user_history)[-20:]

    for movie_id in recent_movies:
        candidates = retrieve_similar_items(movie_id, k=50)  #D

        for item in candidates:
            mid = item['movie_id']
            score = item['similarity']

            if mid in all_candidates:
                all_candidates[mid] = max(all_candidates[mid], score)  #E
            else:
                all_candidates[mid] = score

    candidates = [
        {'movie_id': mid, 'similarity': score}
        for mid, score in all_candidates.items()
    ]  #F

    candidates = score_popularity(candidates)  #G

    candidates = filter_watched(candidates, user_id)  #H

    ranked = rank_candidates(candidates, similarity_weight, k)  #I

    return ranked

# Callout annotations (from the book):
#   #A Get user's watch history
#   #B Handle cold start: return popular movies
#   #C Aggregate candidates from the user's recent watches
#   #D For each recent movie, get similar items
#   #E Keep the highest similarity score for each candidate
#   #F Convert to list of candidates
#   #G Add popularity scores
#   #H Filter out watched movies
#   #I Rank and return top k
