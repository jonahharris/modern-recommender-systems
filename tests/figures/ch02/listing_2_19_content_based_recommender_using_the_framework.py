# Figure - Listing 2.19: Content-based recommender using the framework
# Source: chapters/ch02.md lines 743-782
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def recommend_content_based(user_id, k=10, content_weight=0.7):
  user_history = get_user_history(user_id)

  if len(user_history) == 0:
    popular_movies = movie_counts.head(k).index.tolist()
    return [{'movie_id': int(mid)} for mid in popular_movies]

  all_candidates = {}
  recent_movies = list(user_history)[-20:]

  for movie_id in recent_movies:
    candidates = retrieve_similar_by_content(movie_id, k=50)  #A

    for item in candidates:
      mid = item['movie_id']
      score = item['content_similarity']
      if mid in all_candidates:
        all_candidates[mid] = max(all_candidates[mid], score)
      else:
        all_candidates[mid] = score

  candidates = [
    {'movie_id': mid, 'content_similarity': score}
    for mid, score in all_candidates.items()
  ]  #B
  candidates = filter_watched(candidates, user_id)  #C

  candidates = score_popularity(candidates)  #D

  for item in candidates:  #E
    item['final_score'] = (
      content_weight * item['content_similarity'] +
      (1 - content_weight) * item['popularity']
      )

  ranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)

  return ranked[:k]

# Callout annotations (from the book):
#   #A Stage 1: Retrieval using content similarity
#   #B Aggregate from the user's recent watches
#   #C Stage 2: Filtering watched movies
#   #D Stage 3: Scoring with popularity
#   #E Stage 4: Ranking by weighted combination
