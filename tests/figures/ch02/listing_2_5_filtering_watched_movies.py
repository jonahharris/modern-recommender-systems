# Figure — Listing 2.5: Filtering watched movies
# Source: chapters/ch02.md lines 203-215
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Get set of movies the user has watched
#   #B Keep only unwatched movies
def get_user_history(user_id):
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    return set(int(mid) for mid in user_movies)

def filter_watched(candidates, user_id):
    user_history = get_user_history(user_id)

    filtered = [
        item for item in candidates
        if item['movie_id'] not in user_history
    ]

    return filtered
