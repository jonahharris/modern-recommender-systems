# Figure - Listing 2.5: Filtering watched movies
# Source: chapters/ch02.md lines 203-215
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def get_user_history(user_id):  #A
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    return set(int(mid) for mid in user_movies)

def filter_watched(candidates, user_id):  #B
    user_history = get_user_history(user_id)

    filtered = [
        item for item in candidates
        if item['movie_id'] not in user_history
    ]

    return filtered

# Callout annotations (from the book):
#   #A Get set of movies the user has watched
#   #B Keep only unwatched movies
