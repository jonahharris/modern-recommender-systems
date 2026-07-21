# Figure - Listing 2.16: Attempting to recommend a new movie
# Source: chapters/ch02.md lines 589-594
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
new_movie_id = 209157
new_movie = movies[movies['movieId'] == new_movie_id]  #A

num_ratings = len(ratings[ratings['movieId'] == new_movie_id])  #B

candidates = retrieve_similar_items(new_movie_id, k=10)  #C

# Callout annotations (from the book):
#   #A Look up a movie with very few ratings
#   #B See how many we can find
#   #C Try to find similar items using collaborative filtering
