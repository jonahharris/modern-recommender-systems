# Figure - Listing 2.4: Testing the Retrieval function
# Source: chapters/ch02.md lines 171-173
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
seed_movie_id = 1270
seed_title = movies[movies['movieId'] == seed_movie_id]['title'].values[0]  #A
candidates = retrieve_similar_items(seed_movie_id, k=10)  #B

# Callout annotations (from the book):
#   #A Get the title
#   #B Call retrieval system
