# Figure — Listing 2.4: Testing the Retrieval function
# Source: chapters/ch02.md lines 171-173
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Get the title
#   #B Call retrieval system
seed_movie_id = 1270
seed_title = movies[movies['movieId'] == seed_movie_id]['title'].values[0]
candidates = retrieve_similar_items(seed_movie_id, k=10)
