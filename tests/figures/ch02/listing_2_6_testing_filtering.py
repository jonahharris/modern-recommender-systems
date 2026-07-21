# Figure - Listing 2.6: Testing filtering
# Source: chapters/ch02.md lines 225-229
# Chapter: 2
# Category: needs-real-data  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
test_user = 1

candidates = retrieve_similar_items(seed_movie_id, k=100)  #A

filtered = filter_watched(candidates, test_user)  #B

# Callout annotations (from the book):
#   #A find items with high similarity
#   #B remove items the user has already consumed
