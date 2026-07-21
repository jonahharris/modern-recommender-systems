# Figure - Listing 1.2: A uniform recommender
# Source: chapters/ch01.md lines 146-151
# Chapter: 1
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def uniform_recommender(k=2):  #A
  catalogue_ids = list(catalogue.keys())  #B
  random_keys = np.random.choice(catalogue_ids,
    size=k,
    replace=False)  #C
  return [catalog[key] for key in random_keys]  #D

# Callout annotations (from the book):
#   #A The uniform recommender method takes a parameter k, which indicates how many items it should recommend
#   #B The list of IDs representing the catalog of items
#   #C Choose k content items at random
#   #D Run through the selected item IDs and return the items
