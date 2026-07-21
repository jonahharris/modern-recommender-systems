# Figure — Listing 1.2: A uniform recommender
# Source: chapters/ch01.md lines 146-156
# Chapter: 1
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
def uniform_recommender(k=2):
  catalogue_ids = list(catalogue.keys())
  random_keys = np.random.choice(catalogue_ids,
    size=k,
    replace=False)
  return [catalog[key] for key in random_keys]

#A The uniform recommender method takes a parameter k, which indicates how many items it should recommend
#B The list of IDs representing the catalog of items
#C Choose k content items at random
#D Run through the selected item IDs and return the items
