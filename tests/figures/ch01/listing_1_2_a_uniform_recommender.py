# Figure - Listing 1.2: A uniform recommender
# Source: chapters/ch01.md lines 146-151
# Chapter: 1
# Category: needs-prior  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
import numpy as np  #A

def uniform_recommender(k=2):  #B
  catalogue_ids = list(catalogue.keys())  #C
  random_keys = np.random.choice(catalogue_ids,
    size=k,
    replace=False)  #D
  return [catalogue[key] for key in random_keys]  #E

# Callout annotations (from the book):
#   #A NumPy powers the random selection below
#   #B The uniform recommender method takes a parameter k, which indicates how many items it should recommend
#   #C The list of IDs representing the catalog of items
#   #D Choose k content items at random
#   #E Run through the selected item IDs and return the items
