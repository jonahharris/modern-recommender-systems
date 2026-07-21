# Figure - Listing 7.1: Discriminative scoring: iterate and rank
# Source: chapters/ch09.md lines 36-43
# Chapter: 9
# Category: pseudocode  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
item_scores = []
for item in catalog:  #A
  item_scores.append(model(user, item))  #B
return top_k_items(item_scores)  #C

#A Iterate over all possible items
#B Score each item using a model, with the user as context
#C Return the k highest-scored items
