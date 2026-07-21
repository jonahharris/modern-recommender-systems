# Figure — Listing 7.1: Discriminative scoring: iterate and rank
# Source: chapters/ch09.md lines 36-43
# Chapter: 9
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
item_scores = []
for item in catalog:
  item_scores.append(model(user, item))
return top_k_items(item_scores)

#A Iterate over all possible items
#B Score each item using a model, with the user as context
#C Return the k highest-scored items
