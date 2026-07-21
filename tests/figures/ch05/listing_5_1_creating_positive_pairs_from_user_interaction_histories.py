# Figure - Listing 5.1: Creating positive pairs from user interaction histories
# Source: chapters/ch05.md lines 53-62
# Chapter: 5
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def create_pairs_for_user(group, label=1.0):
  positive_pairs = []  #A
  labels = []  #A
  for i in range(group.shape[0]):  #B
    for j in range(i + 1, group.shape[0]):  #B
      positive_pairs.append(
        (group.iloc[i]['movieId'], group.iloc[j]['movieId'])
      )  #C
      labels.append(label)  #D
  return positive_pairs, labels  #E

# Callout annotations (from the book):
#   #A Create empty lists for pairs and labels
#   #B Iterate over all unique pairs of items in the user's history
#   #C Append the pair of movie IDs
#   #D Label each pair as positive
#   #E Return all pairs and their labels
