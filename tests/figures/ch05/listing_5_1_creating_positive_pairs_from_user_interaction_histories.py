# Figure — Listing 5.1: Creating positive pairs from user interaction histories
# Source: chapters/ch05.md lines 53-62
# Chapter: 5
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Create empty lists for pairs and labels
#   #B Iterate over all unique pairs of items in the user's history
#   #C Append the pair of movie IDs
#   #D Label each pair as positive
#   #E Return all pairs and their labels
def create_pairs_for_user(group, label=1.0):
  positive_pairs = []
  labels = []
  for i in range(group.shape[0]):
    for j in range(i + 1, group.shape[0]):
      positive_pairs.append(
        (group.iloc[i]['movieId'], group.iloc[j]['movieId'])
      )
      labels.append(label)
  return positive_pairs, labels
