# Figure - Listing 4.1: Splitting data into train and test data
# Source: chapters/ch04.md lines 159-162
# Chapter: 4
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)  #A
split_idx = int(len(ratings_sorted) * 0.9)  #B
train_val_ratings = ratings_sorted.iloc[:split_idx].copy()  #C
test_ratings = ratings_sorted.iloc[split_idx:].copy()

# Callout annotations (from the book):
#   #A Sort ratings by timestamp
#   #B Calculate split point (90% for train, 10% for test)
#   #C Split the data temporally
