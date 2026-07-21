# Figure — Listing 4.1: Splitting data into train and test data
# Source: chapters/ch04.md lines 159-162
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Sort ratings by timestamp
#   #B Calculate split point (90% for train, 10% for test)
#   #C Split the data temporally
ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(ratings_sorted) * 0.9)
train_val_ratings = ratings_sorted.iloc[:split_idx].copy()
test_ratings = ratings_sorted.iloc[split_idx:].copy()
