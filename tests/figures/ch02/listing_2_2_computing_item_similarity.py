# Figure — Listing 2.2: Computing item similarity
# Source: chapters/ch02.md lines 125-127
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Transpose to get items × users, then compute item-item similarity
#   #B Keep result sparse to save memory
item_similarity = cosine_similarity(
  user_item_matrix.T,
  dense_output=False)
