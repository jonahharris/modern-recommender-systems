# Figure - Listing 2.2: Computing item similarity
# Source: chapters/ch02.md lines 125-127
# Chapter: 2
# Category: needs-prior  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
item_similarity = cosine_similarity(  #A
  user_item_matrix.T,  #B
  dense_output=False)

# Callout annotations (from the book):
#   #A Transpose to get items × users, then compute item-item similarity
#   #B Keep result sparse to save memory
