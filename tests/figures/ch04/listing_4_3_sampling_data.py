# Figure - Listing 4.3: Sampling data
# Source: chapters/ch04.md lines 198-201
# Chapter: 4
# Category: needs-training  (executable=False, expected=skip)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
sampled_users = ratings['userId'].sample(frac=0.5,
                                  random_state=42).unique()  #A

ratings = ratings[ratings['userId'].isin(sampled_users)]  #B

# Callout annotations (from the book):
#   #A sample 50% of the users
#   #B Filter the ratings to keep only those from the sampled users
