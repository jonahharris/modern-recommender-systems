# Figure — Listing 4.3: Sampling data
# Source: chapters/ch04.md lines 198-201
# Chapter: 4
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A sample 50% of the users
#   #B Filter the ratings to keep only those from the sampled users
sampled_users = ratings['userId'].sample(frac=0.5,
                                  random_state=42).unique()

ratings = ratings[ratings['userId'].isin(sampled_users)]
