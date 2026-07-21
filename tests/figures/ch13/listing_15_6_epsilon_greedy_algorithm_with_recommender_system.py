# Figure — Listing 15.6: Epsilon-greedy algorithm with recommender system
# Source: chapters/ch13.md lines 330-336
# Chapter: 13
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A if random number is less than epsilon then explore
#   #B select between new coldstart products
#   #C otherwise call the recommender systems.
if np.random.random() < epsilon:

  return select_a_cold_start_item()

else:

  return recsys_model.recs()
