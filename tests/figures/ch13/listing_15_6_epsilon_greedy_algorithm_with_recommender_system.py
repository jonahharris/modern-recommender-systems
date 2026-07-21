# Figure - Listing 15.6: Epsilon-greedy algorithm with recommender system
# Source: chapters/ch13.md lines 330-336
# Chapter: 13
# Category: pseudocode  (executable=False, expected=skip)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def select_recommendation(epsilon):
    if np.random.random() < epsilon:  #A
        return select_a_cold_start_item()  #B
    else:
        return recsys_model.recs()  #C

# Callout annotations (from the book):
#   #A if random number is less than epsilon then explore
#   #B select between new coldstart products
#   #C otherwise call the recommender systems.
