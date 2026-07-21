# Figure - Listing 5.2: Creating negative pairs by random sampling
# Source: chapters/ch05.md lines 75-89
# Chapter: 5
# Category: standalone  (executable=True, expected=pass)
# Verbatim from the book; code lines keep their inline #A/#B callout markers.
def create_negative_pairs(
  positive_pairs: list[tuple],
  all_movie_ids: list,
  num_neg_per_pos: int = 5
):
  neg_pairs = []  #A
  labels = []  #A
  for movie1, movie2 in positive_pairs:  #B
    for _ in range(num_neg_per_pos):  #C
      movie3 = random.choice(all_movie_ids)  #D
      while movie3 == movie1 or movie3 == movie2:  #E
        movie3 = random.choice(all_movie_ids)  #E
      neg_pairs.append((movie1, movie_to_idx[movie3]))  #F
      labels.append(0.0)  #G
  return neg_pairs, labels

# Callout annotations (from the book):
#   #A Empty lists for negative pairs and labels
#   #B Iterate through every positive pair
#   #C Generate multiple negatives per positive pair
#   #D Randomly select a movie from the full catalog
#   #E Ensure the sampled movie is not in the original positive pair
#   #F Add the negative pair
#   #G Label as negative (0.0)
