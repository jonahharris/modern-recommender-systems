# Figure — Listing 1.5: Content-based recommender
# Source: chapters/ch01.md lines 216-230
# Chapter: 1
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Iterate through all comics in the catalog
#   #B Calculate the similarity between the current comic and each other comic
#   #C Store comic ID and its similarity score
#   #D Sort by similarity score (highest first)
#   #E Return the titles of the top k most similar comics
def content_based_recommender(comic_id, k=2):

  similarities = []

  for other_id in comic_tags:

    if comic_id != other_id:

      sim = content_similarity(comic_id, other_id)

      similarities.append((other_id, sim))

  similarities.sort(key=lambda x: x[1], reverse=True)

  return [catalogue[id] for id, sim in similarities[:k]]
