# Figure - Listing 1.7: Item-based Collaborative filtering
# Source: chapters/ch01.md lines 322-330
# Chapter: 1
# Category: standalone  (executable=True, expected=pass)
# CORRECTED for the book (see the with-figures branch for the original as-printed).
def find_similar_items(matrix: np.array, item_id: int, top_n: int = 3):

    item_sim = np.dot(matrix.T[item_id], matrix)  #A

    item_sim[item_id] = 0  #B

    most_similar_items = np.argsort(item_sim)[::-1][:top_n]  #C

    return most_similar_items.tolist()

# Callout annotations (from the book):
#   #A Dot product on a transposed matrix measures item co-consumption
#   #B Exclude self-similarity
#   #C Return indices of the top\_n most similar items
