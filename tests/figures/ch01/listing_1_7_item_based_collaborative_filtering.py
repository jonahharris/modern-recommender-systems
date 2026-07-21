# Figure — Listing 1.7: Item-based Collaborative filtering
# Source: chapters/ch01.md lines 322-330
# Chapter: 1
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Dot product on a transposed matrix measures item co-consumption
#   #B Exclude self-similarity
#   #C Return indices of the top\_n most similar items
def find_similar_items(matrix: np.array, item_id: int, top_n: int = 3):

    item_sim = np.dot(matrix.T[item_id], matrix.T)

    item_sim[item_id] = 0

    most_similar_items = np.argsort(item_sim)[::-1][:top_n]

    return most_similar_items.tolist()
