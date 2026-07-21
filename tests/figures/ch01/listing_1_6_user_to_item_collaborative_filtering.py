# Figure — Listing 1.6: User-to-item Collaborative filtering
# Source: chapters/ch01.md lines 280-302
# Chapter: 1
# Category: TBD
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Dot product measures overlap: users who consumed the same items get higher scores
#   #B Exclude self-similarity
#   #C Return indices of the top\_n most similar users
#   #D Mask of items the target user hasn't consumed (1 \= not consumed)
#   #E Count how many similar users consumed each item
#   #F Score each item: only items the user hasn't consumed get a nonzero score
def find_similar_users(matrix: np.array, user_id: int, top_n: int = 3):

    user_sim = np.dot(matrix[user_id], matrix.T)

    user_sim[user_id] = 0

    most_similar_users = np.argsort(user_sim)[::-1][:top_n]

    return most_similar_users.tolist()

def get_recommendations(matrix: np.array, user_id: int,

                        sim_users: list, top_n: int = 3):

    not_consumed = (matrix[user_id] == 0).astype(int)

    sim_user_counts = matrix[sim_users].sum(axis=0)

    scores = not_consumed * sim_user_counts

    rec_ids = np.argsort(scores)[::-1][:top_n]

    return [catalog[id] for id in rec_ids]
