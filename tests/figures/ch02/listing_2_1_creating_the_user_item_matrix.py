# Figure — Listing 2.1: Creating the user-item matrix
# Source: chapters/ch02.md lines 94-107
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Create mappings from IDs to matrix indices
#   #B Build lists of row indices, column indices, and values
#   #C Create sparse matrix (only stores non-zero values)
#   #D Check matrix properties
user_ids = ratings['userId'].unique()
movie_ids = ratings['movieId'].unique()
user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
idx_to_movie = {idx: mid for mid, idx in movie_to_idx.items()}

rows = [user_to_idx[uid] for uid in ratings['userId']]
cols = [movie_to_idx[mid] for mid in ratings['movieId']]
data = [1] * len(ratings)

user_item_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(user_ids), len(movie_ids))
)
