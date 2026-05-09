from scipy.sparse import csr_matrix
import numpy as np

def create_user_item_matrix(ratings_df, 
                            user_col='userId', 
                            item_col='movieId', 
                            value_col='confidence'):
    """
    Convert ratings DataFrame to sparse CSR matrix.

    Returns:
        matrix: CSR matrix (users × items)
        user_id_map: dict mapping userId to matrix row index
        item_id_map: dict mapping movieId to matrix column index
    """
    user_ids = ratings_df[user_col].unique()
    item_ids = ratings_df[item_col].unique()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}

    row_indices = ratings_df[user_col].map(user_id_map)
    col_indices = ratings_df[item_col].map(item_id_map)

    values = ratings_df[value_col].values

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(user_ids), len(item_ids))
    )

    return matrix, user_id_map, item_id_map