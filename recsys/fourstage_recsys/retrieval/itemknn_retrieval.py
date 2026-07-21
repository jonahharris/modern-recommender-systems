import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from recsys.fourstage_recsys.item_context import ScoredItem

class ItemKNNRetrieval:
    def __init__(self, ratings):
        
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

        self.item_similarity = None
        
        user_item_matrix = self._prepare_data(ratings)
        self._compute_item_similarity(user_item_matrix)
    
    def _compute_item_similarity(self, user_item_matrix):
        self.item_similarity = cosine_similarity(
            user_item_matrix.T,
            dense_output=False  #A
        )
        print(f"item_similarity shape: {self.item_similarity.shape}")

#A Keep result sparse to save memory — dense would require ~26 GB for ml-25m
        
    def _prepare_data(self, ratings):
        user_ids = ratings['userId'].unique()
        movie_ids = ratings['movieId'].unique()
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)} #A
        self.movie_to_idx = {str(mid): idx for idx, mid in enumerate(movie_ids)} #A
        self.idx_to_movie = {idx: mid for mid, idx in self.movie_to_idx.items()} #A

        rows = [self.user_to_idx[uid] for uid in ratings['userId']] #B
        cols = [self.movie_to_idx[str(mid)] for mid in ratings['movieId']] #B
        data = [1] * len(ratings) #B

        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(movie_ids))
        ) #C
        print(f"user_item_matrix shape: {user_item_matrix.shape}")
        return user_item_matrix
    
    def retrieve_similar_items(self, seed_ids: list[str], k=100) -> list[ScoredItem]: #A
        if not seed_ids:
            print("No seed items provided.")
            return []

        valid_ids = [str(sid) for sid in seed_ids if str(sid) in self.movie_to_idx]
        if not valid_ids:
            print("No valid seed items found.")
            return []

        seed_indices = list({self.movie_to_idx[sid] for sid in valid_ids})
        similarities = np.asarray(self.item_similarity[seed_indices].mean(axis=0)).flatten()
        top_indices = [
            idx for idx in np.argsort(similarities)[::-1]
            if idx not in seed_indices
        ][:k]
        
        candidates = []
        for idx in top_indices:
            candidates.append( ScoredItem(
                item_id=self.idx_to_movie[idx],
                scores={'similarity': float(similarities[idx])}
            ))
        
        return candidates