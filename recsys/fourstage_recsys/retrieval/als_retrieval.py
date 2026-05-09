"""
ALS (Alternating Least Squares) Collaborative Filtering Retrieval

Uses matrix factorization to provide:
- User-to-item recommendations based on history
- Item-to-item recommendations based on similarity
"""

from typing import List, Dict
from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval


class ALSRetrieval(Retrieval):
    """
    Collaborative filtering retrieval using ALS matrix factorization.
    Handles both item-to-item and user-to-item recommendations.
    """

    def __init__(
        self,
        model,
        user_item_matrix,
        user_map: Dict[int, int],
        item_map: Dict[int, int],
        k: int = 100
    ):
        """
        Initialize ALSRetrieval.

        Args:
            model: Trained ALS model from implicit library
            user_item_matrix: Sparse user-item interaction matrix
            user_map: Dict mapping userId to matrix row index
            item_map: Dict mapping movieId to matrix column index
            k: Number of candidates to retrieve
        """
        self.model = model
        self.user_item_matrix = user_item_matrix
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_item_map = {idx: iid for iid, idx in item_map.items()}
        self.k = k

    def retrieve(self, context: RecommendationContext) -> List[ScoredItem]:
        """
        Retrieve candidates based on user history or seed item.

        Args:
            context: RecommendationContext with user_id or seed_items

        Returns:
            List of ScoredItem objects sorted by score (descending)
        """
        all_candidates = {}

        if context.user_id and context.user_id in self.user_map:
            all_candidates = self.with_user_id(context)
        elif context.seed_items:
            all_candidates = self.with_seeds(context)

        candidates = []
        for idx, score in all_candidates.items():
            candidates.append(ScoredItem(
                item_id=self.reverse_item_map[idx],
                scores={'als_cf': float(score)}
            ))

        return sorted(candidates, key=lambda x: x.scores['als_cf'], reverse=True)

    def with_user_id(self, context: RecommendationContext) -> Dict:
        """
        Generate recommendations based on user's full history.

        Args:
            context: RecommendationContext with user_id

        Returns:
            Dict mapping item indices to similarity scores
        """
        user_idx = self.user_map[context.user_id]

        movie_indices, scores = self.model.recommend(
            user_idx,
            self.user_item_matrix[user_idx],
            N=self.k,
            filter_already_liked_items=True
        )

        return dict(zip(movie_indices, scores))

    def with_seeds(self, context: RecommendationContext) -> Dict:
        """
        Generate recommendations based on seed item(s).
        Handles both single item and multiple items.

        Args:
            context: RecommendationContext with seed_items

        Returns:
            Dict mapping item indices to similarity scores
        """
        all_candidates = {}

        for seed_id in context.seed_items:
            if seed_id not in self.item_map:
                continue

            item_idx = self.item_map[seed_id]
            similar_items = self.model.similar_items(
                item_idx,
                N=self.k,
                filter_items=[item_idx]
            )

            movie_indices, scores = similar_items

            for idx, score in zip(movie_indices, scores):
                if idx in all_candidates:
                    all_candidates[idx] = max(
                        all_candidates[idx],
                        float(score)
                    )
                else:
                    all_candidates[idx] = float(score)

        return all_candidates
