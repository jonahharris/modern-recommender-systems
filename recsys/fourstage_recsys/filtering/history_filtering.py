from typing import List

import pandas as pd
import numpy as np

from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.stages.filtering import Filtering

class HistoryFiltering(Filtering): #A
    def __init__(self, ratings):
        self.ratings = ratings

    def get_user_history(self, user_id, k=None) -> set[str]: #A
        user_rows = self.ratings[self.ratings["userId"].astype(str) == str(user_id)]
        user_rows = user_rows.sort_values("timestamp")
        movie_ids = [str(mid) for mid in user_rows["movieId"].tolist()]
        return set(movie_ids) if k is None else set(movie_ids[-k:])

    def filter(self, candidates: List[ScoredItem], context) -> List[ScoredItem]: #B
        user_id = getattr(context, "user_id", context) #B accept a RecommendationContext or a raw user id
        user_history = self.get_user_history(user_id)
        
        filtered = [
            item for item in candidates
            if item.item_id not in user_history
        ]    
        return filtered