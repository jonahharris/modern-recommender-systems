from typing import List

from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem 
from recsys.fourstage_recsys.stages.scoring import Scorer

class PopularityScoring(Scorer):
    def __init__(self, ratings):
        movie_counts = ratings['movieId'].value_counts() #A
        max_count = movie_counts.max() #B

        self.popularity_scores = {
            int(mid): float(count / max_count)
            for mid, count in movie_counts.items()
        }
    
    def score_popularity(self, movie_id, context: RecommendationContext=None):
        return self.popularity_scores.get(movie_id, 0.0)
    
    def score(self, candidates: List[ScoredItem], context: RecommendationContext=None):
        for item in candidates:
            item['popularity'] = self.score_popularity(item['movie_id'], context)
        return candidates