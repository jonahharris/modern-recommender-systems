from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem
from recsys.fourstage_recsys.stages.ordering import Ordering

class WeightedRanker(Ordering):
    def __init__(self, weights=None):
        """
        Rank candidates by a weighted combination of scores.
 
        Args:
            weights: dict mapping score names to weights, e.g.
                     {'similarity': 0.7, 'popularity': 0.3}
        """
        self.weights = weights or {'similarity': 0.7, 'popularity': 0.3}
 
    def order(self, candidates: list[ScoredItem], context: RecommendationContext) -> list[ScoredItem]:
        for item in candidates:
            item['final_score'] = sum(
                weight * item.get(score_name, 0.0)
                for score_name, weight in self.weights.items()
            ) #A
 
        ranked = sorted(
            candidates,
            key=lambda x: x['final_score'],
            reverse=True
        ) #B
 
        return ranked[:context.k] if context.k else ranked
 
#A Compute weighted combination of all available scores
#B Sort by final score descending
 