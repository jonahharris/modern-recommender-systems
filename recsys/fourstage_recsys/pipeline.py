from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.stages.filtering import Filtering
from recsys.fourstage_recsys.stages.scoring import Scorer
from recsys.fourstage_recsys.stages.ordering import Ordering

class FourStageRecommender:
    def __init__(
        self,
        retrieval: Retrieval,
        filter: Filtering,          
        scorer: Scorer,
        ordering: Ordering,
    ):
        self.retrieval = retrieval
        self.filter = filter
        self.scorer = scorer
        self.ordering = ordering
 
    def recommend(self, context: RecommendationContext):
        """Run the four-stage pipeline: 
            retrieve → filter → score → order.
        """
        
        candidates = self.retrieval.retrieve_similar_items(context.seed_item, k=100) #A
        candidates = self.filter.filter(candidates, context) #B
        candidates = self.scorer.score(candidates, context) #C
        ordered = self.ordering.order(candidates, context) #D
        
        return ordered[:context.k]
 
#A Stage 1: Retrieval
#B Stage 2: Filtering
#C Stage 3: Scoring
#D Stage 4: Ordering