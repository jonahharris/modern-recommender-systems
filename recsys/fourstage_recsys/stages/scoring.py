from abc import ABC, abstractmethod
from typing import List

from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem 
class Scorer(ABC):
  @abstractmethod
  def score(
    self, 
    candidates: List[ScoredItem], 
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
