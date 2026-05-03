from abc import ABC, abstractmethod
from typing import List

from recsys.fourstage_recsys.pipeline import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem

class Filtering(ABC):
  
  @abstractmethod
  def filter(
    self, 
    scored_items: List[ScoredItem],
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
