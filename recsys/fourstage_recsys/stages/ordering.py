
from abc import ABC, abstractmethod
from typing import List

from recsys.fourstage_recsys.recsys_context import RecommendationContext
from recsys.fourstage_recsys.item_context import ScoredItem

class Ordering(ABC):
  @abstractmethod
  def order(
    self, 
    filtered_items: List[ScoredItem],
    context: RecommendationContext
  ) -> List[ScoredItem]:
    pass
