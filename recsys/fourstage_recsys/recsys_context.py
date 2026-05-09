from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class RecommendationContext(BaseModel):
  user_id: Optional[str] = None #A
  seed_items: Optional[List[str]] = None #B
  k: int = 10 #C
  filters: Dict[str, Any] = {}
  metadata: Dict[str, Any] = {}

  class Config:
    arbitrary_types_allowed = True
