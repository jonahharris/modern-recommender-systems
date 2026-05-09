import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from recsys.fourstage_recsys.retrieval.retrieval import Retrieval
from recsys.fourstage_recsys.recsys_context import RecommenderContext

class SentenceTransformerRetrieval(Retrieval):
  """
  Content-based retrieval using Sentence Transformers.
  
  Features:
  - Automatic caching
  - Incremental updates
  - Efficient similarity search
  - Framework integration
  """
  
  def __init__(
    self,
    model: SentenceTransformer,
    items_df: pd.DataFrame,
    text_column: str,
    item_id_column: str = 'movieId',
    cache_path: Optional[Path] = None,
    k: int = 100
  ):
    self.model = model
    self.items_df = items_df
    self.text_column = text_column
    self.item_id_column = item_id_column
    self.cache_path = cache_path
    self.k = k
    
    self.embeddings = None
    self.item_ids = None
    self.id_to_idx = {}
  
  def build_index(self, force_rebuild: bool = False):
    """
    Build or load embedding index.
    """
    # Try to load from cache
    if not force_rebuild and self.cache_path and self.cache_path.exists():
      if self._load_from_cache():
        return
    
    print("Building embedding index...")
    texts = self.items_df[self.text_column].tolist()
    self.item_ids = self.items_df[self.item_id_column].values
    
    start = time.time()
    self.embeddings = self.model.encode(
      texts,
      show_progress_bar=True,
      batch_size=32,
      convert_to_numpy=True
    )
    elapsed = time.time() - start
    
    print(f"Embedded {len(texts):,} items in {elapsed:.1f}s")
    print(f"Rate: {len(texts)/elapsed:.1f} items/second")
    
    # Build lookup index
    self.id_to_idx = {
      int(item_id): idx
      for idx, item_id in enumerate(self.item_ids)
    }
    
    # Save to cache
    if self.cache_path:
      self._save_to_cache()
  
  def _load_from_cache(self) -> bool:
    """
        Load embeddings from cache.
    """
    try:
      with open(self.cache_path, 'rb') as f:
        cache_data = pickle.load(f)
      
      self.embeddings = cache_data['embeddings']
      self.item_ids = cache_data['item_ids']
      self.id_to_idx = cache_data['id_to_idx']
      
      print(f"Loaded {len(self.item_ids):,} embeddings from cache")
      return True
    except Exception as e:
      print(f"Failed to load cache: {e}")
      return False
  
  def _save_to_cache(self):
    """
      Save embeddings to cache.
    """
    cache_data = {
      'embeddings': self.embeddings,
      'item_ids': self.item_ids,
      'id_to_idx': self.id_to_idx,
      'metadata': {
        'model_name': self.model._model_card_data.model_name,
        'created_at': pd.Timestamp.now().isoformat(),
        'num_items': len(self.item_ids)
      }
    }
    
    with open(self.cache_path, 'wb') as f:
      pickle.dump(cache_data, f)
    
    size_mb = self.cache_path.stat().st_size / 1024 / 1024
    print(f"Saved cache ({size_mb:.1f} MB)")
  
  def retrieve(self, context: RecommenderContext) -> List[Dict]:
    """
      Retrieve candidates based on seed items.
    """
    if not context.seed_items:
      return []
    
    all_candidates = {}
    
    for seed_id in context.seed_items:
      if seed_id not in self.id_to_idx:
        continue
      
      seed_idx = self.id_to_idx[seed_id]
      candidates = self._find_similar(seed_idx, k=self.k)
      
      for item_id, score in candidates:
        if item_id in all_candidates:
          all_candidates[item_id] = max(all_candidates[item_id], score)
        else:
          all_candidates[item_id] = score
    
    # Convert to list and sort
    result = [
      {
        'item_id': item_id,
        'score': score,
        'source': 'content_similarity'
      }
      for item_id, score in all_candidates.items()
    ]
    
    result.sort(key=lambda x: x['score'], reverse=True)
    return result[:self.k]
  
  def _find_similar(self, query_idx: int, k: int) -> List[tuple]:
    """
      Find k most similar items using vectorized similarity.
    """
    query_emb = self.embeddings[query_idx:query_idx+1]
    similarities = cosine_similarity(query_emb, self.embeddings)[0]
    
    similarities[query_idx] = -1
    
    top_indices = np.argsort(similarities)[-(k):][::-1]
    
    return [
      (int(self.item_ids[idx]), float(similarities[idx]))
      for idx in top_indices
    ]
  
  def add_items(self, new_items_df: pd.DataFrame):
    """
      Incrementally add new items without re-embedding everything.
    """
    new_texts = new_items_df[self.text_column].tolist()
    new_ids = new_items_df[self.item_id_column].values
    
    print(f"Adding {len(new_items_df)} new items...")
    new_embeddings = self.model.encode(
      new_texts,
      show_progress_bar=True,
      batch_size=32
    )
    
    # Append to existing
    self.embeddings = np.vstack([self.embeddings, new_embeddings])
    self.item_ids = np.concatenate([self.item_ids, new_ids])
    
    # Update index
    start_idx = len(self.id_to_idx)
    for i, item_id in enumerate(new_ids):
      self.id_to_idx[int(item_id)] = start_idx + i
    
    # Save updated cache
    if self.cache_path:
      self._save_to_cache()
    
    print(f"Index now contains {len(self.item_ids):,} items")
