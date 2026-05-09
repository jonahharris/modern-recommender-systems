"""
ALS (Alternating Least Squares) Model Wrapper

Provides training, saving, and loading of implicit ALS models with caching.
"""

import pickle
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class ALSModel:
    """
    Wrapper for implicit ALS model with persistence and training utilities.
    
    Handles:
    - Creating and training ALS models
    - Saving/loading models to/from cache
    - Managing user/item mappings
    """
    
    def __init__(
        self,
        factors: int = 50,
        iterations: int = 20,
        regularization: float = 0.01,
        random_state: int = 42,
        cache_dir: Path = None
    ):
        """
        Initialize ALS model.
        
        Args:
            factors: Number of latent factors
            iterations: Number of training iterations
            regularization: Regularization parameter
            random_state: Random seed for reproducibility
            cache_dir: Directory to save/load models. Defaults to data/cache
        """
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.random_state = random_state
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = Path("data/cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and mappings
        self.model = None
        self.user_map = None
        self.item_map = None
        self.user_item_matrix = None
        
    def _create_model(self) -> AlternatingLeastSquares:
        """Create a fresh ALS model instance."""
        return AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            calculate_training_loss=False,
            num_threads=4
        )
    
    def train(
        self,
        user_item_matrix: csr_matrix,
        user_map: dict,
        item_map: dict,
        show_progress: bool = True
    ) -> Tuple[AlternatingLeastSquares, dict, dict]:
        """
        Train ALS model on user-item interactions.
        
        Args:
            user_item_matrix: Sparse CSR matrix of user-item interactions
            user_map: Dict mapping userId to matrix row index
            item_map: Dict mapping itemId to matrix column index
            show_progress: Whether to show training progress
            
        Returns:
            Tuple of (trained_model, user_map, item_map)
        """
        print(f"Training ALS model ({self.factors} factors, {self.iterations} iterations)...")
        print(f"Matrix shape: {user_item_matrix.shape}")
        
        # Create and train model
        self.model = self._create_model()
        
        start_time = time.time()
        self.model.fit(user_item_matrix, show_progress=show_progress)
        elapsed = time.time() - start_time
        
        print(f"Training complete in {elapsed:.1f}s")
        print(f"  User factors: {self.model.user_factors.shape}")
        print(f"  Item factors: {self.model.item_factors.shape}")
        
        # Store mappings
        self.user_map = user_map
        self.item_map = item_map
        self.user_item_matrix = user_item_matrix
        
        return self.model, user_map, item_map
    
    def save(self, name: str = "als_model") -> Path:
        """
        Save trained model and metadata to cache directory.
        
        Args:
            name: Name of the model (without extension)
            
        Returns:
            Path to saved model file
        """
        if self.model is None:
            raise ValueError("No trained model to save. Call train() first.")
        
        model_path = self.cache_dir / f"{name}.pkl"
        
        # Prepare data for pickling
        cache_data = {
            'model': self.model,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'metadata': {
                'factors': self.factors,
                'iterations': self.iterations,
                'regularization': self.regularization,
                'random_state': self.random_state,
                'timestamp': time.time()
            }
        }
        
        # Save to disk
        with open(model_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model saved to {model_path} ({size_mb:.1f} MB)")
        
        return model_path
    
    @staticmethod
    def load(name: str = "als_model", cache_dir: Path = None) -> Tuple:
        """
        Load trained model from cache directory.
        
        Args:
            name: Name of the model (without extension)
            cache_dir: Directory containing cached models. Defaults to data/cache
            
        Returns:
            Tuple of (model, user_map, item_map, metadata)
        """
        if cache_dir is None:
            cache_dir = Path("data/cache")
        
        model_path = Path(cache_dir) / f"{name}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load from disk
        with open(model_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        model = cache_data['model']
        user_map = cache_data['user_map']
        item_map = cache_data['item_map']
        metadata = cache_data.get('metadata', {})
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"Model loaded from {model_path} ({size_mb:.1f} MB)")
        print(f"  Factors: {metadata.get('factors', 'unknown')}")
        print(f"  Iterations: {metadata.get('iterations', 'unknown')}")
        print(f"  User map size: {len(user_map):,}")
        print(f"  Item map size: {len(item_map):,}")
        
        return model, user_map, item_map, metadata
    
    @staticmethod
    def model_exists(name: str = "als_model", cache_dir: Path = None) -> bool:
        """Check if a cached model exists."""
        if cache_dir is None:
            cache_dir = Path("data/cache")
        
        model_path = Path(cache_dir) / f"{name}.pkl"
        return model_path.exists()
