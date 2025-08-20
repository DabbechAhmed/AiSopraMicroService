from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os
import threading
import time
from typing import List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class OptimizedAIModelManager:
    """High-performance AI Model Manager with async support"""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_loaded = False
    _executor = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OptimizedAIModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        # Create thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_model")
        logger.info("OptimizedAIModelManager initialized")
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        if self._model_loaded and self._model is not None:
            return self._model
        
        with self._lock:
            if self._model_loaded and self._model is not None:
                return self._model
            
            try:
                from app.core.config import settings
                
                os.makedirs(settings.model_cache_dir, exist_ok=True)
                
                logger.info(f"🔥 Loading model: {settings.model_name}")
                start_time = time.time()
                
                # Load model with optimizations
                self._model = SentenceTransformer(
                    settings.model_name,
                    cache_folder=settings.model_cache_dir,
                    device='cpu'  # Explicit CPU usage for consistency
                )
                
                # Optimize model for inference
                self._model.eval()
                
                load_time = time.time() - start_time
                self._model_loaded = True
                
                logger.info(f"✅ Model loaded and optimized in {load_time:.2f}s")
                return self._model
                
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                self._model_loaded = False
                raise
    
    def get_model(self) -> SentenceTransformer:
        """Get the cached model instance"""
        if not self._model_loaded or self._model is None:
            self._load_model_sync()
        return self._model
    
    def _calculate_similarity_sync(self, text1: str, text2: str) -> float:
        """Synchronous similarity calculation"""
        try:
            if not text1 or not text2:
                return 0.0
            
            model = self.get_model()
            
            # Fast encoding with optimizations
            with threading.Lock():  # Ensure thread safety
                embeddings = model.encode(
                    [text1, text2], 
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Pre-normalize for faster cosine similarity
                )
            
            # Fast cosine similarity with normalized embeddings
            similarity = np.dot(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    async def calculate_similarity_async(self, text1: str, text2: str) -> float:
        """Async similarity calculation using thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._calculate_similarity_sync, 
            text1, 
            text2
        )
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Synchronous similarity calculation for backward compatibility"""
        return self._calculate_similarity_sync(text1, text2)
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready"""
        return self._model_loaded and self._model is not None
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)

# Create global instance
ai_model_manager = OptimizedAIModelManager()