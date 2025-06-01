import requests
import time
import random
from typing import List, Dict, Any
import numpy as np
from .config import Config

class EmbeddingsAPI:
    def __init__(self, max_retries: int = None, base_delay: float = None):
        Config.validate()
        self.api_key = Config.GS_API_KEY
        self.api_url = Config.API_URL
        self.model = Config.EMBEDDING_MODEL
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.base_delay = base_delay or Config.BASE_RETRY_DELAY
        self.timeout = Config.REQUEST_TIMEOUT
          
    def get_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Get embeddings for a list of texts with retry logic
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Try to get embeddings for this batch with retries
            batch_embeddings = self._get_batch_embeddings_with_retry(batch, batch_num)
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting - small delay between requests
            if i + batch_size < len(texts):
                time.sleep(0.1)
                
        return all_embeddings
    
    def _get_batch_embeddings_with_retry(self, batch: List[str], batch_num: int) -> List[List[float]]:
        """
        Get embeddings for a batch with exponential backoff retry logic
        
        Args:
            batch: List of texts in this batch
            batch_num: Batch number for logging
            
        Returns:
            List of embeddings for the batch
        """
        data = {
            "model": self.model,
            "input": batch
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                
                # Extract embeddings from response
                if "data" in result:
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                elif "embeddings" in result:
                    batch_embeddings = result["embeddings"]
                else:
                    print(f"⚠️  Unexpected response structure for batch {batch_num}: {list(result.keys())}")
                    batch_embeddings = []
                
                if batch_embeddings and len(batch_embeddings) == len(batch):
                    if attempt > 0:
                        print(f"✅ Batch {batch_num} succeeded on attempt {attempt + 1}")
                    return batch_embeddings
                else:
                    print(f"⚠️  Batch {batch_num} returned {len(batch_embeddings)} embeddings for {len(batch)} texts")
                    
            except requests.exceptions.Timeout as e:
                print(f"⏱️  Timeout for batch {batch_num}, attempt {attempt + 1}: {e}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    print(f"🚦 Rate limited for batch {batch_num}, attempt {attempt + 1}")
                elif e.response.status_code >= 500:  # Server error
                    print(f"🔥 Server error for batch {batch_num}, attempt {attempt + 1}: {e.response.status_code}")
                else:
                    print(f"❌ HTTP error for batch {batch_num}, attempt {attempt + 1}: {e}")
                    # For client errors (4xx), don't retry
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        break
            except requests.exceptions.RequestException as e:
                print(f"🌐 Network error for batch {batch_num}, attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"💥 Unexpected error for batch {batch_num}, attempt {attempt + 1}: {e}")
            
            # If this wasn't the last attempt, wait before retrying
            if attempt < self.max_retries:
                # Exponential backoff with jitter
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Retrying batch {batch_num} in {delay:.1f} seconds...")
                time.sleep(delay)
        
        # If all retries failed, return zero embeddings
        print(f"💀 All retries failed for batch {batch_num}, using zero embeddings")
        zero_embedding = [0.0] * Config.VECTOR_DIMENSION
        return [zero_embedding] * len(batch)
    
    def get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * Config.VECTOR_DIMENSION
