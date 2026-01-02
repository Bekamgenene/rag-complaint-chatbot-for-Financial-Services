from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class ComplaintEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedder with specified model"""
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text chunks in batches"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def embed_dataframe(self, df: 'pd.DataFrame', text_column: str) -> np.ndarray:
        """Generate embeddings from DataFrame"""
        texts = df[text_column].tolist()
        return self.embed_texts(texts)