from typing import List, Dict, Any
from chromadb import Client
from chromadb.config import Settings
import numpy as np

class ComplaintVectorStore:
    def __init__(self, collection_name: str = "complaint_embeddings"):
        """Initialize ChromaDB vector store"""
        self.client = Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./vector_store"
        ))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Add embeddings and their metadata to the store"""
        ids = [str(i) for i in range(len(embeddings))]
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search_similar(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find similar complaints based on embedding"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_by_id(self, id: str) -> Dict[str, Any]:
        """Retrieve specific embedding by ID"""
        return self.collection.get(ids=[id])