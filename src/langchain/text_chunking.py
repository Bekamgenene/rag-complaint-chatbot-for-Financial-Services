from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ComplaintTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize text splitter with specified chunk parameters"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )

    def split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        chunks = self.splitter.split_text(text)
        return [{
            "text": chunk,
            "metadata": metadata
        } for chunk in chunks]

    def split_dataframe(self, df: 'pd.DataFrame', text_column: str) -> List[Dict[str, Any]]:
        """Split text from DataFrame rows"""
        all_chunks = []
        for _, row in df.iterrows():
            metadata = row.to_dict()
            chunks = self.split_text(row[text_column], metadata)
            all_chunks.extend(chunks)
        return all_chunks