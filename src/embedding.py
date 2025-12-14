import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = SentenceTransformer(model_name)
        print(f"[INFO] Initialized EmbeddingPipeline with model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating Embeddings for {len(texts)} chunks...")
        embeddings = self.model_name.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings