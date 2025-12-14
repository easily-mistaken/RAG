import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadatas: List[dict] = []

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        arr = np.array(embeddings).astype("float32")
        self.add_embeddings(arr, metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[dict]):
        if embeddings is None or embeddings.size == 0:
            print("[WARN] No embeddings provided to add.")
            return

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array (n_vectors, dim)")

        n, dim = embeddings.shape

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        else:
            existing_dim = self.index.d if hasattr(self.index, "d") else None
            if existing_dim is not None and existing_dim != dim:
                raise ValueError(f"Dimension mismatch: index expects {existing_dim}, embeddings are {dim}")

        self.index.add(embeddings)

        if metadatas:
            if len(metadatas) != n:
                raise ValueError("Length of metadatas must equal number of embeddings")
            self.metadatas.extend(metadatas)

        print(f"[INFO] Added {n} vectors (dim={dim}). Total metadata items: {len(self.metadatas)}")

    def save(self):
        if self.index is None:
            print("[WARN] No index to save.")
            return

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(faiss_path)
            with open(metadata_path, "rb") as f:
                self.metadatas = pickle.load(f)
            print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")
        else:
            print(f"[WARN] No existing Faiss index found in {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        if self.index is None:
            raise RuntimeError("Faiss index is not initialized. Add embeddings or load an index first.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            metadata = self.metadatas[idx] if (idx >= 0 and idx < len(self.metadatas)) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": metadata})
        print(f"[INFO] Retrieved {len(results)} results for the query.")
        return results

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        print(f"[INFO] Querying vector store with: '{query}'")
        query_embedding = self.model.encode([query]).astype("float32")
        return self.search(query_embedding, top_k=top_k)
