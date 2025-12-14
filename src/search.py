import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_openai import OpenAI

load_dotenv()

class RAGSearch:
    def __init__(self,persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-4o-mini"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        metadata_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(metadata_path)):
            from src.data_loader import load_all_documents
            print("[INFO] Vector store not found, building from documents...")
            documents = load_all_documents("data")
            self.vectorstore.build_from_documents(documents)
        else:
            print("[INFO] Loading existing vector store...")
            self.vectorstore.load()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.llm = OpenAI(model_name=llm_model, openai_api_key=openai_api_key, temperature=0)
        print(f"[INFO] Initialized RAGSearch with LLM model: {llm_model}")
    
    def _extract_text_from_response(self, response) -> str:
        # Robustly handle different response shapes returned by various LLM client wrappers
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        # LangChain-style object may have .content or .text
        if hasattr(response, "content"):
            return response.content if isinstance(response.content, str) else str(response.content)
        if hasattr(response, "text"):
            return response.text if isinstance(response.text, str) else str(response.text)
        # OpenAI-like dict/obj with choices
        if hasattr(response, "choices"):
            try:
                choice = response.choices[0]
                if isinstance(choice, dict):
                    return choice.get("text") or choice.get("message", {}).get("content", "")
                return getattr(choice, "text", "") or getattr(choice, "message", None) and getattr(choice.message, "content", "")
            except Exception:
                pass

        if isinstance(response, dict):
            return response.get("content") or response.get("text") or str(response)
        return str(response)

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [res['metadata']['text'] for res in results if res['metadata']]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant information found."
        prompt = f"Summarize the following context for the query '{query}':\n\n{context}\n\nSummary:"
        response = self.llm.invoke(prompt)
        return self._extract_text_from_response(response)