from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported document types from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON 
    """

    # Use project root as data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    all_documents = []

    # PDF files
    pdf_files = list(data_path.rglob("*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF file: {pdf_file}")
        try:
             loader = PyPDFLoader(str(pdf_file))
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {pdf_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load PDF file {pdf_file}: {e}")

    # TXT files
    txt_files = list(data_path.rglob("*.txt"))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT file: {txt_file}")
        try:
             loader = TextLoader(str(txt_file), encoding="utf-8")
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {txt_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load TXT file {txt_file}: {e}")

    # CSV files
    csv_files = list(data_path.rglob("*.csv"))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV file: {csv_file}")
        try:
             loader = CSVLoader(str(csv_file), encoding="utf-8")
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {csv_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load CSV file {csv_file}: {e}")

    # Excel files
    excel_files = list(data_path.rglob("*.xlsx")) + list(data_path.rglob("*.xls"))
    print(f"[DEBUG] Found {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")
    for excel_file in excel_files:
        print(f"[DEBUG] Loading Excel file: {excel_file}")
        try:
             loader = UnstructuredExcelLoader(str(excel_file))
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {excel_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load Excel file {excel_file}: {e}")

    # Word files
    docx_files = list(data_path.rglob("*.docx"))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word file: {docx_file}")
        try:
             loader = Docx2txtLoader(str(docx_file))
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {docx_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load Word file {docx_file}: {e}")
    
    # JSON files
    json_files = list(data_path.rglob("*.json"))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON file: {json_file}")
        try:
             loader = JSONLoader(str(json_file))
             documents = loader.load()
             print(f"[DEBUG] Loaded {len(documents)} documents from {json_file}")
             all_documents.extend(documents)
        except Exception as e:
             print(f"[ERROR] Failed to load JSON file {json_file}: {e}")
    
    print(f"[INFO] Total documents loaded: {len(all_documents)}")
    return all_documents