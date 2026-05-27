import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Configuration ───────────────────────────────────────────────
ENABLE_CONTENT_EXTRACTION = True  # Set to False to skip content extraction (saves API calls)

# ── Paths ───────────────────────────────────────────────────────
PDF_PATHS = [
    r"c:\Users\SaiJayanthMulugu\Downloads\database-data-warehousing-guide.pdf",
    r"c:\Users\SaiJayanthMulugu\Downloads\IJRTI2304061(AI resource paper).pdf",
    r"C:\Users\SaiJayanthMulugu\OneDrive - Winfo Solutions\Documents\Databricks-Big-Book-Of-GenAI-FINAL.pdf",
    r"C:\Users\SaiJayanthMulugu\Downloads\azure-databricks.pdf"
]  # Start with two manageable documents

FAISS_INDEX_PATH = "output/faiss_index"  # folder to save/load index
DOCUMENT_CHUNKS_PATH = "output/document_chunks.txt"

# ── Global Models ───────────────────────────────────────────────
def get_llm():
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)

def get_embeddings():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Embedding device: {device}")
    
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'local_files_only': False, 'device': device},
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )
