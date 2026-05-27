import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Configuration ───────────────────────────────────────────────
ENABLE_CONTENT_EXTRACTION = True  # Set to False to skip content extraction (saves API calls)

# ── Paths ───────────────────────────────────────────────────────
# Load PDFs from relative 'pdfs/' folder (works on local & Render)
PDFS_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
PDF_PATHS = [
    os.path.join(PDFS_DIR, f)
    for f in os.listdir(PDFS_DIR)
    if f.lower().endswith(".pdf")
] if os.path.exists(PDFS_DIR) else []

if not PDF_PATHS:
    print(f"⚠️  Warning: No PDFs found in {PDFS_DIR}")

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
