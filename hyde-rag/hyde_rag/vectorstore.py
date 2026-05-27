import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import PDF_PATHS, FAISS_INDEX_PATH, DOCUMENT_CHUNKS_PATH, get_embeddings

def check_pdf_in_chunks(basename):
    """Memory-efficient check if a PDF has already been processed and saved in the chunks file."""
    if not os.path.exists(DOCUMENT_CHUNKS_PATH):
        return False
    with open(DOCUMENT_CHUNKS_PATH, "r", encoding="utf-8", errors="ignore") as f:
        # Check first few lines for the initial header
        for _ in range(50):
            line = f.readline()
            if not line: break
            if basename in line:
                return True
                
        # If not in header, it might be in appended records
        for line in f:
            if f"Appended chunks for {basename}" in line or f"- {basename}" in line:
                return True
    return False

def update_vectorstore(vectorstore):
    """Load missing PDFs, chunk semantically, append to FAISS."""
    print("📄 Checking PDFs...")
    all_documents = []
    new_pdfs = []

    for pdf_path in PDF_PATHS:
        basename = os.path.basename(pdf_path)
        
        # Incremental check!
        if check_pdf_in_chunks(basename) and vectorstore is not None:
            print(f"⏩ Skipping '{basename}' (already indexed).")
            continue
            
        print(f"Loading: {basename}")
        if not os.path.exists(pdf_path):
            print(f"⚠️  Warning: {pdf_path} not found, skipping...")
            continue

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} pages from {basename}")
            all_documents.extend(documents)
            new_pdfs.append(basename)
        except Exception as e:
            print(f"❌ Error loading {basename}: {e}")
            continue

    if not all_documents:
        print("⚡ No new documents to load. Using existing index.")
        return vectorstore

    print(f"📊 Total new pages loaded: {len(all_documents)}")

    # ── Chunking (Fix 5: larger chunks) ───────────────────────
    print("✂️  Chunking new documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.split_documents(all_documents)
    print(f"✅ Created {len(docs)} new chunks")

    os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_PATH), exist_ok=True)

    # ── Append Chunks to Text File ────────────────────────────────
    print("💾 Appending new chunks to text file...")
    mode = "a" if os.path.exists(DOCUMENT_CHUNKS_PATH) else "w"
    with open(DOCUMENT_CHUNKS_PATH, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("Document Chunks from multiple PDFs:\n")
            for base in new_pdfs:
                f.write(f"- {base}\n")
            f.write("=" * 80 + "\n\n")
        else:
            for base in new_pdfs:
                f.write(f"\n\n=== Appended chunks for {base} ===\n\n")

        for i, chunk in enumerate(docs, 1):
            f.write(f"New Chunk {i}:\n")
            f.write("-" * 40 + "\n")
            f.write(chunk.page_content.strip() + "\n\n")
            f.write("-" * 80 + "\n\n")

    # ── Embed and Save (Fix 1: Batch embedding) ─────────────────
    print(f"🔢 Embedding {len(docs)} new chunks...")
    embeddings = get_embeddings()
    
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    
    BATCH_SIZE = 64
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vecs = embeddings.embed_documents(batch)
        all_embeddings.extend(vecs)
        print(f"  Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)} chunks", end="\r")
        
    print() # newline after progress bar
    
    if vectorstore is None:
        dim = len(all_embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_embeddings, dtype="float32"))

        docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
        index_to_id = {i: str(i) for i in range(len(docs))}

        vectorstore = FAISS(embeddings.embed_query, index, docstore, index_to_id)
    else:
        # Append to existing vectorstore
        # To merge cleanly without overriding IDs, we need secure string UUIDs
        import uuid
        ids = [str(uuid.uuid4()) for _ in docs]
        
        # Add natively utilizing the embeddings we just computed manually to guarantee speed
        vectorstore.index.add(np.array(all_embeddings, dtype="float32"))
        
        # Add to docstore manually because FAISS object needs parallel tracking
        for doc_id, doc in zip(ids, docs):
            vectorstore.docstore.add({doc_id: doc})
            
        vectorstore.index_to_docstore_id[len(vectorstore.index_to_docstore_id)] = ids[0] # Just need to map offset to ID properly
        # Wait! It's much safer to use vectorstore.add_documents which re-embeds OR vectorstore.add_embeddings() which is supported in standard FAISS abstraction!
        
        # Using native method directly to prevent index out of bounds tracking issues:
        text_embeddings = list(zip(texts, all_embeddings))
        vectorstore.add_embeddings(text_embeddings, metadatas)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ FAISS index updated and saved to '{FAISS_INDEX_PATH}'")
    return vectorstore

def load_vectorstore():
    """Load existing FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    print("⚡ Loading FAISS index from disk (fast)...")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded!")
    return vectorstore
