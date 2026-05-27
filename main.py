"""
Explainable HyDE RAG — Streamlit App
======================================
Fix for WinError 1114 (c10.dll):
  The crash chain is:
    langchain_text_splitters → transformers → torch → c10.dll → OSError

  Root cause: langchain-text-splitters imports transformers at module level
  when the package is installed, even if you never use a tokenizer splitter.

  Solution applied here:
    Use SentenceTransformerEmbeddings (from langchain_community) which loads
    sentence-transformers directly — bypasses the broken torch/transformers path.

  If the error persists after switching embeddings, run:
    pip uninstall torch torchvision torchaudio transformers -y
    pip install torch --index-url https://download.pytorch.org/whl/cpu
  Then re-run the app.

Required packages:
    pip install streamlit python-dotenv faiss-cpu rank-bm25 pymupdf scikit-learn
                langchain langchain-community langchain-groq langchain-text-splitters
                sentence-transformers
"""

import os
import torch  # Fix for WinError 1114 (c10.dll): Torch must be imported before Streamlit to avoid DLL initialization failure
import pickle
import warnings
import logging

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings  # avoids transformers import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity

# ── Silence noisy loggers ──────────────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()

st.set_page_config(page_title="Explainable RAG", layout="wide")

PDF_CONFIG = {
    "data_warehouse": r"c:\Users\SaiJayanthMulugu\Downloads\database-data-warehousing-guide.pdf",
    "ai_paper":       r"c:\Users\SaiJayanthMulugu\Downloads\IJRTI2304061(AI resource paper).pdf",
    "research":       r"c:\Users\SaiJayanthMulugu\Downloads\Research Methodology .pdf",
}

FAISS_INDEX_PATH = "faiss_index"
BM25_INDEX_PATH  = "bm25_index.pkl"

# llama-3.1-70b-versatile was deprecated by Groq in Oct 2024
AVAILABLE_MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


# ══════════════════════════════════════════════════════════════════════════════
# Custom Ensemble Retriever — Weighted RRF
# ══════════════════════════════════════════════════════════════════════════════
class CustomEnsembleRetriever:
    """Weighted Reciprocal Rank Fusion over BM25 + FAISS retrievers."""

    def __init__(self, retrievers, weights, k_value: int = 8, rank_constant: int = 60):
        self.retrievers    = retrievers
        self.weights       = weights
        self.k_value       = k_value
        self.rank_constant = rank_constant

    def invoke(self, query: str) -> list:
        doc_lists  = [r.invoke(query) for r in self.retrievers]
        rrf_scores = {}

        for weight, docs in zip(self.weights, doc_lists):
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in rrf_scores:
                    rrf_scores[key] = {"doc": doc, "score": 0.0}
                rrf_scores[key]["score"] += weight / (rank + self.rank_constant)

        ranked = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked][: self.k_value]


# ══════════════════════════════════════════════════════════════════════════════
# Embedding model (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_embeddings():
    """
    SentenceTransformerEmbeddings loads the model via the sentence-transformers
    package directly, NOT through HuggingFace transformers.
    This avoids the transformers → torch → c10.dll crash chain.
    """
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# ══════════════════════════════════════════════════════════════════════════════
# Index building
# ══════════════════════════════════════════════════════════════════════════════
def build_index(embeddings):
    """
    Load PDFs → chunk → build and persist FAISS + BM25 indices.

    chunk_size=900 keeps chunks inside all-MiniLM-L6-v2's 256-token window.
    The original 4000-char chunks were silently truncated by the embedder.
    """
    all_docs = []
    for name, path in PDF_CONFIG.items():
        if not os.path.exists(path):
            st.warning(f"PDF not found, skipping **{name}**  \n`{path}`")
            continue
        try:
            loader = PyMuPDFLoader(path)
            docs   = loader.load()
            for d in docs:
                d.metadata["source"] = name
            all_docs.extend(docs)
            st.sidebar.write(f"✓ Loaded {len(docs)} pages — **{name}**")
        except Exception as exc:
            st.warning(f"Could not load **{name}**: {exc}")

    if not all_docs:
        st.error("No documents loaded. Check PDF paths in PDF_CONFIG.")
        return None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks   = splitter.split_documents(all_docs)
    st.sidebar.write(f"Total chunks built: **{len(chunks)}**")

    vectorstore    = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return vectorstore, bm25_retriever


# ══════════════════════════════════════════════════════════════════════════════
# System initialisation
# ══════════════════════════════════════════════════════════════════════════════
def initialize_system(model_name: str, k_value: int, force_rebuild: bool = False):
    embeddings   = get_embeddings()
    index_exists = os.path.exists(FAISS_INDEX_PATH) and os.path.exists(BM25_INDEX_PATH)

    if index_exists and not force_rebuild:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        vectorstore, bm25_retriever = build_index(embeddings)
        if vectorstore is None:
            return

    faiss_retriever    = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_value, "fetch_k": 20},
    )
    bm25_retriever.k   = k_value

    st.session_state.retriever   = CustomEnsembleRetriever(
        [bm25_retriever, faiss_retriever], [0.5, 0.5], k_value
    )
    st.session_state.vectorstore  = vectorstore
    st.session_state.embeddings   = embeddings
    st.session_state.llm          = ChatGroq(model_name=model_name, temperature=0.3)
    st.session_state.k_value      = k_value


# ══════════════════════════════════════════════════════════════════════════════
# Validation metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(answer: str, docs: list) -> tuple:
    """
    Returns (similarity, coverage, eval_label, explanation, grounded_bool).

    Fixes vs original:
    • Single batched embed_documents() call instead of N+2 separate calls.
    • Context = mean of chunk embeddings — avoids 256-token truncation.
    • Grounding uses .startswith("YES") — robust against "YES, because…".
    """
    emb      = st.session_state.embeddings
    texts    = [answer] + [d.page_content[:500] for d in docs]
    all_embs = emb.embed_documents(texts)

    ans_emb    = all_embs[0]
    chunk_embs = all_embs[1:]

    ctx_emb = (
        [sum(v) / len(v) for v in zip(*chunk_embs)]
        if chunk_embs else ans_emb
    )

    sim        = float(cosine_similarity([ans_emb], [ctx_emb])[0][0])
    cov_scores = cosine_similarity([ans_emb], chunk_embs)[0]
    cov        = float(max(cov_scores)) if len(cov_scores) > 0 else 0.0

    context_sample = "\n\n".join([d.page_content[:400] for d in docs[:5]])
    eval_prompt = (
        "Strict grounding check. Reply with YES or NO and nothing else.\n"
        "Does the answer below contain ONLY information present in the context?\n\n"
        f"Context:\n{context_sample}\n\nAnswer:\n{answer}"
    )
    eval_raw   = st.session_state.llm.invoke(eval_prompt).content.strip().upper()
    grounded   = eval_raw.startswith("YES")
    eval_label = "YES ✓" if grounded else "NO ✗"

    reasons = []
    if sim < 0.2:    reasons.append("Low similarity — retrieved chunks may be irrelevant")
    if cov < 0.2:    reasons.append("Low coverage — answer not found in retrieved documents")
    if not grounded: reasons.append("LLM grounding check failed — possible hallucination")

    explanation = (
        "Answer is fully grounded in the retrieved context."
        if not reasons
        else "Failure reasons:\n- " + "\n- ".join(reasons)
    )
    return sim, cov, eval_label, explanation, grounded


# ══════════════════════════════════════════════════════════════════════════════
# RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════
def rag_pipeline(question: str, strategy: str, status) -> tuple:
    hypothetical_doc = None
    k                = st.session_state.get("k_value", 8)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    if strategy == "HyDE (Hypothetical Doc Embeddings)":
        status.update(label="Generating hypothetical document (HyDE)…")
        hyde_prompt = (
            "You are a technical documentation expert specialising in databases, "
            "AI research, and research methodology. Write a concise, factual "
            "3-sentence passage that directly answers the following question. "
            "Use precise technical terminology as it would appear in an academic "
            "or reference document.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        hypothetical_doc = st.session_state.llm.invoke(hyde_prompt).content.strip()
        status.update(label="Searching hybrid index with hypothetical document…")
        docs = st.session_state.retriever.invoke(hypothetical_doc)

    elif strategy == "Multi-Query":
        status.update(label="Generating query variations…")
        expand_prompt = (
            "Rewrite the following question in a different way that preserves its "
            "meaning but uses different vocabulary. Return only the rewritten question.\n\n"
            f"Question: {question}"
        )
        expanded = st.session_state.llm.invoke(expand_prompt).content.strip()
        queries  = [question, expanded, f"Explain: {question}"]

        seen, docs = set(), []
        status.update(label="Searching hybrid index for query variations…")
        for q in queries:
            for d in st.session_state.retriever.invoke(q):
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    docs.append(d)

    else:
        status.update(label="Searching hybrid index (BM25 + FAISS)…")
        docs = st.session_state.retriever.invoke(question)

    # ── Dedup & context ────────────────────────────────────────────────────────
    status.update(label="Ranking and deduplicating results…")
    docs_dict = {d.page_content: d for d in docs}
    docs      = list(docs_dict.values())[:k]
    context   = "\n\n".join([d.page_content for d in docs])

    # ── Answer generation ──────────────────────────────────────────────────────
    status.update(label="Generating answer…")
    answer_prompt = (
        "You are a precise technical assistant. Answer the question using ONLY the "
        "information in the context below. If the context does not contain enough "
        "information, say: 'The provided context does not contain sufficient information "
        "to answer this question.' Do not speculate or add outside knowledge.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = st.session_state.llm.invoke(answer_prompt).content.strip()

    # ── Metrics ────────────────────────────────────────────────────────────────
    status.update(label="Calculating validation metrics…")
    sim, cov, eval_label, explanation, grounded = compute_metrics(answer, docs)

    status.update(label="Done!", state="complete", expanded=False)
    return answer, docs, sim, cov, eval_label, explanation, hypothetical_doc, grounded


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    defaults = {
        "vectorstore": None,
        "retriever":   None,
        "llm":         None,
        "embeddings":  None,
        "k_value":     8,
        "messages":    [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")

    model_name = st.sidebar.selectbox("LLM model", AVAILABLE_MODELS)
    k_value    = st.sidebar.slider("Top-K documents", min_value=3, max_value=10, value=8)
    strategy   = st.sidebar.selectbox(
        "Retrieval strategy",
        ["HyDE (Hypothetical Doc Embeddings)", "Multi-Query", "Standard"],
    )

    col1, col2 = st.sidebar.columns(2)
    init_btn    = col1.button("Initialize",    use_container_width=True)
    rebuild_btn = col2.button("Rebuild Index", use_container_width=True)

    if init_btn or rebuild_btn:
        with st.spinner("Setting up system…"):
            initialize_system(model_name, k_value, force_rebuild=rebuild_btn)
        if st.session_state.vectorstore is not None:
            st.sidebar.success("System ready!")

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    ready = st.session_state.vectorstore is not None
    st.sidebar.markdown(f"**Status:** {'🟢 Ready' if ready else '🔴 Not initialised'}")

    # ── Main area ──────────────────────────────────────────────────────────────
    st.title("🧠 Explainable RAG System")
    st.markdown("Ask questions grounded in your PDF documents.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.warning("Please click **Initialize** in the sidebar first.")
        return

    with st.chat_message("assistant"):
        with st.status("Running RAG pipeline…", expanded=True) as status:
            answer, docs, sim, cov, eval_label, explanation, hypothetical_doc, grounded = rag_pipeline(
                question, strategy, status
            )

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Metric cards
        if docs:
            m1, m2, m3 = st.columns(3)
            m1.metric("Similarity score", f"{sim:.3f}",
                      help="Cosine similarity: answer vs mean context embedding")
            m2.metric("Coverage score",   f"{cov:.3f}",
                      help="Best cosine match between answer and any single chunk")
            m3.metric("LLM grounding",    eval_label,
                      help="Did the LLM confirm the answer is grounded in context?")

        # HyDE expander
        if hypothetical_doc:
            with st.expander("💭 Hypothetical document used for retrieval (HyDE)"):
                st.info(
                    "The LLM generated this passage to represent an ideal answer. "
                    "Its embedding was used to query the vector store."
                )
                st.write(hypothetical_doc)

        # Grounding validation
        if not grounded or "Failure" in explanation:
            with st.expander("🚨 Grounding issues detected", expanded=True):
                st.error(explanation)
        else:
            with st.expander("✅ Grounding validation"):
                st.success(explanation)

        # Sources
        if docs:
            with st.expander(f"📚 Retrieved sources ({len(docs)} chunks)"):
                for idx, d in enumerate(docs):
                    source = d.metadata.get("source", "unknown")
                    page   = d.metadata.get("page", "?")
                    st.markdown(f"**[{idx + 1}] {source} — page {page}**")
                    st.info(d.page_content)


if __name__ == "__main__":
    main()