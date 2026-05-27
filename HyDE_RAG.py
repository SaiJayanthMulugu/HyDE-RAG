import os
import time  # Add time import for rate limiting
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'local_files_only': False}
)

# ── Configuration ───────────────────────────────────────────────
ENABLE_CONTENT_EXTRACTION = True  # Set to False to skip content extraction (saves API calls)

# ── Paths ───────────────────────────────────────────────────────
PDF_PATHS = [
    r"c:\Users\SaiJayanthMulugu\Downloads\database-data-warehousing-guide.pdf",
    r"c:\Users\SaiJayanthMulugu\Downloads\IJRTI2304061(AI resource paper).pdf",
    r"c:\Users\SaiJayanthMulugu\Downloads\azure-databricks.pdf"
]  # Start with two manageable documents
FAISS_INDEX_PATH = "faiss_index"  # folder to save/load index

# ── Load or Build Vector Store ──────────────────────────────────
def build_vectorstore():
    """Load multiple PDFs, chunk semantically, embed and save to disk."""
    print("📄 Loading PDFs...")
    all_documents = []

    for pdf_path in PDF_PATHS:
        print(f"Loading: {os.path.basename(pdf_path)}")
        if not os.path.exists(pdf_path):
            print(f"⚠️  Warning: {pdf_path} not found, skipping...")
            continue

        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
            all_documents.extend(documents)
        except Exception as e:
            print(f"❌ Error loading {os.path.basename(pdf_path)}: {e}")
            print("Skipping this document...")
            continue

    if not all_documents:
        raise ValueError("No documents were loaded. Please check your PDF paths.")

    print(f"📊 Total pages loaded: {len(all_documents)}")

    # ── Chunking (using RecursiveCharacterTextSplitter for reliability) ───────────────────────
    print("✂️  Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks: 500 characters per chunk
        chunk_overlap=100,  # 100 character overlap
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.split_documents(all_documents)
    print(f"✅ Created {len(docs)} chunks")

    # ── Save Chunks to Text File ────────────────────────────────
    print("💾 Saving chunks to text file...")
    with open("document_chunks.txt", "w", encoding="utf-8") as f:
        f.write("Document Chunks from multiple PDFs:\n")
        for pdf_path in PDF_PATHS:
            f.write(f"- {os.path.basename(pdf_path)}\n")
        f.write("=" * 80 + "\n\n")

        for i, chunk in enumerate(docs, 1):
            f.write(f"Chunk {i}:\n")
            f.write("-" * 40 + "\n")
            f.write(chunk.page_content.strip() + "\n\n")
            f.write("-" * 80 + "\n\n")

    print(f"✅ Chunks saved to 'document_chunks.txt' ({len(docs)} chunks total)")

    # ── Embed and Save ──────────────────────────────────────────
    print("🔢 Embedding chunks and building FAISS index (only done once)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to '{FAISS_INDEX_PATH}'")
    return vectorstore

def load_vectorstore():
    """Load existing FAISS index from disk."""
    print("⚡ Loading FAISS index from disk (fast)...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded!")
    return vectorstore

# ── Smart Load: Build once, reuse forever ───────────────────────
if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = load_vectorstore()
else:
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── HyDE Prompt ─────────────────────────────────────────────────
hyde_prompt = PromptTemplate.from_template("""You are a document retrieval specialist. Your role is to generate hypothetical document passages used in a HyDE (Hypothetical Document Embeddings) retrieval pipeline.

## What is HyDE?
HyDE improves retrieval accuracy by embedding a hypothetical answer passage instead of the raw query. You generate the passage; a separate system embeds it and retrieves the nearest real documents.

## Your task
Given the question below, do the following:

STEP 1 — Classify the question.
Identify: (a) domain (e.g., biomedical, legal, software engineering, general knowledge), (b) answer type (factual, procedural, comparative, conceptual), (c) specificity level (narrow/specific or broad/general).

STEP 2 — Generate the hypothetical passage.
Write a 100–150 word passage as it would appear in an authoritative real-world document. The passage must:
- Use the precise vocabulary of the identified domain.
- State facts, figures, mechanisms, or procedures without hedging.
- Be written in third person or impersonal form (no "you" or "I").
- Not reference the question or the user.
- Contain no markdown formatting — plain prose only.

STEP 3 — Output format.
Return a JSON object with exactly these fields:

{
  "domain": "",
  "answer_type": "",
  "passage": ""
}

No text outside the JSON object.

Question: {question}""")

# ── Extract Relevant Content Prompt ─────────────────────────────
extract_prompt = PromptTemplate.from_template("""You are a context extraction engine in a multi-stage RAG pipeline. Your output is used as grounding context for a downstream answer-generation step. Precision and recall both matter — missed relevant passages degrade answer quality; irrelevant passages cause hallucination.

## Task
Given a document and a question, extract all passages from the document that are relevant to answering the question. Assign each passage a relevance score and classify the coverage level.

## Relevance scoring
Score each extracted passage on a 1–3 scale:
- 3 — Directly answers the question. Contains the specific fact, definition, step, or claim needed.
- 2 — Partially relevant. Provides context, background, or related information that supports an answer.
- 1 — Weakly relevant. Tangentially related; include only if no score-2 or score-3 passages exist.

## Extraction rules
1. Extract verbatim or near-verbatim — do not paraphrase unless a passage is unreadably long (>150 words); in that case, compress to the key claim while preserving original wording as much as possible.
2. Do not inject outside knowledge. Every extracted passage must be traceable to the document.
3. If the document answers the question but only implicitly (inference required), include the passage and flag it with "inferred": true.
4. If no relevant content exists, set "coverage" to "none" and return an empty passages array.

## Output format (strict JSON, no text outside it)
{
  "coverage": "full" | "partial" | "none",
  "passages": [
    {
      "excerpt": "",
      "score": 1 | 2 | 3,
      "inferred": false
    }
  ]
}

Coverage definitions:
- "full"    — At least one score-3 passage exists; the question can be directly answered.
- "partial" — Only score-1 or score-2 passages exist; the question can be partially addressed.
- "none"    — No relevant passages found.

Document: {document}

Question: {question}""")

def extract_relevant_content(document, question):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(1 + attempt * 0.5)  # Progressive delay: 1s, 1.5s, 2s
            extract_chain = extract_prompt | llm | StrOutputParser()
            return extract_chain.invoke({"document": document, "question": question})
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    print(f"⚠️  Rate limit hit, retrying in {2 + attempt * 1}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 + attempt * 1)  # Longer delay for rate limits: 2s, 3s, 4s
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts, using original content")
                    return document[:300] + "..." if len(document) > 300 else document
            else:
                print(f"❌ API error: {e}")
                return document[:300] + "..." if len(document) > 300 else document

# ── Final Answer Prompt ──────────────────────────────────────────
final_prompt = PromptTemplate.from_template("""You are an answer-generation module in a production RAG system. Your output is consumed programmatically. You must assess context quality, generate a grounded answer, and return structured metadata.

## Task
Given a set of context passages and a question, produce a grounded answer with a confidence grade and supporting evidence. Every claim must originate from the context.

## Answer quality rules
1. Use only information present in the context. No outside knowledge.
2. Be direct. Answer the question in the first sentence if possible.
3. If the context partially answers the question, answer what is supported and clearly flag gaps.
4. If the context is entirely irrelevant or empty, set answer to null and confidence to "none".
5. Never guess, speculate, or hallucinate. When uncertain, say so explicitly.

## Confidence grading
- "high"    — Context directly and completely answers the question. No gaps.
- "medium"  — Context partially answers the question or requires minor inference.
- "low"     — Context is only tangentially related; answer is indirect or incomplete.
- "none"    — Context contains no relevant information.

## Output format (strict JSON, no text outside)
{
  "answer": "",
  "confidence": "high" | "medium" | "low" | "none",
  "supporting_excerpts": [
    ""
  ],
  "gaps": ""
}

Context:
{context}

Question: {question}""")

# ── HyDE Chain ───────────────────────────────────────────────────
hyde_chain = hyde_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Full HyDE RAG Chain ──────────────────────────────────────────
def hyde_rag_chain(question):
    try:
        # 1. Generate hypothetical answer
        hypothetical_answer = hyde_chain.invoke({"question": question})
        print(f"\n-> Hypothetical Answer Generated:\n{hypothetical_answer}\n")
        time.sleep(1)  # Rate limiting between API calls

        # 2. Retrieve using hypothetical answer (HyDE key step)
        retrieved_docs = retriever.invoke(hypothetical_answer)
        context = format_docs(retrieved_docs)

        # 3. Generate final answer
        time.sleep(1)  # Rate limiting before final answer generation
        final_answer = final_prompt | llm | StrOutputParser()
        answer = final_answer.invoke({"context": context, "question": question})

        return answer, retrieved_docs, hypothetical_answer

    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print(f"❌ Rate limit error in main chain: {e}")
            print("💡 Consider upgrading to Groq Dev Tier or reducing API calls")
            return "Rate limit exceeded. Please try again later.", [], "Rate limit error"
        else:
            print(f"❌ Error in HyDE chain: {e}")
            return f"Error processing question: {e}", [], "Error occurred"


if __name__ == "__main__":
    questions = [
        "What is a Data Warehouse? Explain its key characteristics?",
        "What is the difference between OLTP and Data Warehousing systems?",
        "What are common tasks performed in a Data Warehouse?",
        "What is a Snowflake Schema and how does it differ from a Star Schema?",
        "Get the SQL query example for creating a fact table in a Data Warehouse and explain the components of the query."
    ]

    for question in questions:
        print("=" * 60)
        print(f"-> Question: {question}")
        answer, sources, hypothesis = hyde_rag_chain(question)
        print(f"-> Final Answer: {answer}")
        print("\n-> Retrieved Sources:")
        for i, source in enumerate(sources):
            if ENABLE_CONTENT_EXTRACTION:
                relevant_content = extract_relevant_content(source.page_content, question)
            else:
                # Use original content with length limit to avoid overwhelming output
                relevant_content = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
            print(f"  {i+1}. Page {source.metadata.get('page', 'N/A')}: {relevant_content}")
        print()
        time.sleep(2)  # Rate limiting: wait 2 seconds between questions
# ```

# ---

# **Flow visualization:**
# ```
# Question
#    │
#    ▼
# LLM generates Hypothetical Answer        ← HyDE Step
#    │
#    ▼
# Embed Hypothetical Answer
#    │
#    ▼
# Search Vector Store (FAISS)
#    │
#    ▼
# Retrieved Real Documents
#    │
#    ▼
# Extract Relevant Content per Document    ← NEW: Filter relevant info
#    │
#    ▼
# LLM generates Final Answer