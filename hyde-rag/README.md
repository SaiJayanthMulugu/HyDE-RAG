# HyDE RAG

A structured and modularized implementation of a Retrieval-Augmented Generation (RAG) system utilizing Hypothetical Document Embeddings (HyDE).

## Structure
- `config.py`: Contains global constants and setups.
- `hyde_rag/`: Core modularized logic (chain, vectorstore, extraction, prompts).
- `data/pdfs/`: Place your source PDF files here.
- `output/`: Where FAISS index and generated document chunks text are saved.
- `main.py`: Entry point for querying the system.

## Usage
1. Provide your original unstructured PDFs into `data/pdfs/` or define paths in `config.py`.
2. Add a `.env` file based on `.env.example`.
3. Run `python main.py` to index the data and execute queries!

