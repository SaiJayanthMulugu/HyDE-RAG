import os
import sys

# Windows console emoji printing fix
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
from dotenv import load_dotenv
from hyde_rag.vectorstore import update_vectorstore, load_vectorstore
from hyde_rag.chain import hyde_rag_chain
from hyde_rag.extraction import extract_relevant_content
from config import ENABLE_CONTENT_EXTRACTION

def main():
    # Load environment variables
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Please set GROQ_API_KEY in your .env file")

    # ── Smart Load: Incremental rebuilding based on source hashes ────
    vectorstore = load_vectorstore()
    vectorstore = update_vectorstore(vectorstore)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n" + "="*60)
    print("🤖 HyDE RAG System Ready!")
    print("Type your questions below. Type 'quit' or 'exit' to stop.")
    print("="*60)

    while True:
        try:
            question = input("\n🤔 Your Question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
                
            if not question:
                continue

            print("\n" + "=" * 60)
            answer, sources, hypothesis = hyde_rag_chain(question, retriever)
            print(f"\n-> Final Answer: {answer}")
            print("\n-> Retrieved Sources:")
            for i, source in enumerate(sources):
                if ENABLE_CONTENT_EXTRACTION:
                    relevant_content = extract_relevant_content(source.page_content, question)
                else:
                    # Use original content with length limit to avoid overwhelming output
                    relevant_content = source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content
                print(f"  {i+1}. Page {source.metadata.get('page', 'N/A')}: {relevant_content}")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
