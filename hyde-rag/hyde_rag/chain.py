import time
from langchain_core.output_parsers import StrOutputParser
from config import get_llm
from .prompts import hyde_prompt, final_prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def hyde_rag_chain(question, retriever):
    llm = get_llm()
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    
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
        final_answer_chain = final_prompt | llm | StrOutputParser()
        answer = final_answer_chain.invoke({"context": context, "question": question})

        return answer, retrieved_docs, hypothetical_answer

    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            print(f"❌ Rate limit error in main chain: {e}")
            print("💡 Consider upgrading to Groq Dev Tier or reducing API calls")
            return "Rate limit exceeded. Please try again later.", [], "Rate limit error"
        else:
            print(f"❌ Error in HyDE chain: {e}")
            return f"Error processing question: {e}", [], "Error occurred"
