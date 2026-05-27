import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = r"c:\Users\SaiJayanthMulugu\Downloads\database-data-warehousing-guide.pdf"
FAISS_INDEX_PATH = "faiss_index"

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_vectorstore():
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = load_vectorstore()
else:
    vectorstore = build_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

hyde_prompt = PromptTemplate.from_template("Question: {question}\nHypothetical Answer:")
final_prompt = PromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:")

hyde_chain = hyde_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def hyde_rag_chain(question):
    hypothetical = hyde_chain.invoke({"question": question})
    docs = retriever.invoke(hypothetical)
    context = format_docs(docs)
    final_chain = final_prompt | llm | StrOutputParser()
    answer = final_chain.invoke({"context": context, "question": question})
    return answer, docs, hypothetical

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer, docs, hypothesis = hyde_rag_chain(query.question)
    return {
        "answer": answer,
        "hypothesis": hypothesis,
        "sources": [
            {"page": d.metadata.get("page", "N/A"), "content": d.page_content[:200]}
            for d in docs
        ]
    }
