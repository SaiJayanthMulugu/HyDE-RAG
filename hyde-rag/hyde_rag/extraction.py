import time
from langchain_core.output_parsers import StrOutputParser
from config import get_llm
from .prompts import extract_prompt

def extract_relevant_content(document, question):
    llm = get_llm()
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
