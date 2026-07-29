from openai import OpenAI
from typing import Tuple, List, Dict, Any
import re

def ask_rag_openrouter(question: str, context: List[str], pages: List[Dict[str, Any]], openrouter_api_key: str, model_name: str = "openrouter/free") -> Tuple[str, List[Dict[str, Any]]]:
    """Asks the LLM a question with provided context and returns the answer and pages used."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    messages = [
        {
            "role": "system",
            "content": """You are an expert machine learning professor. Answer the question accurately based ONLY on the provided context. If the answer is not in the context, state that you don't know.
    
    CRITICAL FORMATTING RULES:
    1. Use single dollar signs $...$ for all inline math equations.
    2. Use double dollar signs $$...$$ ONLY for standalone display blocks outside tables.
    3. NEVER use \\(...\\), \\[...\\] syntax, or the \\displaystyle command.
    4. INSIDE MARKDOWN TABLES: Keep all text and equations strictly on a single line. Do NOT use multi-line elements, raw newlines, or block equations ($$...$$) inside any table cell. Use only simple inline math $...$ inside tables.""",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    print(f"🔍 Searching in pages: {pages}...")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content, pages
    

