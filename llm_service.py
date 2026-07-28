from openai import OpenAI
from typing import Tuple, List, Dict, Any

def ask_rag_openrouter(question: str, context: List[str], pages: List[Dict[str, Any]], openrouter_api_key: str, model_name: str = "openrouter/free") -> Tuple[str, List[Dict[str, Any]]]:
    """Asks the LLM a question with provided context and returns the answer and pages used."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert ML professor. Answer the question accurately based ONLY on the provided context. If not found in context, say you don't know."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]

    print(f"🔍 Searching in pages: {pages}...")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content, pages
