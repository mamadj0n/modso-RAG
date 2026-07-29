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
            "content": """
            You are an expert ML professor. Answer the question accurately based ONLY on the provided context. If not found in context, say you don't know.
            Format all mathematical expressions strictly using Markdown dollars: $math$ for inline equations and $$math$$ for display block equations. Do NOT use \( or \[ syntax. Inside markdown tables, keep all equations on a single line and avoid raw newlines within table cells.
            """
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
    

def fix_llm_markdown(text: str) -> str:
  # ۱. تبدیل فرمول‌های بلوکی \[ ... \] به $$ ... $$
  text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

  # ۲. تبدیل فرمول‌های درون‌متنی \( ... \) به $ ... $
  text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

  # ۳. حل مشکل شکستن خطوط لاتخ داخل جدول‌های Markdown (جلوگیری از به‌هم‌ریختگی ستون‌ها)
  lines = text.split("\n")
  in_table = False
  fixed_lines = []

  for line in lines:
    if "|" in line:
      in_table = True
      # حذف Newlineهای احتمالی و جایگزینی [ ] اضافه داخل جداول
      line = re.sub(r"\[\s*", "$$", line)
      line = re.sub(r"\s*\]", "$$", line)
      fixed_lines.append(line)
    else:
      in_table = False
      fixed_lines.append(line)

  text = "\n".join(fixed_lines)

  # ۴. جایگزینی کاراکترهای [ و ] تنهای باقی‌مانده که دور فرمول‌ها قرار گرفته‌اند
  text = re.sub(r"(?m)^\s*\[\s*", "$$\n", text)
  text = re.sub(r"(?m)^\s*\]\s*", "\n$$", text)

  return text
