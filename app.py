import os
import streamlit as st
from download_and_setup import download_pdf, install_and_load_dependencies
from llm_service import ask_rag_openrouter
from retriever import retrieve_relevant_chunks
from vector_store import process_pdf_into_vector_store

# --- Streamlit UI ---
st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 سیستم پرسش و پاسخ RAG")
st.markdown("سوالات خود را بر اساس محتوای فایل PDF بپرسید.")

# Initialize session state variables
if "embedding_model" not in st.session_state:
  st.session_state["embedding_model"] = None
if "chroma_client" not in st.session_state:
  st.session_state["chroma_client"] = None
if "text_splitter" not in st.session_state:
  st.session_state["text_splitter"] = None
if "collection" not in st.session_state:
  st.session_state["collection"] = None
if "pdf_downloaded" not in st.session_state:
  st.session_state["pdf_downloaded"] = False

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY"
)
if not OPENROUTER_API_KEY:
  st.error(
      "OpenRouter API Key not found. Please set the OPENROUTER_API_KEY"
      " environment variable or add it to `secrets.toml`."
  )
  st.stop()

# Setup section
with st.sidebar:
  st.header("تنظیمات RAG")

  # انتخاب روش ورود فایل PDF
  source_type = st.radio(
      "منبع فایل PDF را انتخاب کنید:", ("آپلود فایل PDF", "لینک URL")
  )

  pdf_file_name = None

  if source_type == "آپلود فایل PDF":
    uploaded_file = st.file_uploader("فایل PDF خود را انتخاب کنید", type=["pdf"])
    if uploaded_file is not None:
      # ذخیره فایل آپلود شده در دایرکتوری جاری
      pdf_file_name = uploaded_file.name
      with open(pdf_file_name, "wb") as f:
        f.write(uploaded_file.getbuffer())
  else:
    github_raw_file_url = st.text_input(
        "آدرس URL فایل PDF (لینک مستقیم .pdf):",
        value=(
            "https://github.com/probml/pml-book/releases/download/2025-04-18/book1.pdf"
        ),
    )

  if st.button("راه‌اندازی سیستم RAG"):
    if source_type == "آپلود فایل PDF" and not uploaded_file:
      st.error("لطفاً ابتدا یک فایل PDF آپلود کنید.")
    else:
      with st.spinner("در حال راه‌اندازی سیستم RAG..."):
        # load dependencies
        embedding_model, chroma_client, text_splitter = (
            install_and_load_dependencies()
        )
        st.session_state["embedding_model"] = embedding_model
        st.session_state["chroma_client"] = chroma_client
        st.session_state["text_splitter"] = text_splitter

        # دریافت فایل (از لینک یا آپلود)
        if source_type == "لینک URL":
          pdf_file_name = download_pdf(github_raw_file_url)

        st.session_state["pdf_path"] = pdf_file_name
        st.session_state["pdf_downloaded"] = True
        st.success("فایل PDF آماده‌سازی شد.")

        # پردازش و ساخت Vector Store
        collection = process_pdf_into_vector_store(
            pdf_file_name,
            st.session_state["embedding_model"],
            st.session_state["text_splitter"],
            st.session_state["chroma_client"],
            collection_name="book",
        )
        st.session_state["collection"] = collection
        st.success("سیستم RAG با موفقیت راه‌اندازی شد!")

# Main interaction section
if st.session_state["pdf_downloaded"] and st.session_state["collection"]:
  st.subheader("پرسش از PDF")
  user_question = st.text_area("سوال خود را اینجا وارد کنید:", height=100)

  if st.button("پرسیدن سوال"):
    if user_question:
      with st.spinner("در حال جستجو و تولید پاسخ..."):
        retrieved_docs, metadatas = retrieve_relevant_chunks(
            user_question,
            st.session_state["embedding_model"],
            st.session_state["collection"],
        )

        answer, pages_used = ask_rag_openrouter(
            user_question,
            retrieved_docs,
            metadatas,
            OPENROUTER_API_KEY,
        )

        st.markdown("### پاسخ AI:")
        st.markdown(answer)

        pages_info = (
            ", ".join([f"صفحه {p['page']}" for p in pages_used])
            if pages_used
            else "ناشناس"
        )
        st.info(f"📖 پاسخ استخراج‌شده از صفحات: {pages_info}")
    else:
      st.warning("لطفاً یک سوال وارد کنید.")
else:
  st.info("لطفاً سیستم RAG را از نوار کناری راه‌اندازی کنید.")
