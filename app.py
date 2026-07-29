import streamlit as st
import os
from download_and_setup import install_and_load_dependencies, download_pdf
from vector_store import process_pdf_into_vector_store
from retriever import retrieve_relevant_chunks
from llm_service import ask_rag_openrouter 

# --- Streamlit UI --- 
st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 سیستم پرسش و پاسخ RAG")
st.markdown("سوالات خود را بر اساس محتوای فایل PDF بپرسید.")

# Initialize session state variables if not already present
if 'embedding_model' not in st.session_state:
    st.session_state['embedding_model'] = None
if 'chroma_client' not in st.session_state:
    st.session_state['chroma_client'] = None
if 'text_splitter' not in st.session_state:
    st.session_state['text_splitter'] = None
if 'collection' not in st.session_state:
    st.session_state['collection'] = None
if 'pdf_downloaded' not in st.session_state:
    st.session_state['pdf_downloaded'] = False

OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY"
)
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API Key not found. Please set the OPENROUTER_API_KEY environment variable or add it to `secrets.toml`.")
    st.stop()

# Setup section
with st.sidebar:
    st.header("تنظیمات RAG")
    github_raw_file_url = st.text_input(
        "آدرس URL فایل PDF (لینک باید با فرمت (.pdf) باشه)",
        value='https://github.com/probml/pml-book/releases/download/2025-04-18/book1.pdf'
    )

    if st.button("راه‌اندازی سیستم RAG"):
        with st.spinner("در حال راه‌اندازی سیستم RAG..."):
            embedding_model, chroma_client, text_splitter = install_and_load_dependencies()
            st.session_state['embedding_model'] = embedding_model
            st.session_state['chroma_client'] = chroma_client
            st.session_state['text_splitter'] = text_splitter

            pdf_file_name = download_pdf(github_raw_file_url)
            st.session_state['pdf_path'] = pdf_file_name
            st.session_state['pdf_downloaded'] = True
            st.success("فایل PDF دانلود شد.")

            collection = process_pdf_into_vector_store(
                pdf_file_name,
                st.session_state['embedding_model'],
                st.session_state['text_splitter'],
                st.session_state['chroma_client'],
                collection_name='book'
            )
            st.session_state['collection'] = collection
            st.success("سیستم RAG با موفقیت راه‌اندازی شد!")

# Main interaction section
if st.session_state['pdf_downloaded'] and st.session_state['collection']:
    st.subheader("پرسش از PDF")
    user_question = st.text_area("سوال خود را اینجا وارد کنید:", height=100)

    if st.button("پرسیدن سوال"):
        if user_question:
            #try:
                with st.spinner("در حال جستجو و تولید پاسخ..."):
                    retrieved_docs, metadatas = retrieve_relevant_chunks(
                        user_question,
                        st.session_state['embedding_model'],
                        st.session_state['collection']
                    )

                    answer, pages_used = ask_rag_openrouter(
                        user_question,
                        retrieved_docs,
                        metadatas,
                        OPENROUTER_API_KEY
                    )
                    
                    st.markdown("### پاسخ AI:")
                    st.markdown(answer)

                    pages_info = ", ".join([f"صفحه {p['page']}" for p in pages_used]) if pages_used else "ناشناس"
                    st.info(f"📖 پاسخ استخراج‌شده از صفحات: {pages_info}")

            #except Exception as e:
            #    st.error(f"خطایی رخ داد: {e}\nلطفاً از تکمیل بودن تنظیمات RAG اطمینان حاصل کنید.")
        else:
            st.warning("لطفاً یک سوال وارد کنید.")
else:
    st.info("لطفاً سیستم RAG را از نوار کناری راه‌اندازی کنید.")
