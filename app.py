import os
import tempfile
import uuid
import requests
import streamlit as st
from download_and_setup import download_pdf, install_and_load_dependencies
from llm_service import ask_rag_openrouter
from retriever import retrieve_relevant_chunks
from vector_store import process_pdf_into_vector_store

# --- Streamlit UI Config ---
st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 سیستم پرسش و پاسخ RAG")
st.markdown("سوالات خود را بر اساس محتوای فایل PDF بپرسید.")


# --- بهینه‌سازی بارگذاری مدل‌ها با Cache ---
@st.cache_resource(show_spinner="در حال بارگذاری مدل‌های امبدینگ و تنظیمات پایه...")
def get_rag_dependencies():
  return install_and_load_dependencies()


# مقداردهی متغیرهای سشن
if "collection" not in st.session_state:
  st.session_state["collection"] = None
if "processed_files" not in st.session_state:
  st.session_state["processed_files"] = []  # تاریخچه فایل‌ها

# بررسی وجود کلید API
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get(
    "OPENROUTER_API_KEY"
)
if not OPENROUTER_API_KEY:
  st.warning(
      "⚠️ کلید OpenRouter API یافت نشد. بخش تولید پاسخ با LLM غیرفعال خواهد"
      " بود."
  )

# --- Sidebar UI ---
with st.sidebar:
  st.header("تنظیمات RAG")

  source_type = st.radio(
      "منبع فایل PDF را انتخاب کنید:", ("آپلود فایل PDF", "لینک URL")
  )

  uploaded_file = None
  github_raw_file_url = ""

  if source_type == "آپلود فایل PDF":
    uploaded_file = st.file_uploader(
        "فایل PDF خود را انتخاب کنید", type=["pdf"]
    )
  else:
    github_raw_file_url = st.text_input(
        "آدرس URL فایل PDF (لینک مستقیم .pdf):",
        value=(
            "https://github.com/probml/pml-book/releases/download/2025-04-18/book1.pdf"
        ),
    )

  if st.button("راه‌اندازی سیستم RAG"):
    pdf_path = None
    is_temp_file = False

    try:
      with st.spinner("در حال پردازش فایل و ساخت دیتابیس برداری..."):
        embedding_model, chroma_client, text_splitter = get_rag_dependencies()

        if source_type == "آپلود فایل PDF":
          if not uploaded_file:
            st.error("لطفاً ابتدا یک فایل PDF آپلود کنید.")
            st.stop()

          temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
          temp_file.write(uploaded_file.getbuffer())
          temp_file.close()
          pdf_path = temp_file.name
          is_temp_file = True
          display_name = uploaded_file.name

        else:
          url = github_raw_file_url.strip()
          if not url.lower().endswith(".pdf"):
            try:
              head_resp = requests.head(url, timeout=5)
              content_type = head_resp.headers.get("Content-Type", "")
              if "application/pdf" not in content_type:
                st.error("لینک وارد شده به یک فایل PDF معتبر اشاره نمی‌کند.")
                st.stop()
            except Exception:
              st.error("آدرس URL وارد شده معتبر نیست یا در دسترس نمی‌باشد.")
              st.stop()

          pdf_path = download_pdf(url)
          display_name = url.split("/")[-1]

        unique_collection_name = f"col_{uuid.uuid4().hex}"

        collection = process_pdf_into_vector_store(
            pdf_path,
            embedding_model,
            text_splitter,
            chroma_client,
            collection_name=unique_collection_name,
        )

        st.session_state["collection"] = collection

        # ذخیره اطلاعات فایل در لیست برای نمایش در جدول
        st.session_state["processed_files"].append(
            {"نام فایل": display_name, "روش دریافت": source_type}
        )

        st.success("سیستم RAG با موفقیت راه‌اندازی شد!")

    except Exception as e:
      st.error(f"خطایی هنگام پردازش PDF یا ساخت Vector Store رخ داد: {e}")

    finally:
      if is_temp_file and pdf_path and os.path.exists(pdf_path):
        os.remove(pdf_path)

  # --- نمایش جدول تاریخچه فایل‌های آپلود شده ---
  st.markdown("---")
  st.subheader("📋 فایل‌های پردازش‌شده")

  if st.session_state["processed_files"]:
    st.dataframe(
        st.session_state["processed_files"],
        use_container_width=True,
        hide_index=True,
    )
  else:
    st.caption("هنوز فایلی پردازش نشده است.")

# --- Main Interaction Section ---
if st.session_state["collection"] is not None:
  st.subheader("پرسش از PDF")
  user_question = st.text_area("سوال خود را اینجا وارد کنید:", height=100)

  if st.button("پرسیدن سوال"):
    if not user_question.strip():
      st.warning("لطفاً یک سوال وارد کنید.")
    else:
      try:
        with st.spinner("در حال جستجو در مستندات..."):
          embedding_model, chroma_client, text_splitter = get_rag_dependencies()

          retrieved_docs, metadatas = retrieve_relevant_chunks(
              user_question,
              embedding_model,
              st.session_state["collection"],
          )

          if not retrieved_docs:
            st.warning("هیچ بخشی از فایل PDF مرتبط با سوال شما پیدا نشد.")
            st.stop()

          if not OPENROUTER_API_KEY:
            st.info("📌 بخش متون مرتبط از کتاب یافت شد:")
            for doc in retrieved_docs:
              st.write(f"- {doc}")
            st.stop()

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

      except Exception as e:
        st.error(f"خطا در فراخوانی مدل زبانی یا بازیابی اطلاعات: {e}")

else:
  st.info("لطفاً سیستم RAG را از نوار کناری راه‌اندازی کنید.")
