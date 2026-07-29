
import sys
import os
import subprocess
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def install_and_load_dependencies():
    """Installs necessary packages and initializes global components."""
    print("Installing dependencies...")
    # Using subprocess to run shell commands to manage pip
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "fitz", "-y"], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "PyMuPDF", "-y"], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "chromadb", "sentence_transformers", "langchain_text_splitters", "openai", "PyMuPDF", "streamlit"], capture_output=True, text=True)
    print("Dependencies installed.")

    print("Initializing embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")

    print("Initializing ChromaDB client...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db_large_book")
    print("ChromaDB client initialized.")

    print("Initializing text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    print("Text splitter initialized.")
    return embedding_model, chroma_client, text_splitter

def download_pdf(url):
  file_name = url.split("/")[-1]
  if not os.path.exists(file_name):
    print(f"Downloading {file_name}...")
    response = requests.get(url)
    response.raise_for_status()  # بررسی صحت دانلود و عدم وجود ارور ۴۰۴ یا ۵۰۰

    with open(file_name, "wb") as f:
      f.write(response.content)

    print(f"File downloaded to: {file_name}")
  else:
    print(f"{file_name} already exists.")

  return file_name

if __name__ == '__main__':
    github_raw_file_url = 'https://github.com/probml/pml-book/releases/download/2025-04-18/book1.pdf'
    install_and_load_dependencies()
    download_pdf(github_raw_file_url)
