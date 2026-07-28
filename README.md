# Modso RAG (Retrieval-Augmented Generation System)

A modular Python-based Retrieval-Augmented Generation (RAG) system designed for document retrieval and context-aware LLM text generation.

## Project Structure

- **`app.py`**: Main application entry point (typically a Streamlit or Gradio UI).
- **`download_and_setup.py`**: Automated script for downloading necessary datasets, assets, or models.
- **`llm_service.py`**: Integration layer for communicating with Large Language Models.
- **`retriever.py`**: Logic for searching and fetching relevant context chunks from the vector store.
- **`vector_store.py`**: Manages vector embeddings, database creation, and similarity searches.
- **`requirements.txt`**: Project Python dependencies.

## Requirements

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running the Application

To start the app:
```bash
python app.py
```
