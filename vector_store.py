import fitz
from tqdm import tqdm

def process_pdf_into_vector_store(pdf_path: str, embedding_model, text_splitter, chroma_client, collection_name: str = "pdf_1000_pages"):
    """Processes a PDF file, chunks its content, embeds the chunks, and stores them in ChromaDB."""
    print(f"Processing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Total pages: {total_pages}")

    collection = chroma_client.get_or_create_collection(name=collection_name)

    chunks_buffer = []
    metadatas_buffer = []
    ids_buffer = []
    chunk_counter = 0
    BATCH_SIZE = 100

    for page_num in tqdm(range(total_pages)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text = text.replace('\n', ' ')
        if not text.strip():
            continue

        page_chunks = text_splitter.split_text(text)

        for i, chunk in enumerate(page_chunks):
            chunks_buffer.append(chunk)
            metadatas_buffer.append({"page": page_num + 1})
            ids_buffer.append(f"doc_p{page_num + 1}_c{i}_{chunk_counter}")
            chunk_counter += 1

        if len(chunks_buffer) >= BATCH_SIZE:
            embeddings = embedding_model.encode(chunks_buffer).tolist()
            collection.add(
                documents=chunks_buffer,
                embeddings=embeddings,
                metadatas=metadatas_buffer,
                ids=ids_buffer
            )
            chunks_buffer, metadatas_buffer, ids_buffer = [], [], []
            print(f"🔄 Processed up to page {page_num + 1} of {total_pages}...")

    if chunks_buffer:
        embeddings = embedding_model.encode(chunks_buffer).tolist()
        collection.add(
            documents=chunks_buffer,
            embeddings=embeddings,
            metadatas=metadatas_buffer,
            ids=ids_buffer
        )

    print(f"✅ PDF processing complete! Total {chunk_counter} chunks stored in ChromaDB.")
    return collection
