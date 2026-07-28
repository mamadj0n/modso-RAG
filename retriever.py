def retrieve_relevant_chunks(question: str, embedding_model, collection, top_k: int = 3):
    """Retrieves relevant document chunks from ChromaDB based on a question."""
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )
    retrieved_docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    return retrieved_docs, metadatas
