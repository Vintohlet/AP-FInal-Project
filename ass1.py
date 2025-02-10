import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

# Конфигурация ChromaDB
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))
collection_name = "rag_collection_demo"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with SentenceTransformer embeddings"}
)

# Загрузка модели SentenceTransformer для эмбеддингов
try:
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Размерность: 384
except Exception as e:
    st.error(f"Error loading embedding model: {e}")
    embedding_model = None

# Функция для генерации эмбеддингов
def generate_embeddings(documents):
    if embedding_model:
        return embedding_model.encode(documents)
    else:
        raise RuntimeError("Embedding model not loaded!")

# Функция для добавления документов в ChromaDB
def add_documents_to_chromadb(documents, ids):
    if not documents or not ids:
        return
    embeddings = generate_embeddings(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )

# Функция для извлечения контекста из ChromaDB
def query_chromadb(query_text, n_results=1):
    if collection.count() == 0:
        return [], []
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results.get('documents', []), results.get('metadatas', [])

# Функция для удаления всех документов из ChromaDB
def clear_chromadb_collection():
    ids = collection.get()['ids'] 
    collection.delete(ids)

# Функция для запроса к Ollama через LlamaIndex
def query_ollama_with_llamaindex(prompt, model="llama3.2"):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        messages = [ChatMessage(role="user", content=prompt)]
        response = ""
        for chunk in llm.stream_chat(messages):
            response += chunk.delta
        return response
    except Exception as e:
        st.error(f"Error querying Ollama: {str(e)}")
        return None

# Интеграция ChromaDB и Ollama (RAG pipeline)
def rag_pipeline(query_text, model="llama3.2"):
    # Извлечение контекста из ChromaDB
    retrieved_docs, _ = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Создание запроса для Ollama с использованием контекста
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    print("######## Augmented Prompt ########")
    print(augmented_prompt)

    # Запрос к Ollama через LlamaIndex
    response = query_ollama_with_llamaindex(augmented_prompt, model)
    return response

# Streamlit веб-интерфейс
def main():
    st.title("Interactive Web App with Ollama and ChromaDB")

    # Секция добавления документа
    st.subheader("Add Document to ChromaDB")
    new_document = st.text_area("Enter document text", "")
    if st.button("Add Document"):
        if new_document:
            doc_id = f"doc{collection.count() + 1}"
            try:
                add_documents_to_chromadb([new_document], [doc_id])
                st.success(f"Document added with ID: {doc_id}")
            except Exception as e:
                st.error(f"Error adding document: {e}")
        else:
            st.error("Please enter a document to add.")

    # Секция для очистки коллекции
    st.subheader("Clear ChromaDB Contexts")
    if st.button("Clear All Documents"):
        try:
            clear_chromadb_collection()
            st.success("All documents have been removed from ChromaDB.")
        except Exception as e:
            st.error(f"Error clearing ChromaDB: {e}")

    # Секция для запроса к Ollama
    st.subheader("Ask a Question to Ollama")
    question = st.text_input("Enter your question", "")
    selected_model = "llama3.2"
    if st.button("Ask Ollama"):
        if question:
            try:
                response = rag_pipeline(question, model=selected_model)
                st.write("Response from Ollama:", response)
            except Exception as e:
                st.error(f"Error querying Ollama: {e}")
        else:
            st.error("Please enter a question to ask Ollama.")

    # Секция отображения документов в ChromaDB
    st.subheader("Documents in ChromaDB")
    if collection.count() > 0:
        documents, _ = query_chromadb("", collection.count())
        st.write("Documents in ChromaDB:")
        for doc in documents:
            st.write(doc)
    else:
        st.write("No documents in ChromaDB.")

if __name__ == "__main__":
    main()
