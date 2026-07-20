# Called on to get the answer from the model and db
# 1. Takes the users raw query
# 2. Uses the embedder to convert the query into a vector -- used nomic-embed-text
# 3. Uses the vector to query the database (e.g., ChromaDB) -- 3 most similar chunks to this vector
# 4. Returns chunks and the document it came from

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_PATH, TOP_K_RESULTS

def getVectorStore(collection_name: str = "home_docs"):
    """Initialize the vector store and return it"""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

def retrieve(query: str, collection_name: str = "home_docs"):
    """Given a user query, 
    embed it and return the top k results from the vector store"""
    vector_store = getVectorStore(collection_name)
    results = vector_store.similarity_search_with_score(query, k=TOP_K_RESULTS)

    chunks = []
    for doc, score in results:
        if score > 1.5:  # skip chunks that are too dissimilar (lower = more similar in Chroma)
            continue
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "metadata": doc.metadata,
            "score": round(score, 4)
        })
    return chunks