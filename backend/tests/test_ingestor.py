from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_PATH, TOP_K_RESULTS

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

vector_store = Chroma(
    collection_name="home_docs",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

results = vector_store.similarity_search("when does the fridge warranty expire?", k=TOP_K_RESULTS)

print(f"Found {len(results)} chunks:")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} --- source: {doc.metadata.get('source', 'unknown')}")
    print(doc.page_content)
    print()