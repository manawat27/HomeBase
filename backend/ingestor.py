# This is my RAG library
# Run once per document
# 1. Load the document from /data/uploads
# 2. Process the document (e.g., extract text, split into overlapping chunks)
# 3. Pass each chunk to the embedder (nomic-embed-text) to get the vector representation
# 4. Store the vector in the database (e.g., ChromaDB) along with metadata (e.g., document ID, chunk ID)
# This makes the document available to query

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def loadDocument(file_path: str):
    """Load a file and return LangChain Document objects"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext in ['.txt', '.md']:
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader.load()

def chunkDocuments(documents):
    """Split documents into chunks with overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def deleteExistingChunks(source, collection_name: str = "home_docs"):
    """Delete all existing chunks for a given source file"""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    existing = vector_store.get(where={"source": source})
    if existing and existing["ids"]:
        print(f"Removing {len(existing['ids'])} old chunks for {source}")
        vector_store._collection.delete(ids=existing["ids"])

def embedAndStore(chunks, document_id, collection_name: str = "home_docs"):
    """Embed chunks and store in ChromaDB"""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Remove old chunks for this document before adding new ones
    source = chunks[0].metadata.get("source") if chunks else None
    if source:
        deleteExistingChunks(source, collection_name)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PATH
    )

    return vector_store

def ingestFile(file_path: str):
    """Full pipeline to load, chunk, embed, and store a document"""
    file_path = os.path.abspath(file_path)

    print(f"Loading document...{file_path}")
    documents = loadDocument(file_path)

    print("Chunking document...")
    chunks = chunkDocuments(documents)
    document_id = os.path.basename(file_path)

    print("Embedding and storing document...")
    embedAndStore(chunks, document_id)

    print("Document ingested successfully.")
    return len(chunks)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestor.py <file_path>")
        sys.exit(1)

    ingestFile(sys.argv[1])