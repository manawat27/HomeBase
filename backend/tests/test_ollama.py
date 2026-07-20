from langchain_ollama import OllamaLLM, OllamaEmbeddings

llm = OllamaLLM(model="llama3.2")
response = llm.invoke("What is RAG in the context of AI?")
print("LLM Response:", response)

embedder = OllamaEmbeddings(model="nomic-embed-text")
vector = embedder.embed_query("What is the boiling point of water?")
print("Embedding Vector:", vector)
print("Vector Length:", len(vector))