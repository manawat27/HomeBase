# Converts the retrieved chunks into a final answer
# This will have a prompt given to the AI
# Sends prompt to the llama3.2 model and gets the response

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from config import LLM_MODEL, OLLAMA_BASE_URL

PROMPT_TEMPLATE = """
You are a helpful home assitant. Answer the questions using ONLY the context provided below.
Be consice and direct. 

If the answer is not contained in the context, say:
"I could not find that in your documents."

Context:
{context}

Question:
{question}

Answer:"""

def buildPrompt(question: str, chunks: list):
    """Combine retrieved chunks into a single context block."""
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.format(context=context, question=question)

def generate(question: str, chunks: list) -> dict:
    """Send the prompt to prompt to Ollama and return the answer and the source"""
    if not chunks:
        return {
            "answer": "I could not find that in your documents.",
            "source": []
        }
    
    llm = OllamaLLM(
        model=LLM_MODEL, 
        base_url=OLLAMA_BASE_URL,
        temperature=0.1 # low temperature for more factual answers (high for creative)
    )

    prompt = buildPrompt(question, chunks)
    answer = llm.invoke(prompt)

    sources = list({c["source"] for c in chunks})
    return {
        "answer": answer.strip(),
        "source": sources
    }