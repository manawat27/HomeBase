import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from retriever import retrieve
from generator import generate

questions = [
    "What is the wifi network and password?",
    "When does the fridge warranty expire?",
    "What is the boiling point of water?", # should say could not find
    "Who is the emergency plumber?",
    "What is the meaning of life?", # should say could not find
    "Who are the tenants in the lease aggreement?" ,
    "When is the move in date for unit B204?"
]

for question in questions:
    print(f"Question: {question}")
    chunks = retrieve(question)
    result = generate(question, chunks)
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['source']}")
    print("-" * 50)