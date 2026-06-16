from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Python is a programming language.",    
    "The cat is on the roof.",
    "I love machine learning.",
    "The sky is blue."
]

embeddings = model.encode(sentences)
dimension = embeddings.shape[1]
index = IndexFlatL2(dimension)
index.add(np.array(embeddings))

query = "What color is the sky?"
query_embedding = model.encode([query])

distances, indices = index.search(np.array(query_embedding),k=2)

retrieved_chunks = [sentences[i] for i in indices[0]]
print("retrieved chunks:", retrieved_chunks)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt= f"""You are a helpful assistant. 
Answer the questions using the context provifed below.
Context: {chr(10).join(retrieved_chunks)}

Question : {query}
Answer:"""
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content":prompt}]
)
print("generated answer:",response.choices[0].message.content)