from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
import numpy as np
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()

print("loading dataset...")
dataset = load_dataset("fancyzhx/ag_news",split="train[:50]")

raw_texts = [article["text"] for article in dataset]
print(f"Loaded {len(raw_texts)} articles")
print(f"sample article (first 200 chars):\n{raw_texts[0][:200]}\n")

print("chunking...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks=[]
for text in raw_texts:
    split = splitter.split_text(text)
    chunks.extend(split)

print(f"total chunks created: {len(chunks)}")
print(f"sample chunk: \n{chunks[0]}\n")

print("embedding chunks...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"embeddings shape: {embeddings.shape}\n")

dimension = embeddings.shape[1]
index = IndexFlatL2(dimension)
index.add(np.array(embeddings))
print(f"FAISS index size: {index.ntotal} vectors\n")

query = "Which country is facing major oil issues currently?"
query_embedding = model.encode([query])
distances, indices = index.search(np.array(query_embedding),k=3)

retrieved_chunks = [chunks[i] for i in indices[0]]
print("retrieved chunks:")
for i,chunk in enumerate(retrieved_chunks):
    print(f"[{i+1}] {chunk[:150]}...\n")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
prompt=f"""You are a helpful assistant.
Answer the question using the context provided below.
Context: {chr(10).join(retrieved_chunks)}
Question: {query}
Answer:"""
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user","content": prompt}]
)

print("generated answer:", response.choices[0].message.content)
