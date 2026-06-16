from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
import chromadb
import os
load_dotenv()

print("loading dataset...")
dataset = load_dataset("rajpurkar/squad", split="train[:100]")

raw_texts = list(set([row["context"] for row in dataset]))
print(f"unique paras loaded: {len(raw_texts)}")
print(f"sample para (first 200 chars):\n{raw_texts[0][:300]}\n")

print("chunking...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks=[]
for text in raw_texts:
    splits = splitter.split_text(text)
    chunks.extend(splits)
    
print(f"total chunks: {len(chunks)}\n")

print("embedding")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks,show_progress_bar=True)
print(f"embeddings shape: {embeddings.shape}\n")

print("storing in chromaDB")
client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("rag_docs")
except:
    pass

collection = client.create_collection("rag_docs")
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    embeddings = embeddings.tolist(),
    documents = chunks
)

print(f"chromadb collection size: {collection.count()} vectors \n")
query = "Where do college students stay when studying away from home?"
query_embedding = model.encode([query]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

retrived_chunks = results["documents"][0]
print("retrieved chunks:")
for i, chunk in enumerate(retrived_chunks):
    print(f"[{i+1}] {chunk}\n")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
prompt = f"""You are a helpful assistant.   
Answer the question using the context provided below. If the answer is not present in the context, say you don't know. Do not use outside knowledge. 
Context : {chr(10).join(retrived_chunks)}
Question : {query}
Answer:"""

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content":prompt}]
)

print("generated answer:", response.choices[0].message.content)