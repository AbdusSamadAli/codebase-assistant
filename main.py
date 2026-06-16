from fastapi import FastAPI
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import chromadb
import os
load_dotenv()

app = FastAPI()

model=None
collection=None
groq_client=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global model,collection, groq_client

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
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    yield

app=FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question : str

@app.get("/")
def root():
      return {"status": "RAG API running"}

@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = model.encode([request.question]).tolist()
    results = collection.query(
        query_embeddings = query_embedding,
        n_results=3
    )
    retrived_chunks = results["documents"][0]
    prompt = f"""You are a helpful assistant.   
Answer the question using the context provided below. If the answer is not present in the context, say you don't know. Do not use outside knowledge.
Context: {chr(10).join(retrived_chunks)}
Question: {request.question}
Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content": prompt}]
    )
    answer = response.choices[0].message.content
    return {
        "question": request.question,
        "answer":answer,
        "retrieved_chunks": retrived_chunks
    }