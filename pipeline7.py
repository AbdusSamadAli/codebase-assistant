from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from datasets import load_dataset
from dotenv import load_dotenv

import os

load_dotenv()

print("loading dataset...")
dataset = load_dataset("rajpurkar/squad", split="train[:100]")
raw_texts = list(set([row["context"] for row in dataset]))

documents = [Document(page_content=text) for text in raw_texts]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=os.getenv("CHROMA_DB_DAY6", "./chroma_db_day6")
)

retriever = vector_store.as_retriever(search_kwargs={"k":3})

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

template = """You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."
Do NOT use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question:str) -> dict:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    filled_prompt = prompt.invoke({
        "context":context,
        "question":question
    })
    response = llm.invoke(filled_prompt.text)
    answer = response.content
    return {
        "question": question,
        "answer": answer,
        "source_docs": docs

    }
question = "Where do college students stay when studying away from home?"

result = ask(question)

print(result["answer"])
