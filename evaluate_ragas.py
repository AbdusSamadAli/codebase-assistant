from github import Github, Auth
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

def fetch_repo_files(repo_name, max_files=100):
    g = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN")))
    repo = g.get_repo(repo_name)
    documents = []
    skip_dirs = {"docs", "tests", "test", ".github", "node_modules", "examples"}

    def walk(path=""):
        if len(documents) >= max_files:
            return
        contents = repo.get_contents(path)
        for item in contents:
            if len(documents) >= max_files:
                return
            if item.type == "dir":
                if item.name.lower() not in skip_dirs:
                    walk(item.path)
            elif item.name.endswith(".py"):
                try:
                    code = item.decoded_content.decode("utf-8")
                    documents.append(Document(page_content=code, metadata={"source": item.name, "path": item.path}))
                except Exception:
                    pass

    walk("")
    return documents


print("Fetching repo...")
repo_name = "encode/httpx"
docs = fetch_repo_files(repo_name)
print(f"Fetched {len(docs)} files")

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=500, chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

template = """You are a helpful code assistant.
Answer the question using ONLY the code context below.
Always mention which file(s) the relevant code is from.
If the answer is not in the context, say "I don't know based on the provided code."

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)


def format_docs(docs):
    return "\n\n".join(f"[File: {d.metadata.get('source','unknown')}]\n{d.page_content}" for d in docs)


def ask(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)
    filled = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(filled.text)
    return {
        "question": question,
        "answer": response.content,
        "contexts": [d.page_content for d in docs]
    }

questions = [
    "How does httpx handle timeouts?",
    "What is the Client class used for?",
    "How does the async client work?",
    "What HTTP methods are supported?",
    "What is the purpose of the Transport class?",
]

print("\nRunning RAG pipeline...")
results = []
for q in questions:
    r = ask(q)
    results.append(r)
    print(f"Q: {q}")
    print(f"A: {r['answer'][:150]}...\n")

print("Running RAGAS evaluation...")

ragas_samples = [
    {"user_input": r["question"], "response": r["answer"], "retrieved_contexts": r["contexts"]}
    for r in results
]

eval_dataset = EvaluationDataset.from_list(ragas_samples)

ragas_llm = LangchainLLMWrapper(llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

scores = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n── RAGAS Scores ──────────────────")
print(scores)