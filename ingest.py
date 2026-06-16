from github import Github, Auth
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def fetch_repo_files(repo_name:str, max_files:int=100):
    """Fetch all .py files from a public Github repo"""
    g = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN")))
    repo = g.get_repo(repo_name)

    documents = []

    skip_dirs = {"docs", "tests", "test", ".github", "node_modules", "examples"}

    def walk(path=""):
        if len(documents) >= max_files:
              return
        contents = repo.get_contents(path)
        for item in contents:
            if item.type =="dir":
                if item.name.lower() not in skip_dirs:
                    walk(item.path)
            elif item.name.endswith(".py"):
                try:
                    code = item.decoded_content.decode("utf-8")
                    documents.append(
                        Document(page_content=code, metadata={"source": item.name, "path": item.path})
                    )
                except Exception:
                    pass

    walk("")
    return documents

def chunk_documents(documents):
            """split code documents using Python-aware splitter."""
            splitter = RecursiveCharacterTextSplitter.from_language(
                 language=Language.PYTHON,
                 chunk_size=500,
                 chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)
            return chunks
    
def build_vectorstore(chunks, repo_name: str):
         """Embed chunks and store in ChromaDB"""
         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")

         vectorstore = Chroma.from_documents(
              documents=chunks,
              embedding=embeddings
         )
         return vectorstore, embeddings
    
if __name__ == "__main__":
         repo_name = "psf/requests"
         print(f"fetching files from {repo_name}")
         docs = fetch_repo_files(repo_name)
         print(f"Fetched {len(docs)} Python files")
         print(f"sample file : {docs[0].metadata['path']}\n")

         print("chunking")
         chunks = chunk_documents(docs)
         print(f"total chunks: {len(chunks)}")
         print(f"sample chunk metadata: {chunks[0].metadata}")
         print(f"sample chunk content:\n{chunks[0].page_content[:200]}\n")

         print("building vector store")
         vectorstore, embeddings = build_vectorstore(chunks, repo_name)
         print(f"vectorstore ready: {vectorstore._collection.count()} chunks stored")
               
    


