from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from ingest import fetch_repo_files, chunk_documents, build_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()

def format_docs(docs):
    """Format retrieved chunks with source file attribution"""
    formatted=[]
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[File: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_chain(retriever):
    """Build full LCEL RAG chain"""
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name = "llama-3.3-70b-versatile")

    template = """You are a helpful code assistant.
Answer the question using ONLY the code context below.
Always mention which file(s) the relevant code is from.
If the answer is not in the context, say "I don't know based on the provided code."

Context: {context}
Question: : {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    chain = (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x:x)})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    repo_name = "psf/requests"
    print("loading existing vectorstore")
    docs = fetch_repo_files(repo_name)
    chunks = chunk_documents(docs)
    vectorstore, embeddings = build_vectorstore(chunks, repo_name)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("building LCEL chain")
    chain = build_chain(retriever)

    question = "How are HTTP headers handled in requests?"
    print(f"\nQuestion:{question}\n")
    answer = chain.invoke(question)
    print("Answer:")
    print(answer)

    print("\n Source chunks retrieved:")
    for doc in retriever.invoke(question):
        print(f" - {doc.metadata['source']}")
