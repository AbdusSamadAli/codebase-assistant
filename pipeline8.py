from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset
from dotenv import load_dotenv
import os

load_dotenv()
dataset = load_dataset("rajpurkar/squad", split="train[:100]")
raw_texts = list(set([row["context"] for row in dataset]))

documents = [Document(page_content=text) for text in raw_texts]
splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chromadb_day7"
)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

template = """You are a helpful assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."
Do NOT use outside knowledge.

Context : {context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask(question : str) -> dict:
    docs = retriever.invoke(question)
    context = format_docs(docs)
    filled_prompt = prompt.invoke({
        "context": context,
        "question": question
    })
    response = llm.invoke(filled_prompt.text)
    return {
        "question": question,
        "answer" : response.content,
        "contexts" : [doc.page_content for doc in docs]
    }

questions = [
    "Where do college students stay when studying away from home?",
    "How many residence halls are there for undergraduates?",
    "What is the visiting policy in dormitories called?",
    "Do residence halls have social spaces for students?",
    "What percentage of undergraduates live on campus?"
]

results=[]
for q in questions:
    result = ask(q)
    results.append(result)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")

def evaluate_faithfulness(answer, context):
    """Check if the answer is grounded in the context."""
    eval_prompt = f"""You are an evaluator. Given a context and an answer, 
determine if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Is every claim in the answer directly supported by the context?
Respond with ONLY a number between 0 and 1, where:
1 = fully supported, no hallucination
0 = not supported at all, completely hallucinated
0.5 = partially supported

Score:"""
    
    response = llm.invoke(eval_prompt)
    try:
        score = float(response.content.strip().split()[0])
        return max(0, min(1, score))
    except:
        return None

def evaluate_relevancy(question, answer):
    """Check if the answer addresses the question."""
    eval_prompt = f"""You are an evaluator. Given a question and an answer,
determine how relevant the answer is to the question.

Question:
{question}

Answer:
{answer}

Does the answer directly address the question?
Respond with ONLY a number between 0 and 1, where:
1 = fully relevant, directly answers the question
0 = not relevant at all
0.5 = partially relevant

Score:"""
    
    response = llm.invoke(eval_prompt)
    try:
        score = float(response.content.strip().split()[0])
        return max(0, min(1, score))
    except:
        return None

faithfulness_scores = []
relevancy_scores = []

for r in results:
    context = "\n\n".join(r["contexts"])
    
    f_score = evaluate_faithfulness(r["answer"], context)
    r_score = evaluate_relevancy(r["question"], r["answer"])
    
    if f_score is not None:
        faithfulness_scores.append(f_score)
    if r_score is not None:
        relevancy_scores.append(r_score)
    
    print(f"Q: {r['question'][:60]}...")
    print(f"  Faithfulness: {f_score}")
    print(f"  Relevancy:    {r_score}\n")

avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
