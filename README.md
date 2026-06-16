# 🤖 AI Codebase Assistant
<img width="901" height="404" alt="rag-repo-white" src="https://github.com/user-attachments/assets/357a73dd-10d4-4f8f-98ba-4ecf1d8ddd3c" />


Ask natural language questions about any public GitHub repository using RAG.

## Features
- Accepts any public GitHub repo URL or owner/repo format
- Language-aware Python code chunking
- File-level source attribution in answers
- Full LCEL pipeline (RunnableParallel + RunnableSequence)
- Evaluated using RAGAS: Faithfulness 0.59, Answer Relevancy 0.75

## Tech Stack
LangChain · ChromaDB · Groq (Llama 3.3) · HuggingFace · Streamlit · PyGithub

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`


## Live Demo
https://codebase-assistant-llrvo89tngdveg33mqg7ds.streamlit.app/


