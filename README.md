# 🤖 AI Codebase Assistant

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
3. Copy `.env.example` to `.env` and fill in your keys
4. Run: `streamlit run app.py`

## Evaluation
RAGAS evaluation run on encode/httpx:
- Faithfulness: 0.59
- Answer Relevancy: 0.75
- 
## Live Demo
https://codebase-assistant-llrvo89tngdveg33mqg7ds.streamlit.app/


