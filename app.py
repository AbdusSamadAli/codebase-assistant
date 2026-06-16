import streamlit as st
from ingest import fetch_repo_files, chunk_documents, build_vectorstore
from rag_chain import build_chain

st.set_page_config(page_title="AI Codebase Assistant")
st.title("AI Codebase Assistant")
st.caption("Enter a public Github repo (owner/repo) and ask questions about its code")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = None
if "messages" not in st.session_state:
    st.session_state.messages = []

repo_name = st.text_input("Github repo", placeholder="owner/repo")

def parse_repo_input(text: str) -> str:
    """Accept both 'owner/repo' and full GitHub URLs."""
    text = text.strip().rstrip("/")
    if "github.com" in text:
        parts = text.split("github.com/")[-1]
        parts = parts.split("/")
        return f"{parts[0]}/{parts[1]}"
    return text

if st.button("load repo"):
    if repo_name:
        with st.spinner(f"fetching and indexing {repo_name}"):
            try:
                docs = fetch_repo_files(parse_repo_input(repo_name))
                if not docs:
                    st.error("no python files found in this repo")
                else:
                    chunks = chunk_documents(docs)
                    vectorstore,embeddings = build_vectorstore(chunks, repo_name)
                    retriever = vectorstore.as_retriever(search_kwargs={"k":4})
                    st.session_state.chain = build_chain(retriever)
                    st.session_state.repo_loaded = repo_name
                    st.success(f"loaded {len(docs)} files, {len(chunks)} chunks from {repo_name}")
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.warning("enter a repo name")

if st.session_state.chain:
    st.divider()
    st.subheader(f"chat with: {st.session_state.repo_loaded}")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
        
    question = st.chat_input("ask question about codebase")

    if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("thinking"):
                    answer = st.session_state.chain.invoke(question)
                    st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
            st.info("load a repo above to start chatting")
        