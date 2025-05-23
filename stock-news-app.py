import json
import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Load and parse news
@st.cache_data(show_spinner=True)
def load_news(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    docs = []
    for ticker, articles in raw.items():
        for article in articles:
            title = article['title']
            full_text = article['full_text']
            link = article.get('link', '')
            meta = {'ticker': ticker, 'title': title, 'link': link}
            docs.append(Document(page_content=full_text, metadata=meta))
    return docs

# Create FAISS vector store from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(texts, embeddings)

# Streamlit UI
st.set_page_config(page_title="Financial News Chat", layout="wide")
st.title("📈 Financial News Chat (RAG + GenAI)")

query = st.text_input("Ask a question about recent financial news:", "")

# Load vector store once
if "vectordb" not in st.session_state:
    with st.spinner("Loading and embedding news articles..."):
        news_docs = load_news("stock_news.json")
        st.session_state.vectordb = create_vector_store(news_docs)

# Query response
if query:
    retriever = st.session_state.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=retriever)
    response = qa_chain.run(query)
    st.markdown("### 🧠 Answer")
    st.write(response)

    # Optional: show sources
    st.markdown("---")
    st.markdown("### 📚 Retrieved Documents")
    docs = retriever.get_relevant_documents(query)
    for doc in docs:
        st.markdown(f"**{doc.metadata['title']}**")
        st.markdown(f"{doc.page_content[:300]}...")
        st.markdown(f"[Read more]({doc.metadata.get('link', '#')})")