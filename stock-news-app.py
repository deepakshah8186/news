import json
import os
from typing import List

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Constants
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEARCH_K = 4
NEWS_FILE = "stock_news.json"

def initialize_app() -> None:
    """Initialize Streamlit app configuration."""
    st.set_page_config(page_title="Financial News Chat", layout="wide")
    st.title("Financial News Chat")

def get_api_key() -> str:
    """Retrieve OpenAI API key from environment"""
    return os.getenv("OPENAI_API_KEY")

@st.cache_data(show_spinner="Loading news data...")
def load_news(filepath: str) -> List[Document]:
    """Load and parse news articles into Document objects.

    Args:
        filepath: Path to JSON file containing news data

    Returns:
        List of Document objects with article content and metadata
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        return [
            Document(
                page_content=article["full_text"],
                metadata={
                    "ticker": ticker,
                    "title": article["title"],
                    "link": article.get("link", "#")
                }
            )
            for ticker, articles in raw_data.items()
            for article in articles
        ]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading news data: {str(e)}")
        return []

@st.cache_resource(show_spinner="Embedding documents...")
def create_vector_store(_docs: List[Document], api_key: str) -> FAISS:
    """Create FAISS vector store from documents.

    Args:
        _docs: List of Document objects to embed (underscore prefix avoids hashing)
        api_key: OpenAI API key for embeddings

    Returns:
        FAISS vector store containing document embeddings
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    split_docs = splitter.split_documents(_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(split_docs, embeddings)

def display_results(answer: str, relevant_docs: List[Document]) -> None:
    """Display QA results and relevant documents.

    Args:
        answer: Generated answer to display
        relevant_docs: List of relevant documents
    """
    st.markdown("## Answer")
    st.write(answer)

    st.markdown("---")
    st.markdown("### Retrieved Documents")

    for doc in relevant_docs:
        title = doc.metadata.get('title', 'Untitled')
        content = doc.page_content[:300]
        link = doc.metadata.get('link', '#')

        st.markdown(f"**{title}**")
        st.markdown(f"{content}...")
        st.markdown(f"[Read more]({link})")

def main():
    initialize_app()
    api_key = get_api_key()

    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return

    # Initialize vector store in session state
    if "vectordb" not in st.session_state:
        news_docs = load_news(NEWS_FILE)
        if news_docs:  # Only proceed if documents were loaded successfully
            st.session_state.vectordb = create_vector_store(news_docs, api_key)
        else:
            return

    # User query interface
    query = st.text_input("Ask a question about recent financial news:")

    if not query:
        return

    # Process query
    retriever = st.session_state.vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": DEFAULT_SEARCH_K}
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=api_key),
        chain_type="stuff",
        retriever=retriever
    )

    with st.spinner("Generating answer..."):
        try:
            answer = qa.run(query)
            relevant_docs = retriever.get_relevant_documents(query)
            display_results(answer, relevant_docs)
        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")

if __name__ == "__main__":
    main()