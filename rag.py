from dotenv import load_dotenv
from pathlib import Path
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHUNK_SIZE = 1000
VECTOR_STORE_DIR = "vector_store"

llm = None
vector_store = None


def initialize_components():
    global llm


def process_urls(urls):
    global vector_store

    if not llm:
        initialize_components()

    # ✅ Proper loader (handles headers better than UnstructuredURLLoader)
    loader = WebBaseLoader(
        urls,
        requests_kwargs={
            "headers": {
                "User-Agent": "Mozilla/5.0"
            }
        }
    )

    data = loader.load()

    yield f"✅ Loaded {len(data)} documents"

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)

    yield f"✅ Split into {len(docs)} chunks"

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store in Chroma
    vector_store = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )

    yield "✅ Data stored successfully!"


def generate_answer(query):
    global vector_store

    if not vector_store:
        raise RuntimeError("❌ Process URLs first!")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    result = chain({"question": query}, return_only_outputs=True)

    return result["answer"], result["sources"]
