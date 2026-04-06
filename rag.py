from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()


def load_documents(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def build_vector_store(file_path: str):
    docs = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(file_path: str):
    vector_store = build_vector_store(file_path)
    return vector_store.as_retriever(search_kwargs={"k": 3})
