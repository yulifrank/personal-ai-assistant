from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()


class HybridRetriever(BaseRetriever):
    """
    Combines FAISS semantic search (60%) and BM25 keyword search (40%).
    Deduplicates by page_content and returns top-k unique results.
    """
    faiss_retriever: object
    bm25_retriever: object
    k: int = 4

    def _get_relevant_documents(self, query: str) -> List[Document]:
        semantic_docs = self.faiss_retriever.invoke(query)
        keyword_docs = self.bm25_retriever.invoke(query)

        seen = set()
        merged = []

        # Interleave: semantic first (higher weight), then keyword
        pairs = list(zip(semantic_docs, keyword_docs))
        remainder_semantic = semantic_docs[len(pairs):]
        remainder_keyword = keyword_docs[len(pairs):]

        for s_doc, k_doc in pairs:
            for doc in (s_doc, k_doc):
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        for doc in remainder_semantic + remainder_keyword:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[:self.k]


def load_documents(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def build_retriever(file_path: str):
    docs = load_documents(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # ── Semantic (FAISS) ───────────────────────────────────────────────────────
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 3})

    # ── Keyword (BM25) ────────────────────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # ── Hybrid ────────────────────────────────────────────────────────────────
    hybrid = HybridRetriever(
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        k=4,
    )

    return hybrid, len(chunks)


def get_retriever(file_path: str):
    return build_retriever(file_path)
