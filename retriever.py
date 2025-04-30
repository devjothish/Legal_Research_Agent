# retriever.py
import os
from dotenv import load_dotenv
import faiss
import pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

# Load environment variables
load_dotenv()

# Initialize Embedding
embedding_model = OpenAIEmbeddings()

def ingest_documents():
    loader = JSONLoader(file_path='./data/case_law.json', jq_schema='.cases[]', text_content=False)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(texts, embedding_model)
    vectordb.save_local('./vectorstore')

def retrieve_documents(query, k=5):
    vectordb = FAISS.load_local('./vectorstore', embedding_model, allow_dangerous_deserialization=True)
    docs = vectordb.similarity_search(query, k=k)
    return docs
