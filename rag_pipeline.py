from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from config import MODEL_CHAT, MODEL_EMBEDDING
import streamlit as st

load_dotenv()
@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDING)

@st.cache_resource
def create_vectorstore(texts, _embeddings):
    return FAISS.from_texts(texts, _embeddings)

@st.cache_resource
def create_qa_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(model=MODEL_CHAT, temperature=0)
    retriever = _vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

def process_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)
