import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from pypdf import PdfReader

load_dotenv()

@st.cache_resource
def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

@st.cache_resource
def create_vectorstore(texts, _embeddings):
    return FAISS.from_texts(texts, _embeddings)

@st.cache_resource
def create_qa_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0)
    retriever = _vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return chain

def process_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        st.error("Formato não suportado. Use PDF ou TXT.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks

def main():
    st.title("RAG com LangChain + Gemini + Streamlit")

    uploaded_file = st.file_uploader("Faça upload do arquivo PDF ou TXT", type=["pdf", "txt"])
    if uploaded_file is not None:
        chunks = process_file(uploaded_file)
        if chunks:
            embeddings = load_embeddings()
            vectorstore = create_vectorstore(chunks, embeddings)
            chain = create_qa_chain(vectorstore)

            question = st.text_input("Digite sua pergunta:")
            if question:
                answer = chain.run(question)
                st.markdown(f"**Resposta:** {answer}")

if __name__ == "__main__":
    main()
