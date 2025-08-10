import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from rag_pipeline import load_embeddings, create_vectorstore, create_qa_chain, process_text
from utils import extract_text_from_pdf

def main():
    st.title("RAG com LangChain + Gemini + Streamlit")

    uploaded_file = st.file_uploader("Faça upload do arquivo PDF ou TXT", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.getvalue().decode("utf-8")

        chunks = process_text(text)
        embeddings = load_embeddings()
        vectorstore = create_vectorstore(chunks, embeddings)
        chain = create_qa_chain(vectorstore)

        question = st.text_input("Digite sua pergunta:")
        if question:
            answer = chain.run(question)
            st.markdown(f"**Resposta:** {answer}")

if __name__ == "__main__":
    main()
