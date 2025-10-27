import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA  # ✅ Correct import for new version
import pickle
import os

with st.sidebar:
    st.title('Welcome to LLM Chat App')
    st.markdown('''
    # About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model
    ''')
    st.sidebar.write("\n" * 5)
    st.write('By Lochana M')

# Load environment variables
load_dotenv()

# ---------------- Main App ----------------
def main():
    st.header("Chat with PDF 📄")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(f"**Uploaded File:** {pdf.name}")

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Create or load FAISS vector store
        store_name = pdf.name[:-4]
        embeddings = OpenAIEmbeddings()

        if os.path.exists(store_name):
            VectorStore = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
            st.success(f"Loaded existing FAISS index: {store_name}")
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(store_name)
            st.success(f"FAISS index created and saved as: {store_name}")

        # Take user query
        query = st.text_input("Ask a question about your PDF:")

        # Run RetrievalQA chain
        if query:
            llm = ChatOpenAI(temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=VectorStore.as_retriever())
            response = qa_chain.run(query)

            st.subheader("Answer:")
            st.write(response)

# Run the app
if __name__ == '__main__':
    main()