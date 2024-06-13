__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import sqlite3
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
# from langchain.embeddings.groq import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader

# Input .txt file
# Format file
# Split file
# Create embeddings
# Store embeddings in vector store
# Input query
# Run QA chain
# Output

def generate_response(file, groq_api_key, query):
    #format file
    reader = PdfReader(file)
    formatted_document = []
    for page in reader.pages:
        formatted_document.append(page.extract_text())
    #split file
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.create_documents(formatted_document)
    #create embeddings
    embeddings = HuggingFaceEmbeddings()
    #load to vector database
    #store = Chroma.from_documents(texts, embeddings)

    store = FAISS.from_documents(docs, embeddings)
    
    #create retrieval chain
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0, api_key=groq_api_key, model="llama3-70b-8192"),
        chain_type="stuff",
        retriever=store.as_retriever()
    )
    #run chain with query
    return retrieval_chain.invoke(query)

st.set_page_config(
    page_title="Q&A from a long PDF Document"
)
st.title("Q&A from a long PDF Document")

uploaded_file = st.file_uploader(
    "Upload a .pdf document",
    type="pdf"
)

query_text = st.text_input(
    "Enter your question:",
    placeholder="Write your question here",
    disabled=not uploaded_file
)

result = []
with st.form(
    "myform",
    clear_on_submit=True
):
    groq_api_key = st.text_input(
        "Groq API Key:",
        type="password",
        disabled=not (uploaded_file and query_text)
    )
    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )
    if submitted and groq_api_key.startswith("gsk_"):
        with st.spinner(
            "Wait, please. I am working on it..."
            ):
            response = generate_response(
                uploaded_file,
                groq_api_key,
                query_text
            )
            result.append(response)
            del groq_api_key
            
if len(result):
    st.info(response)