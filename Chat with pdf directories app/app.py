from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import streamlit as st
import os

os.environ['GOOGLE_API_KEY'] = ''

llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key="")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

prompt_template = """
Use the following pieces of context to answer the question,
if you don't know the answer, just say that i don't know, don't try to make up an answer.
{context}
Question: {question}
"""
PROMPT = ChatPromptTemplate.from_template(prompt_template)

def create_vectorstore():
    docs = PyPDFDirectoryLoader("pdfs").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore = vectorstore.as_retriever()
    return vectorstore

def conversation_chain(vectorstore, input):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = chain({"query": input})
    return result['result']

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'query' not in st.session_state:
    st.session_state.query = ""

if 'result' not in st.session_state:
    st.session_state.result = ""

st.title("Chat With Multiple PDFs")

if st.button("Load Documents"):
    with st.spinner("processing..."):
        st.session_state.vectorstore = create_vectorstore()
        st.success("Documents Loaded Successfully")


input = st.text_input("Enter the Query")

if st.session_state.vectorstore and input:
    with st.spinner("processing..."):
        if input != st.session_state.query:
            st.session_state.query = input
            st.session_state.result = conversation_chain(st.session_state.vectorstore, input)

if st.session_state.result:
    st.write(st.session_state.result)
