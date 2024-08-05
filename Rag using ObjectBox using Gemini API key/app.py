import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_objectbox.vectorstores import ObjectBox

os.environ['GOOGLE_API_KEY'] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"

prompt_template = """
Use the following pieces of context to answer the question,
if you don't know the answer, just say that I don't know, don't try to make up an answer.
{context}
Question: {question}
"""
PROMPT = ChatPromptTemplate.from_template(prompt_template)

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro",temperature=0.2)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        st.session_state.loader = PyPDFDirectoryLoader("pdfs")
        st.session_state.doc = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.text = st.session_state.text_splitter.split_documents(st.session_state.doc)
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.text, st.session_state.embeddings, embedding_dimensions=768)
        # st.session_state.vectors = FAISS.from_documents(st.session_state.text, st.session_state.embeddings)
        
if st.button("Load Documents"):
    vector_embeddings()
    st.success("Vector Loaded Successfully")

input_query = st.text_input("Enter query")
chat_history = []
if input_query:
    # documents_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = st.session_state.vectors.as_retriever()
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    response = conversational_chain({"question": input_query,"chat_history":chat_history})
    st.write(response['answer'])
