import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate 


# Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

db_file_path = 'Faiss_index'

def load_doc(doc):
    try:
        loader = PyPDFLoader(doc)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

def create_vectorstore(doc):
    try:
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=200)
        text = text_splitter.split_documents(doc)
        vectorstore = FAISS.from_documents(text, embeddings)
        vectorstore.save_local(db_file_path)
        st.success("Vector store created successfully")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def create_retrieval_chain():
    try:
        db = FAISS.load_local(db_file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(score_threshold=0.7)
        prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. If you don't know the answer,
             just say that you dont know. Don't try to make up an answer. {context} Question: {question} Answer:"""

        )
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            input_key="query",
            memory=memory,
            chain_type_kwargs={"prompt": prompt}
        )
        return chain
    except Exception as e:
        st.error(f"Error creating retrieval chain: {e}")
        return None

memory = ConversationBufferMemory(memory_key="chat_history")
doc = "lstm.pdf"
if doc:
    docs = load_doc(doc)
    create_vectorstore(docs)
    chain = create_retrieval_chain()
chat_history = []
input = st.text_input("Enter the query")
if input and chain:
    response = chain.invoke({"query":input,"chat_history":chat_history})
    st.write(response['result'])