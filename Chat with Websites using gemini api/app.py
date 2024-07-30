import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import os

os.environ['GOOGLE_API_KEY'] = ''
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key="")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create a prompt
system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that muje nhi pata.
----------------
{context}"""

human_template = "{question}"

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

def load_doc(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def create_vectorstore(doc):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = text_splitter.split_documents(doc[:50])
    vectorstore = FAISS.from_documents(text, embeddings)
    return vectorstore

st.header("Chat with Website")



url = ""      # Enter the url Of website
if 'docs' not in st.session_state:
    st.session_state.docs = load_doc(url)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = create_vectorstore(st.session_state.docs)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if 'chain' not in st.session_state:
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        chain_type='stuff',
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        verbose=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

input = st.chat_input("Enter the query")

if input:
    response = st.session_state.chain({"question": input, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append({"question": input, "answer": response['answer']})

for entry in st.session_state.chat_history:
    col1,col2= st.columns([1,3])
    with col1:
        st.write(f"**Question:** {entry['question']}")
    with col2:  
        st.write(f"**Answer:** {entry['answer']}")
