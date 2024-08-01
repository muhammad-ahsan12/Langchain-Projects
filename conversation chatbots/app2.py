import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import os

# Prompt template
prompt_template = """
Use the following pieces of context to answer the question,
if you don't know the answer, just say that I don't know, don't try to make up an answer.
{context}
Question: {question}
"""
PROMPT = ChatPromptTemplate.from_template(prompt_template)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Page configuration
st.set_page_config(page_title="Conversational Chatbot", page_icon="💬")

# Title and caption
st.title("💬 Chat With BOB")
st.caption("🚀 A chatbot powered by Google Generative AI")

# Sidebar for API key input
with st.sidebar:
    key = st.text_input("Enter GOOGLE API KEY", type="password")
    os.environ['GOOGLE_API_KEY'] = key
    # Optional: clear conversation history button
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
text = st.chat_input("Ask a Question")

# Submit button

if text:
    with st.spinner("Generating..."):
        llm = GoogleGenerativeAI(model='gemini-pro')
        response = llm(text)
            # Save the question and response to session state
        st.session_state.chat_history.append({"question": text, "response": response})

# Display conversation history
for chat in st.session_state.chat_history:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: ; padding: 10px; border-radius: 5px; max-width: 70%;">
                 {chat['question']} 🤓
            </div>
        </div>
        <div style="display: flex; justify-content: flex-start; margin-top: -10px; margin-bottom: 20px;">
            <div style="background-color: ; padding: 10px; border-radius: 5px; max-width: 70%;">
                🤖 {chat['response']} 
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
