from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
import streamlit as st
import os

st.set_page_config(page_title="Conversational Chatbot", page_icon="💬")

st.title("💬 A Langchain Chatbot")
st.caption("🚀 A chatbot powered by Google Generative AI")

# User input Google API key
api_key = st.sidebar.text_input("Enter GOOGLE API Key:", key="api_key", type="password")

chat = None  # Define chat object

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key
    try:
        chat = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.5)  # Create ChatGoogleGenerativeAI instance
    except Exception as e:
        st.error(f"Error initializing ChatGoogleGenerativeAI: {e}")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        SystemMessage(content="Hello, I am a chatbot to help you with your queries. How can I help you!")
    ]

st.write(st.session_state['messages'][0].content)

def get_google_response(query):
    st.session_state['messages'].append(HumanMessage(content=query))
    try:
        answer = chat.invoke(st.session_state['messages'])
        st.session_state['messages'].append(AIMessage(content=answer.content))
        return answer.content if answer else "Sorry, I couldn't generate a response."
    except Exception as e:
        st.session_state['messages'].append(AIMessage(content=f"Error: {e}"))
        return f"Error: {e}"

description = "Enter your query here..."
input_text = st.text_input("Input: ", value="", key="input", placeholder=description)

if st.button("Submit"):
    if chat is None:
        st.warning("Please input your Google API key.")
    elif not input_text.strip():
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner('Generating...'):
                response = get_google_response(input_text)
                st.write(response)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
