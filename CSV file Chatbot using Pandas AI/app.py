import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set the Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"

# Set up the Streamlit interface
st.title("Pandas AI Query Engine with Gemini 🧠")
st.write("Upload your CSV file and ask questions!")

# Initialize session state for file and query
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# Sidebar for file upload and load button
with st.sidebar:
    st.header("Upload CSV File 📁")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file and st.sidebar.button("Load File"):
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File loaded successfully! 🎉")
        st.session_state.df = df  # Save the dataframe in session state

# Display dataset preview if file is uploaded
if st.session_state.uploaded_file:
    df = st.session_state.df
    st.write("### Dataset Preview 📊", df)

    # Input for user query in the main area
    st.session_state.user_query = st.text_input("Ask a question or request a graph about the dataset 📝", st.session_state.user_query)

    # Generate button
    if st.button("Generate Answer 💡"):
        if st.session_state.user_query:
            # Initialize LangChain with Gemini LLM
            llm = ChatGoogleGenerativeAI(model="gemini-pro") 
            chain = SmartDataframe(df, config={"llm": llm})

            # Generate the answer using PandasAI
            answer = chain.chat(st.session_state.user_query)
            st.write("### Answer 💬", answer)
        else:
            st.write("Please enter a question before generating an answer.")
