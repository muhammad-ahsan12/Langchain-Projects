import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import pandas as pd
import tempfile
import time

# Set Google API Key
os.environ['GOOGLE_API_KEY'] = ""

def create_agent_from_csv(file_path):
    llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0)
    agent = create_csv_agent(
        llm,
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent

def main():
    st.title("CSV File Question Answering")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        csv_file_path = temp_file.name

        st.write(f"File {uploaded_file.name} uploaded successfully!")

        agent = create_agent_from_csv(csv_file_path)

        query = st.text_input("What would you like to know?")

        if query:
            with st.spinner('Getting the answer...'):
                start = time.time()
                answer = agent.invoke(query)
                end = time.time()

            st.write(f"**Question:** {query}")
            st.write(f"**Answer:** {answer}")
            st.write(f"*Answer took {round(end - start, 2)} seconds*")

if __name__ == "__main__":
    main()
