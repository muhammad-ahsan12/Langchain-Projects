import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import  CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

def json_retriever():
    loader = CSVLoader(file_path='a.csv')
    data = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstores = FAISS.from_documents(data, embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstores.as_retriever()
                                        )
    PDF_tools = Tool(
        name="CSVSearchTool",
        description="Efficiently retrieve and provide accurate answers to user queries by searching through product data in a CSV file.",
        func=chain.run
    )
    return PDF_tools

os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key="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"
)

def button_tool(input: str) -> str:
    # Assuming that the input contains the link to the website
    website_url = "https://www.amazon.com/s?k=shoes&ref=nb_sb_noss_2"  # This should be extracted or dynamic based on input
    return f"""
    <div style="text-align: center; margin-top: 20px;">
        <a href="{website_url}" target="_blank">
            <button style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Go to Website
            </button>
        </a>
    </div>
    """
# Tools
Button_tool = Tool(
    name="ButtonTool",
    description="Engage users by providing a direct link to purchase or explore products on a website with a stylish and clickable button.",
    func=button_tool
)
PDF_tools = json_retriever()

agent = initialize_agent(
    llm=llm,
    tools=[PDF_tools, Button_tool],
    verbose=True,
)

# Initialize session state for chat history and button visibility
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_button' not in st.session_state:
    st.session_state.show_button = False

# Streamlit app layout
st.set_page_config(page_title="Bob - Your Shopping Assistant", page_icon="🛍️")
st.title("🛍️ Chat With Bob")
st.caption("🎉 Your Personal Shopping Assistant")

# User input
text = st.chat_input("Ask Bob anything about products or shopping...")

# Handle user input and generate response
if text:
    with st.spinner("Bob is thinking..."):
        response = agent.run(text)
        st.session_state.chat_history.append({"question": text, "response": response})

        # Check if the response should include a button
        if "button" in response.lower() or "go to website" in response.lower():
            st.session_state.show_button = True
        else:
            st.session_state.show_button = False

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
                🛒 Bob: {chat['response']} 
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display the button if applicable
if st.session_state.show_button:
    pass
