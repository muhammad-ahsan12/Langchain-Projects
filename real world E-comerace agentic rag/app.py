import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent
from langchain.tools import BaseTool
import os

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyA0S7F21ExbBnR06YXkEi7aj94nWP5kJho"
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key="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"
)

class MyCustomTool(BaseTool):
    name: str = "ConversationTool"
    description: str = """Welcome to Bob, your personal shopping assistant! 🎉 Greeting with user attractively!
    I'm thrilled to help you find the perfect products tailored just for you.
Whether you're here to browse or ready to buy, I'm here to make your experience enjoyable and effortless.

If you're just here for a chat today, that's perfectly fine too—I'm always here for a friendly conversation and happy you stopped by! 🌟

But if you're on the hunt for something special, I'm ready to help you discover top-quality products across various categories. From affordable prices to unbeatable benefits, I'll ensure you understand why each item is a fantastic choice for you. 💡

Thanks for spending time with me today! Whenever you're ready, we can explore together. Remember, I'm Bob, your friendly shopping guide, always here to assist you. Let's get started whenever you are!

I'll also store the before input and output😊
"""
    def _run(self, argument: str) -> str:
        # Craft a friendly and engaging response
        return argument


def button_tool(input: str) -> str:
    website_url = "put the link of website here"  # This should be extracted or dynamic based on input
    return f"""
    <div style="text-align: center; margin-top: 20px;">
        <a href="{website_url}" target="_blank">
            <button style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">
                Go to Website
            </button>
        </a>
    </div>
    """

# Initialize Tools
convo_tool = MyCustomTool()
button_tool_instance = Tool(
    name="ButtonTool",
    description="""Engage users by providing a direct link to purchase or explore products on a 
    website with a stylish and clickable button. Provide a button with a link to 
    the website when the user wants to buy products.
    Example: If the user wants to buy a laptop, give the button with a link to the Amazon laptop page.
    If the user wants to buy shoes, give the button with a link to the Amazon shoes page, etc.
    and say "this is our website button Click here to buy products".
    """,
    func=button_tool
)

# Initialize Agent
agent = initialize_agent(
    llm=llm,
    tools=[convo_tool, button_tool_instance],
    verbose=True,
    handle_parsing_errors=True
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

# Display a beautiful greeting message
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #007bff;">Hello there! 🌟</h2>
        <p>I'm Bob, your friendly shopping assistant. How can I help you today? Whether you're here to explore or to make a purchase, I'm here to make your experience delightful. 😊</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User input
text = st.chat_input("Ask Bob anything about products or shopping...")

# Handle user input and generate response
if text:
    with st.spinner("Bob is thinking..."):
        try:
            response = agent.run(text)
            st.session_state.chat_history.append({"question": text, "response": response})

        except Exception as e:
            st.error(f"Sorry, something went wrong: {e}")

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
    st.markdown(button_tool(text), unsafe_allow_html=True)
