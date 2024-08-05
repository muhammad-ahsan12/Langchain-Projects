from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import streamlit as st
from PyPDF2 import PdfReader

# Set up the environment
os.environ['GOOGLE_API_KEY'] = ''

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Set up the Streamlit app
st.set_page_config(page_title="Language Translator and PDF Summarizer", page_icon="🌐")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Language Translator", "PDF Summarizer"])

if app_mode == "Language Translator":
    # Define the prompt template for translation
    translation_template = """you are a helpful assistant,
    translate the following {speech} into {Language}."""
    translation_prompt = PromptTemplate(
        input_variables=["speech", "Language"],
        template=translation_template
    )

    st.title("🌐 Language Translator")
    st.write("Welcome! Translate your speech into the language of your choice. 🌍")

    # Input speech and language
    speech = st.text_area("✏️ Input your speech")
    uploaded_file = st.file_uploader("Or upload a PDF file", type="pdf")
    language = st.selectbox("🌍 Select Language", ["Urdu", "Spanish", "French", "German", "Chinese","Hindi"])

    # Spinner for loading state
    if st.button("Translate"):
        with st.spinner("Translating... 🔄"):
            if uploaded_file is not None:
                # Read the PDF file
                reader = PdfReader(uploaded_file)
                speech = ""
                for page in reader.pages:
                    speech += page.extract_text()

            # Create LLMChain for translation
            translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
            # Run the translation
            translated_text = translation_chain.run({"speech": speech, "Language": language})
            st.success("Translation completed! ✅")
            st.write(translated_text)

elif app_mode == "PDF Summarizer":
    # Define the prompt template for summarization
    summarization_template = """you are a helpful assistant,
    Summarize the following {speech}."""
    summarization_prompt = PromptTemplate(
        input_variables=["speech"],
        template=summarization_template
    )

    st.title("📄 PDF Summarizer")
    st.write("Welcome! Upload a PDF document and get a summary. 📚")

    # File uploader
    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if st.button("Summarize"):
        with st.spinner("Reading and summarizing the document... 🔄"):
            # Read the PDF file
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            
            # Create LLMChain for summarization
            summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
            # Summarize the text
            summarized_text = summarization_chain.run({"speech": text})
            
            st.success("Summary completed! ✅")
            st.write(summarized_text)

