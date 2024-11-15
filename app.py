import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Title
st.title("Intelligent Customer Support Chatbot")

# File uploader
uploaded_files = st.file_uploader("Upload Support Documents", accept_multiple_files=True, type=['pdf', 'txt', 'docx'])

# Load documents
if uploaded_files:
    def load_uploaded_documents(uploaded_files):
        documents = []
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(uploaded_file)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(uploaded_file)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                continue
            documents.extend(loader.load())
        return documents

    documents = load_uploaded_documents(uploaded_files)

    # Process documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Set up QA chain
    llm = OpenAI(model_name='gpt-4', temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")

    # User input
    user_input = st.text_input("Your Question:", key="input")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generate response
        response = qa_chain.run(user_input)

        # Append bot response
        st.session_state.messages.append({"role": "bot", "content": response})

        # Refresh to display new messages
        st.experimental_rerun()
else:
    st.write("Please upload support documents to begin.")
